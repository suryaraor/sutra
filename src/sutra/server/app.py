from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sutra.core.harness import AsynchronousHarnessLoop

app = FastAPI(title="Sutra Agent Harness")

# Sessions are registered by whatever bootstrap code constructs each
# session's AsynchronousHarnessLoop (with its own model client, tools,
# subagents, and budget). This registry just maps a session id to its
# live harness instance for the lifetime of the process.
_HARNESSES: Dict[str, AsynchronousHarnessLoop] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class PermissionResolution(BaseModel):
    approved: bool
    actor: str = "operator"


def register_harness(session_id: str, harness: AsynchronousHarnessLoop) -> None:
    """Call this from your own bootstrap code once a harness is built for a session."""
    _HARNESSES[session_id] = harness


def _get_harness(session_id: str) -> AsynchronousHarnessLoop:
    harness = _HARNESSES.get(session_id)
    if harness is None:
        raise HTTPException(status_code=404, detail=f"No harness session '{session_id}'.")
    return harness


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    harness = _get_harness(req.session_id)

    async def event_source():
        async for event in harness.run(req.message):
            yield event.to_sse()

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/permissions/{request_id}/resolve")
async def resolve_permission(request_id: str, body: PermissionResolution, session_id: str) -> StreamingResponse:
    harness = _get_harness(session_id)

    async def event_source():
        async for event in harness.resume_after_permission(request_id, approved=body.approved, actor=body.actor):
            yield event.to_sse()

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.get("/state/{session_id}")
async def get_state(session_id: str) -> Dict[str, Any]:
    harness = _get_harness(session_id)
    return harness.state.to_dict()
