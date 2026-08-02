# Feature: HTTP Server

**Module:** `src/sutra/server/app.py`  
**Framework:** FastAPI  
**Install:** `pip install -e ".[server]"` (`fastapi>=0.110`, `uvicorn[standard]>=0.29`)

## Purpose

Exposes the harness over HTTP as a streaming SSE API, enabling browser frontends, mobile apps, and backend services to drive agent sessions, receive real-time token streams, and resolve permission requests remotely.

## Current Behavior

### Startup

```bash
uvicorn sutra.server.app:app --reload
```

The app has no built-in harness factory. Callers must register a pre-built harness per session from their own bootstrap code:

```python
from sutra.server.app import register_harness
from sutra.core.harness import AsynchronousHarnessLoop

harness = AsynchronousHarnessLoop(model_client=..., tool_registry=..., budget=...)
register_harness("session-abc", harness)
```

Sessions are stored in a module-level `dict[str, AsynchronousHarnessLoop]` (`_HARNESSES`) for the lifetime of the process.

### Endpoints

#### `POST /chat`

**Request body:**
```json
{ "session_id": "session-abc", "message": "Transfer $4500 to vendor ACC-2002." }
```

**Response:** `text/event-stream` — streams all `SSEEvent.to_sse()` frames from `harness.run(message)` until the final `done` event.

**Errors:** `404` if `session_id` is not registered.

#### `POST /permissions/{request_id}/resolve?session_id=...`

**Request body:**
```json
{ "approved": true, "actor": "alice" }
```

**Response:** `text/event-stream` — streams events from `harness.resume_after_permission(request_id, approved=..., actor=...)`, including `permission_resolved`, subsequent `TOKEN`/`TOOL_RESULT` events, and a final `done`.

Designed to be called by a UI after a human clicks "Approve" in response to a `permission_request` SSE event.

#### `GET /state/{session_id}`

**Response:** JSON — `harness.state.to_dict()` (full `HarnessState` snapshot).

Useful for:
- Debugging
- Reconstructing the permission request details if the client missed the SSE event
- Health monitoring dashboards

### Session Registry

```python
_HARNESSES: Dict[str, AsynchronousHarnessLoop] = {}

def register_harness(session_id: str, harness: AsynchronousHarnessLoop) -> None: ...
def _get_harness(session_id: str) -> AsynchronousHarnessLoop: ...  # raises 404
```

No TTL, no eviction — sessions live until the process restarts.

## Enhancement Ideas

- **Harness factory endpoint**: `POST /sessions` with a JSON body describing the desired toolkit, model, and budget — auto-constructs and registers a harness, returns the `session_id`. Removes the need for caller-side bootstrap code.
- **Session lifecycle management**: `DELETE /sessions/{session_id}` to explicitly destroy a session; TTL-based expiry for idle sessions.
- **Authentication / API keys**: add a FastAPI middleware layer requiring an `Authorization: Bearer <key>` header; tie keys to allowed budget limits.
- **CORS configuration**: add configurable CORS headers so browser-based UIs on a different origin can connect to the SSE stream.
- **WebSocket transport**: offer `WS /chat/{session_id}` as an alternative to SSE for bidirectional communication (client can send follow-up messages mid-stream).
- **Multi-process session store**: replace `_HARNESSES` dict with a Redis-backed session store so the server can run with multiple workers and sticky routing isn't required.
- **Reconnect support**: store the SSE event log per session so clients can reconnect with `?last_event_id=...` and replay missed frames.
- **Health endpoint**: `GET /health` returning `{"status": "ok", "sessions": N}` for load balancer probes.
- **Budget endpoint**: `GET /budget/{session_id}` returning `Budget.snapshot()` for monitoring dashboards.
- **Pending permissions list**: `GET /permissions?session_id=...` returning all open permission requests for a session, so a UI can display them without relying on SSE history.
- **OpenAPI docs**: FastAPI auto-generates `/docs`; add request/response examples for each endpoint using Pydantic's `model_config`.
