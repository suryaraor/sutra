from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from sutra.core.exceptions import ModelInvocationError
from sutra.models.base import ModelClient, StreamChunk


class OpenAIModelClient(ModelClient):
    """Thin async wrapper around the official `openai` SDK (Chat Completions)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", max_tokens: int = 4096) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for OpenAIModelClient. "
                "Install with `pip install sutra[openai]`."
            ) from exc

        self._client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    @staticmethod
    def _to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    async def stream(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[StreamChunk]:
        oai_messages = [{"role": "system", "content": system_prompt}, *messages]
        tool_call_buffers: Dict[int, Dict[str, Any]] = {}

        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=oai_messages,
                tools=self._to_openai_tools(tools) if tools else None,
                stream=True,
                stream_options={"include_usage": True},
            )

            finish_reason = "stop"
            usage: Optional[Dict[str, int]] = None

            async for event in stream:
                if event.usage:
                    usage = {
                        "input_tokens": event.usage.prompt_tokens,
                        "output_tokens": event.usage.completion_tokens,
                    }
                if not event.choices:
                    continue
                choice = event.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if delta.content:
                    yield StreamChunk(delta_text=delta.content)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        buf = tool_call_buffers.setdefault(tc.index, {"id": tc.id, "name": "", "arguments": ""})
                        if tc.id:
                            buf["id"] = tc.id
                        if tc.function and tc.function.name:
                            buf["name"] += tc.function.name
                        if tc.function and tc.function.arguments:
                            buf["arguments"] += tc.function.arguments

            finalized = []
            for buf in tool_call_buffers.values():
                try:
                    args = json.loads(buf["arguments"]) if buf["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                finalized.append({"id": buf["id"], "name": buf["name"], "input": args})

            yield StreamChunk(
                finished=True,
                finish_reason="tool_use" if finalized else finish_reason,
                tool_call_delta={"finalized": finalized} if finalized else None,
                usage=usage or {"input_tokens": 0, "output_tokens": 0},
            )
        except Exception as exc:  # noqa: BLE001 - normalize every provider failure to one error type
            raise ModelInvocationError(f"OpenAI stream failed: {exc}") from exc
