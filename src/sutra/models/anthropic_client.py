from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from sutra.core.exceptions import ModelInvocationError
from sutra.models.base import ModelClient, StreamChunk


class AnthropicModelClient(ModelClient):
    """Thin async wrapper around the official `anthropic` SDK."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-5", max_tokens: int = 4096) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is required for AnthropicModelClient. "
                "Install with `pip install sutra[anthropic]`."
            ) from exc

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    async def stream(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[StreamChunk]:
        try:
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=messages,
                tools=tools or None,
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if getattr(delta, "type", None) == "text_delta":
                            yield StreamChunk(delta_text=delta.text)
                        elif getattr(delta, "type", None) == "input_json_delta":
                            yield StreamChunk(tool_call_delta={"partial_json": delta.partial_json})
                    elif event.type == "content_block_start" and event.content_block.type == "tool_use":
                        yield StreamChunk(
                            tool_call_delta={
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "start": True,
                            }
                        )

                final_message = await stream.get_final_message()
                tool_calls = [
                    {"id": b.id, "name": b.name, "input": b.input}
                    for b in final_message.content
                    if b.type == "tool_use"
                ]
                yield StreamChunk(
                    finished=True,
                    finish_reason=final_message.stop_reason or "stop",
                    tool_call_delta={"finalized": tool_calls} if tool_calls else None,
                    usage={
                        "input_tokens": final_message.usage.input_tokens,
                        "output_tokens": final_message.usage.output_tokens,
                    },
                )
        except Exception as exc:  # noqa: BLE001 - normalize every provider failure to one error type
            raise ModelInvocationError(f"Anthropic stream failed: {exc}") from exc
