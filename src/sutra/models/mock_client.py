from __future__ import annotations

from typing import Any, AsyncIterator, Callable, Dict, List

from sutra.models.base import ModelClient, StreamChunk

ScriptFn = Callable[[List[Dict[str, Any]]], Dict[str, Any]]


class MockModelClient(ModelClient):
    """Deterministic, network-free model client for tests and demos.

    A `script` callable inspects the running wire-format conversation and
    returns the next assistant turn as `{"text": str, "tool_calls": [...]}`.
    This keeps example/demo code runnable without real provider credentials
    while exercising the exact same `ModelClient` interface a production
    client (Anthropic/OpenAI/Ollama) would.
    """

    def __init__(self, script: ScriptFn) -> None:
        self.script = script

    async def stream(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[StreamChunk]:
        turn = self.script(messages)
        text = turn.get("text", "")
        tool_calls = turn.get("tool_calls", [])

        for word in text.split(" "):
            if word:
                yield StreamChunk(delta_text=word + " ")

        input_tokens = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
        output_tokens = max(1, len(text) // 4)

        yield StreamChunk(
            finished=True,
            finish_reason="tool_use" if tool_calls else "stop",
            tool_call_delta={"finalized": tool_calls} if tool_calls else None,
            usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
        )
