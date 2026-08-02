from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class StreamChunk:
    """A single raw chunk from an underlying model stream."""

    delta_text: str = ""
    tool_call_delta: Optional[Dict[str, Any]] = None
    finished: bool = False
    finish_reason: Optional[str] = None  # "stop" | "tool_use" | "max_tokens"
    usage: Optional[Dict[str, int]] = None  # populated on the final chunk


@dataclass
class ModelResponse:
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = "stop"


class ModelClient(ABC):
    """LLM-agnostic model wrapper boundary.

    Any provider — Anthropic, OpenAI, Google GenAI, a local Ollama server —
    plugs into the harness by implementing this interface. The harness loop
    never imports a provider SDK directly; it only ever calls `stream()`.
    """

    @abstractmethod
    def stream(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[StreamChunk]:
        """Yield StreamChunks as they arrive; the final chunk must set `finished=True`."""
        raise NotImplementedError

    async def complete(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> ModelResponse:
        """Default implementation: drain `stream()` into a single response."""
        content = ""
        tool_calls: List[Dict[str, Any]] = []
        input_tokens = 0
        output_tokens = 0
        stop_reason = "stop"

        async for chunk in self.stream(system_prompt=system_prompt, messages=messages, tools=tools):
            content += chunk.delta_text
            if chunk.tool_call_delta and "finalized" in chunk.tool_call_delta:
                tool_calls = chunk.tool_call_delta["finalized"]
            if chunk.finished:
                stop_reason = chunk.finish_reason or "stop"
                if chunk.usage:
                    input_tokens = chunk.usage.get("input_tokens", 0)
                    output_tokens = chunk.usage.get("output_tokens", 0)

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
        )
