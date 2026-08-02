from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List

from sutra.core.exceptions import ModelInvocationError
from sutra.models.base import ModelClient, StreamChunk


class OllamaModelClient(ModelClient):
    """Thin async wrapper around a local Ollama server's `/api/chat` endpoint."""

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434") -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'httpx' package is required for OllamaModelClient. "
                "Install with `pip install sutra[ollama]`."
            ) from exc
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def stream(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[StreamChunk]:
        import httpx

        oai_messages = [{"role": "system", "content": system_prompt}, *messages]
        payload: Dict[str, Any] = {"model": self.model, "messages": oai_messages, "stream": True}
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
                }
                for t in tools
            ]

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        message = obj.get("message", {})
                        content = message.get("content", "")
                        tool_calls = message.get("tool_calls")

                        if content:
                            yield StreamChunk(delta_text=content)
                        if tool_calls:
                            finalized = [
                                {
                                    "id": f"call_{i}",
                                    "name": tc["function"]["name"],
                                    "input": tc["function"].get("arguments", {}),
                                }
                                for i, tc in enumerate(tool_calls)
                            ]
                            yield StreamChunk(tool_call_delta={"finalized": finalized})
                        if obj.get("done"):
                            yield StreamChunk(
                                finished=True,
                                finish_reason="tool_use" if tool_calls else "stop",
                                usage={
                                    "input_tokens": obj.get("prompt_eval_count", 0),
                                    "output_tokens": obj.get("eval_count", 0),
                                },
                            )
        except Exception as exc:  # noqa: BLE001 - normalize every provider failure to one error type
            raise ModelInvocationError(f"Ollama stream failed: {exc}") from exc
