from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator, Dict, List

from sutra.core.exceptions import ModelInvocationError
from sutra.models.base import ModelClient, StreamChunk

# Matches a single well-formed <tool_call>...</tool_call> block, capturing the
# raw text between the tags (which is expected to be one JSON object).
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_CALL_OPEN_TAG = "<tool_call>"


class HermesFormatModelClient(ModelClient):
    """Model client for the "Hermes" in-context tool-calling convention.

    Many open-weight models fine-tuned by Nous Research (and others using the
    same recipe — not only models literally branded "Hermes") do not support
    a structured `tools=[...]` request field. Instead they are trained to
    expect tool definitions rendered as text and appended to the system
    prompt inside a `<tools>...</tools>` block, and they respond by emitting
    tool calls as raw text of the form:

        <tool_call>
        {"name": "get_weather", "arguments": {"city": "Boston"}}
        </tool_call>

    This client talks to the same kind of local Ollama server that
    `OllamaModelClient` does, but never sends a `tools` field on the wire —
    it renders the tool schemas into the system prompt and parses tool calls
    back out of the streamed text instead of relying on Ollama's structured
    `message.tool_calls` field.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'httpx' package is required for HermesFormatModelClient. "
                "Install with `pip install sutra[ollama]`."
            ) from exc
        self.model = model
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def _render_tools_block(tools: List[Dict[str, Any]]) -> str:
        """Render Sutra-native tool schemas into a Hermes-style `<tools>` block.

        Sutra's native tool schema shape (see `Tool.to_schema()`) is:

            {"name": ..., "description": ..., "input_schema": {...}}

        Hermes-trained models expect each tool's JSON schema rendered inside
        a `<tools>...</tools>` block. We render one JSON object per line
        (JSON Lines style, `json.dumps(tool)` per tool) rather than a single
        JSON array — that is the layout used in Nous Research's published
        Hermes function-calling prompts, and it is what these models were
        actually fine-tuned on.
        """
        if not tools:
            return ""
        lines = [json.dumps(tool) for tool in tools]
        return "<tools>\n" + "\n".join(lines) + "\n</tools>"

    @staticmethod
    def _extract_tool_calls(text: str) -> List[Dict[str, Any]]:
        """Parse zero or more `<tool_call>{...}</tool_call>` blocks out of `text`.

        Returns a list of `{"name": ..., "arguments": ...}` dicts, in the
        order they appeared. A malformed block — invalid JSON, or valid JSON
        that isn't an object containing a "name" key — is silently skipped
        rather than raising: one bad tool call emitted by the model should
        not take down the whole stream. Well-formed blocks before or after a
        malformed one are still returned.
        """
        calls: List[Dict[str, Any]] = []
        for match in _TOOL_CALL_RE.finditer(text):
            raw = match.group(1).strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict) or "name" not in parsed:
                continue
            calls.append({"name": parsed["name"], "arguments": parsed.get("arguments", {})})
        return calls

    async def stream(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[StreamChunk]:
        import httpx

        tools_block = self._render_tools_block(tools)
        effective_system_prompt = f"{system_prompt}\n\n{tools_block}" if tools_block else system_prompt

        oai_messages = [{"role": "system", "content": effective_system_prompt}, *messages]
        # No `tools` field on the wire: a Hermes-trained model expects tool
        # definitions purely as the in-context <tools> block above, not as a
        # structured request field the server would otherwise inject into
        # the prompt template itself (which would fight our own rendering).
        payload: Dict[str, Any] = {"model": self.model, "messages": oai_messages, "stream": True}

        try:
            buffer = ""  # full raw text accumulated so far
            released = 0  # buffer[:released] has already been resolved (emitted or absorbed into a tool call)

            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        content = obj.get("message", {}).get("content", "")
                        if content:
                            buffer += content

                        boundary = len(buffer) if obj.get("done") else _safe_emit_boundary(buffer)
                        if boundary > released:
                            segment = buffer[released:boundary]
                            visible = _TOOL_CALL_RE.sub("", segment)
                            if visible:
                                yield StreamChunk(delta_text=visible)
                            released = boundary

                        if obj.get("done"):
                            parsed_calls = self._extract_tool_calls(buffer)
                            finalized = [
                                {"id": f"call_{i}", "name": call["name"], "input": call["arguments"]}
                                for i, call in enumerate(parsed_calls)
                            ]
                            yield StreamChunk(
                                finished=True,
                                finish_reason="tool_use" if finalized else "stop",
                                tool_call_delta={"finalized": finalized} if finalized else None,
                                usage={
                                    "input_tokens": obj.get("prompt_eval_count", 0),
                                    "output_tokens": obj.get("eval_count", 0),
                                },
                            )
        except Exception as exc:  # noqa: BLE001 - normalize every provider failure to one error type
            raise ModelInvocationError(f"Hermes-format Ollama stream failed: {exc}") from exc


def _safe_emit_boundary(buffer: str) -> int:
    """Return the index up to which `buffer` can safely be treated as plain text.

    "Safe" means: not inside an unterminated `<tool_call>` block, and not in
    the middle of what could be the start of a `<tool_call>` opening tag
    (e.g. the chunk boundary landed right after `"<tool_c"`). We hold back
    anything from that point onward until either the tag is disproven by
    further non-matching text, or the block completes.
    """
    last_open = buffer.rfind(_TOOL_CALL_OPEN_TAG)
    last_close = buffer.rfind("</tool_call>")
    if last_open > last_close:
        # Inside an unterminated <tool_call> block: nothing from its start
        # onward is safe to release yet.
        return last_open

    # No open block in progress. Still, hold back a trailing partial prefix
    # of "<tool_call>" (e.g. buffer ends with "<", "<tool", "<tool_ca", ...).
    for length in range(min(len(_TOOL_CALL_OPEN_TAG) - 1, len(buffer)), 0, -1):
        if buffer.endswith(_TOOL_CALL_OPEN_TAG[:length]):
            return len(buffer) - length
    return len(buffer)
