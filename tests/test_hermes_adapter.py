import json

import pytest

from sutra.core.exceptions import ModelInvocationError
from sutra.models.hermes_adapter import HermesFormatModelClient

# ---------------------------------------------------------------------------
# _render_tools_block
# ---------------------------------------------------------------------------


def test_render_tools_block_empty_returns_empty_string():
    assert HermesFormatModelClient._render_tools_block([]) == ""


def test_render_tools_block_wraps_tools_in_xml_tag():
    tools = [
        {
            "name": "get_weather",
            "description": "Look up current weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "city name"}},
                "required": ["city"],
            },
        }
    ]
    block = HermesFormatModelClient._render_tools_block(tools)

    assert block.startswith("<tools>\n")
    assert block.endswith("\n</tools>")
    assert "get_weather" in block
    assert "Look up current weather for a city." in block

    # Every non-tag line must be valid, individually-parseable JSON matching
    # the original tool schema (documented "JSON Lines inside <tools>" format).
    inner_lines = block.splitlines()[1:-1]
    assert len(inner_lines) == 1
    assert json.loads(inner_lines[0]) == tools[0]


def test_render_tools_block_multiple_tools_one_json_object_per_line():
    tools = [
        {"name": "tool_a", "description": "does a", "input_schema": {"type": "object", "properties": {}}},
        {"name": "tool_b", "description": "does b", "input_schema": {"type": "object", "properties": {}}},
    ]
    block = HermesFormatModelClient._render_tools_block(tools)
    inner_lines = block.splitlines()[1:-1]

    assert len(inner_lines) == 2
    assert json.loads(inner_lines[0])["name"] == "tool_a"
    assert json.loads(inner_lines[1])["name"] == "tool_b"


# ---------------------------------------------------------------------------
# _extract_tool_calls
# ---------------------------------------------------------------------------


def test_extract_tool_calls_no_block_returns_empty_list():
    assert HermesFormatModelClient._extract_tool_calls("just a normal reply, no tools needed.") == []


def test_extract_tool_calls_single_well_formed_block():
    text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Boston"}}</tool_call>'
    calls = HermesFormatModelClient._extract_tool_calls(text)

    assert calls == [{"name": "get_weather", "arguments": {"city": "Boston"}}]


def test_extract_tool_calls_two_sequential_blocks():
    text = (
        '<tool_call>{"name": "tool_a", "arguments": {"x": 1}}</tool_call>'
        '<tool_call>{"name": "tool_b", "arguments": {"y": 2}}</tool_call>'
    )
    calls = HermesFormatModelClient._extract_tool_calls(text)

    assert calls == [
        {"name": "tool_a", "arguments": {"x": 1}},
        {"name": "tool_b", "arguments": {"y": 2}},
    ]


def test_extract_tool_calls_malformed_block_is_skipped_not_raised():
    # Not valid JSON inside the tags -- should be silently dropped, not raise.
    text = "<tool_call>{not valid json at all</tool_call>"
    assert HermesFormatModelClient._extract_tool_calls(text) == []


def test_extract_tool_calls_malformed_block_does_not_prevent_valid_ones():
    text = (
        "<tool_call>{not valid json</tool_call>"
        '<tool_call>{"name": "tool_ok", "arguments": {}}</tool_call>'
    )
    calls = HermesFormatModelClient._extract_tool_calls(text)
    assert calls == [{"name": "tool_ok", "arguments": {}}]


def test_extract_tool_calls_json_without_name_key_is_skipped():
    text = '<tool_call>{"arguments": {"x": 1}}</tool_call>'
    assert HermesFormatModelClient._extract_tool_calls(text) == []


def test_extract_tool_calls_missing_arguments_defaults_to_empty_dict():
    text = '<tool_call>{"name": "no_args_tool"}</tool_call>'
    calls = HermesFormatModelClient._extract_tool_calls(text)
    assert calls == [{"name": "no_args_tool", "arguments": {}}]


def test_extract_tool_calls_text_before_and_after_block():
    text = (
        "Sure, let me check that for you.\n"
        '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>\n'
        "I'll let you know once I hear back."
    )
    calls = HermesFormatModelClient._extract_tool_calls(text)
    assert calls == [{"name": "get_weather", "arguments": {"city": "NYC"}}]


# ---------------------------------------------------------------------------
# End-to-end stream() against a fake httpx.AsyncClient
# ---------------------------------------------------------------------------


class _FakeStreamCtx:
    """Fakes the object returned by `httpx.AsyncClient.stream(...)`."""

    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeAsyncClient:
    """Fakes `httpx.AsyncClient` well enough to drive `HermesFormatModelClient.stream()`."""

    last_payload = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, json=None):
        type(self).last_payload = json
        return _FakeStreamCtx(type(self)._next_lines)


def _ollama_lines(*message_contents, done_extra=None):
    """Build fake Ollama JSON-lines output: one content delta per line, then a done line."""
    lines = [json.dumps({"message": {"content": c}, "done": False}) for c in message_contents]
    final = {"message": {"content": ""}, "done": True, "prompt_eval_count": 11, "eval_count": 22}
    if done_extra:
        final.update(done_extra)
    lines.append(json.dumps(final))
    return lines


async def _run_stream(monkeypatch, lines, tools=None):
    import httpx

    _FakeAsyncClient._next_lines = lines
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    client = HermesFormatModelClient(model="hermes3", base_url="http://localhost:11434")
    chunks = [
        chunk
        async for chunk in client.stream(
            system_prompt="You are a helpful agent.",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools or [],
        )
    ]
    return chunks, _FakeAsyncClient.last_payload


async def test_stream_plain_text_no_tool_calls(monkeypatch):
    lines = _ollama_lines("Hello", " there", "!")
    chunks, payload = await _run_stream(monkeypatch, lines)

    text = "".join(c.delta_text for c in chunks if c.delta_text)
    assert text == "Hello there!"

    final = chunks[-1]
    assert final.finished is True
    assert final.finish_reason == "stop"
    assert final.tool_call_delta is None
    assert final.usage == {"input_tokens": 11, "output_tokens": 22}

    # No "tools" field should ever be sent on the wire.
    assert "tools" not in payload


async def test_stream_renders_tools_block_into_system_prompt(monkeypatch):
    tools = [{"name": "get_weather", "description": "look up weather", "input_schema": {"type": "object", "properties": {}}}]
    lines = _ollama_lines("ok")
    _, payload = await _run_stream(monkeypatch, lines, tools=tools)

    system_message = payload["messages"][0]
    assert system_message["role"] == "system"
    assert "<tools>" in system_message["content"]
    assert "get_weather" in system_message["content"]
    assert "You are a helpful agent." in system_message["content"]


async def test_stream_extracts_single_tool_call_split_across_chunks(monkeypatch):
    # Simulate the tag arriving split across multiple network chunks.
    lines = _ollama_lines(
        "Let me check that. ",
        "<tool_call>",
        '{"name": "get_weather", "arguments": {"city": "Boston"}}',
        "</tool_call>",
    )
    chunks, _ = await _run_stream(monkeypatch, lines)

    text = "".join(c.delta_text for c in chunks if c.delta_text)
    assert text == "Let me check that. "
    assert "<tool_call>" not in text

    final = chunks[-1]
    assert final.finish_reason == "tool_use"
    assert final.tool_call_delta == {
        "finalized": [{"id": "call_0", "name": "get_weather", "input": {"city": "Boston"}}]
    }


async def test_stream_extracts_multiple_tool_calls_in_one_turn(monkeypatch):
    lines = _ollama_lines(
        '<tool_call>{"name": "tool_a", "arguments": {"x": 1}}</tool_call>',
        '<tool_call>{"name": "tool_b", "arguments": {"y": 2}}</tool_call>',
    )
    chunks, _ = await _run_stream(monkeypatch, lines)

    final = chunks[-1]
    assert final.tool_call_delta == {
        "finalized": [
            {"id": "call_0", "name": "tool_a", "input": {"x": 1}},
            {"id": "call_1", "name": "tool_b", "input": {"y": 2}},
        ]
    }


async def test_stream_wraps_errors_in_model_invocation_error(monkeypatch):
    import httpx

    class _BoomClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, json=None):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(httpx, "AsyncClient", _BoomClient)

    client = HermesFormatModelClient(model="hermes3")
    with pytest.raises(ModelInvocationError):
        async for _ in client.stream(system_prompt="sys", messages=[], tools=[]):
            pass
