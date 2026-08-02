# Feature: Hermes Tool-Calling Format Adapter

**Module:** `src/sutra/models/hermes_adapter.py`  
**Class:** `HermesFormatModelClient`  
**Implements:** `ModelClient`

## Purpose

Many open-weight models — not only ones literally branded "Hermes," but any model fine-tuned on the tool-calling convention popularized by Nous Research's Hermes releases — do not support a structured `tools=[...]` API field the way `OllamaModelClient` assumes. Instead they are trained on a purely text-based convention: tool schemas are rendered into a `<tools>...</tools>` block appended to the system prompt, and the model emits tool calls as raw text of the form `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` interleaved with its ordinary response text.

`HermesFormatModelClient` adapts that convention to Sutra's `ModelClient` interface so the harness can drive a Hermes-format model exactly like any other provider — the harness never knows tool calls are being parsed out of free text rather than delivered as structured fields.

## Current Behavior

### Constructor

```python
from sutra.models.hermes_adapter import HermesFormatModelClient
client = HermesFormatModelClient(model="hermes-3-llama-3.1-8b", base_url="http://localhost:11434")
```

- Requires `httpx` (imported lazily; raises a friendly `ImportError` pointing at `pip install sutra[ollama]` if missing), same pattern as `OllamaModelClient`.
- Talks to a local Ollama server's `/api/chat` endpoint — Ollama is only the transport; the tool-calling *convention* is what differs from `OllamaModelClient`.

### Rendering tools into the system prompt

Sutra's native tool schema shape (`Tool.to_schema()` in `src/sutra/tools/registry.py`) is:

```json
{"name": "get_weather", "description": "Look up current weather", "input_schema": {"type": "object", "properties": {...}, "required": [...]}}
```

`_render_tools_block(tools)` renders the list of these schemas as one `json.dumps(...)` object per line, wrapped in `<tools>...</tools>`:

```
<tools>
{"name": "get_weather", "description": "Look up current weather", "input_schema": {...}}
{"name": "search_docs", "description": "Search internal docs", "input_schema": {...}}
</tools>
```

This "JSON Lines" layout (one object per line, not a single JSON array) matches the reference prompts Nous Research publishes for Hermes function-calling and is what these models were actually fine-tuned against.

The rendered block is appended to `system_prompt` (separated by a blank line) before the request is sent. **No `tools` field is ever included in the Ollama request payload** — the model has no structured tool-calling path to fall back on, so putting schemas in the request body would do nothing (or, worse, fight the in-context block if Ollama's chat template injects its own).

### Parsing tool calls out of the stream

Ollama's `/api/chat` streaming response is unchanged from `OllamaModelClient` — newline-delimited JSON objects, each with `message.content` (an incremental text delta) and a final line with `"done": true` carrying `prompt_eval_count` / `eval_count`.

`HermesFormatModelClient` accumulates every `message.content` delta into an internal buffer and:

1. Releases newly-arrived plain text as `StreamChunk(delta_text=...)` as soon as it can prove that text isn't part of an in-progress `<tool_call>` block (it conservatively holds back a trailing partial tag, e.g. text ending in `"<tool_c"`, until the next chunk disambiguates it).
2. On the final (`"done": true`) line, runs `_extract_tool_calls()` over the *entire* accumulated buffer to pick up every `<tool_call>{...}</tool_call>` block the model emitted (a single turn may contain several), and flushes any remaining unreleased plain text.
3. Emits the terminal `StreamChunk`:
   - `finished=True`
   - `finish_reason="tool_use"` if any tool calls were found, else `"stop"`
   - `tool_call_delta={"finalized": [...]}` where each entry is `{"id": f"call_{n}", "name": ..., "input": ...}` (`n` is the 0-based index among tool calls found in *this* turn — matching the `{"finalized": [...]}` contract the harness reads via `chunk.tool_call_delta["finalized"]`)
   - `usage={"input_tokens": prompt_eval_count, "output_tokens": eval_count}`

A malformed `<tool_call>` block (invalid JSON, or JSON that isn't an object with a `"name"` key) is skipped rather than raising — one bad emission from the model shouldn't take down the whole stream, and any well-formed blocks before or after it are still returned.

Any exception anywhere in the request/stream/parse path (network failure, non-2xx response, JSON-lines decode failure) is caught once and re-raised as `ModelInvocationError`, matching `OllamaModelClient`'s error-wrapping pattern.

### Wire format comparison

| | `OllamaModelClient` | `HermesFormatModelClient` |
|---|---|---|
| Tool schemas sent as | `payload["tools"] = [...]` (OpenAI-function-shaped) | Rendered into `system_prompt` as `<tools>...</tools>` text; no `tools` field on the wire |
| Tool calls received as | `message.tool_calls` (structured JSON on the response object) | Parsed out of `message.content` text via `<tool_call>{...}</tool_call>` regex matching |
| Streaming text | `message.content` deltas forwarded as-is | `message.content` deltas forwarded, minus any text inside `<tool_call>` blocks |
| Multiple tool calls per turn | One entry per item in `message.tool_calls` | One entry per `<tool_call>...</tool_call>` block found in the accumulated buffer |

### Pure, testable pieces

Two static methods hold all of the format-specific logic and take no network dependency, so they're unit-testable in isolation:

- `HermesFormatModelClient._render_tools_block(tools: list[dict]) -> str`
- `HermesFormatModelClient._extract_tool_calls(text: str) -> list[dict]`

## Enhancement Ideas

- **Streaming-safe tag detection via a proper tokenizer state machine**: the current buffer-scanning approach (`_safe_emit_boundary`) is regex/string-search based and re-scans the whole buffer on each chunk; for very long tool-call-free responses this is O(n) per chunk. A small incremental state machine would make it O(1) amortized.
- **Configurable tag vocabulary**: some Hermes-family fine-tunes use slightly different tags (e.g. `<function_call>` / `<tool_response>` variants seen in earlier Hermes-2 prompts). Add a constructor parameter to override the tag names instead of hardcoding `<tool_call>`.
- **Tool-result rendering**: currently only the outbound tool *definitions* are adapted; a full round trip also needs tool *results* rendered back into the conversation in whatever format the model expects for its next turn (Hermes convention wraps them in `<tool_response>...</tool_response>`). Add a `render_tool_result(name, content) -> str` helper alongside `_render_tools_block`.
- **Partial/streamed tool-call surfacing**: right now tool calls only become visible to the harness on the final chunk. Emit incremental `tool_call_delta` updates (e.g. a `"partial"` key with the in-progress JSON fragment) so a UI could show "calling `get_weather`..." before the arguments finish streaming.
- **Strict-mode validation**: optionally validate parsed `arguments` against the tool's `input_schema` at parse time (reusing `ToolRegistry.validate_arguments`) and surface a recoverable error chunk instead of silently dropping malformed calls.
- **Grammar-constrained decoding**: if the target Ollama build exposes GBNF/JSON-schema-constrained sampling, use it to force well-formed `<tool_call>` JSON instead of relying purely on the model's training to get the format right.
- **System prompt template customization**: the exact wording Nous Research recommends around the `<tools>` block (e.g. instructions on when to call a tool vs. respond in prose) is currently left entirely to the caller's `system_prompt`. Consider a `system_prompt_template` option that injects the recommended boilerplate automatically.
- **Detect non-streaming servers**: fall back to a single non-streamed `/api/chat` call (`stream: false`) for Ollama builds/models that don't support incremental streaming for a given model tag.
