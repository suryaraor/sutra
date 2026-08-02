# Feature: Model Client Interface

**Module:** `src/sutra/models/`  
**Classes:** `ModelClient` (ABC), `StreamChunk`, `ModelResponse`  
**Implementations:** `AnthropicModelClient`, `OpenAIModelClient`, `OllamaModelClient`, `MockModelClient`

## Purpose

The abstraction layer that keeps the harness LLM-agnostic. The harness only ever calls `ModelClient.stream()`; swapping providers requires only changing the constructor argument, not touching any harness logic.

## Current Behavior

### `ModelClient` (Abstract Base Class)

```python
class ModelClient(ABC):
    @abstractmethod
    async def stream(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[StreamChunk]: ...
```

Only one method. The harness calls it on every model turn with:
- `system_prompt`: the effective system prompt (root + subagent overlay + memory context)
- `messages`: the wire-format conversation history (from `_messages_for_wire()`)
- `tools`: JSON schemas of tools visible to the current agent

### `StreamChunk`

```python
@dataclass
class StreamChunk:
    delta_text: Optional[str] = None          # incremental text token
    tool_call_delta: Optional[Dict] = None    # tool call accumulator; "finalized" key holds completed calls
    finished: bool = False                    # True on the last chunk
    usage: Optional[Dict[str, int]] = None   # {"input_tokens": N, "output_tokens": M} on finished chunk
```

### `ModelResponse` (non-streaming)

```python
@dataclass
class ModelResponse:
    text: str
    tool_calls: List[Dict[str, Any]]
    input_tokens: int
    output_tokens: int
```

Used in examples and tests; the harness only uses the streaming path.

### Implementations

#### `AnthropicModelClient`

```python
from sutra.models.anthropic_client import AnthropicModelClient
client = AnthropicModelClient(model="claude-sonnet-4-6")
```

- Requires `pip install -e ".[anthropic]"` (`anthropic>=0.34`)
- Uses Anthropic's streaming Messages API
- Adapts Anthropic's block-structured content format to `StreamChunk`
- Converts tool schemas from `input_schema` format (Sutra native) to Anthropic's expected format

#### `OpenAIModelClient`

```python
from sutra.models.openai_client import OpenAIModelClient
client = OpenAIModelClient(model="gpt-4o")
```

- Requires `pip install -e ".[openai]"` (`openai>=1.40`)
- Uses OpenAI's streaming Chat Completions API

#### `OllamaModelClient`

```python
from sutra.models.ollama_client import OllamaModelClient
client = OllamaModelClient(model="gpt-oss:20b", base_url="http://localhost:11434")
```

- Requires a running Ollama server (`ollama pull <model>`)
- Uses `httpx` for HTTP; no Ollama SDK dependency
- The default model for the CLI

#### `MockModelClient`

```python
from sutra.models.mock_client import MockModelClient
client = MockModelClient(responses=[...])
```

- Deterministic, network-free
- `responses` is a list of `ModelResponse` objects yielded in order
- Used in all tests and `examples/finance_workflow.py`
- Emits one `StreamChunk(delta_text=response.text, ...)` per response followed by a final `StreamChunk(finished=True, usage={...})`

### Wire Message Format

The harness produces OpenAI/Ollama-style flat messages:

```json
[
  {"role": "user", "content": "Help me with an incident"},
  {"role": "assistant", "content": "", "tool_calls": [{"id": "tc_001", "type": "function", "function": {"name": "__handoff__", "arguments": {...}}}]},
  {"role": "tool", "tool_call_id": "tc_001", "name": "__handoff__", "content": "..."}
]
```

`AnthropicModelClient` must adapt this to Anthropic's block-structured format internally.

## Enhancement Ideas

- **`AnthropicModelClient` extended thinking**: add `thinking_budget_tokens` parameter to enable Claude's extended thinking mode; emit `thinking` SSE events for the reasoning text.
- **Unified tool schema format**: currently schemas are in Anthropic's `input_schema` format; each client adapts them. Consider normalizing to OpenAI format at the harness level and having `AnthropicModelClient` adapt inward.
- **Google Gemini client**: add `GeminiModelClient` using the `google-generativeai` SDK.
- **Cohere / Mistral clients**: additional provider adapters.
- **Connection pooling**: `OllamaModelClient` creates a new `httpx` client per call; add connection reuse.
- **Retry with backoff**: wrap `stream()` in retry logic (exponential backoff + jitter) for transient 429/5xx responses.
- **Token pre-estimation**: add `estimate_tokens(messages, tools) -> int` to `ModelClient` so the budget can check affordability before calling.
- **Model capability registry**: a static mapping of model IDs to capabilities (supports tool calls, supports vision, context window size) so the harness can make informed routing decisions.
- **Streaming health check**: `async def health_check() -> bool` abstract method so the harness can verify provider connectivity at startup.
- **Fallback chain**: configure a list of model clients; on failure, automatically try the next one (e.g. primary Anthropic → fallback Ollama).
