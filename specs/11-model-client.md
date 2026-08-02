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
    delta_text: str = ""                      # incremental text token
    tool_call_delta: Optional[Dict] = None    # tool call accumulator; "finalized" key holds completed calls
    finished: bool = False                    # True on the last chunk
    finish_reason: Optional[str] = None       # "stop" | "tool_use" | "max_tokens"
    usage: Optional[Dict[str, int]] = None   # {"input_tokens": N, "output_tokens": M} on finished chunk
```

### `ModelResponse` (non-streaming)

```python
@dataclass
class ModelResponse:
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = "stop"
```

Used in examples and tests; the harness only uses the streaming path.

### Implementations

#### `AnthropicModelClient`

```python
from sutra.models.anthropic_client import AnthropicModelClient
client = AnthropicModelClient(model="claude-sonnet-5")
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

def make_script():
    call_count = {"n": 0}

    def script(_messages):
        call_count["n"] += 1
        n = call_count["n"]

        if n == 1:
            return {
                "text": "Handing off to the finance operations subagent.",
                "tool_calls": [
                    {
                        "id": "call_handoff_1",
                        "name": HANDOFF_TOOL_NAME,
                        "input": {"target_agent_id": "finance_ops_agent", "reason": "..."},
                    }
                ],
            }
        return {"text": "Done.", "tool_calls": []}

    return script

client = MockModelClient(script=make_script())
```

- Deterministic, network-free
- The constructor takes a single `script: ScriptFn` callable — `Callable[[List[Dict[str, Any]]], Dict[str, Any]]` — not a list of `ModelResponse` objects
- On each call, `script(messages)` is invoked with the running wire-format conversation and must return `{"text": str, "tool_calls": [...]}` for the next assistant turn
- Used in all tests and `examples/finance_workflow.py` (see `make_script()` there for a realistic multi-turn scripted scenario)
- Emits one `StreamChunk(delta_text=...)` per whitespace-split word of `text`, followed by a final `StreamChunk(finished=True, finish_reason=..., tool_call_delta=..., usage={...})`

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
