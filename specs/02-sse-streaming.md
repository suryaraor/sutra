# Feature: Server-Sent Events (SSE Streaming)

**Module:** `src/sutra/streaming/sse.py`  
**Classes:** `SSEEvent`, `EventType`  
**Helper:** `format_sse()`

## Purpose

Defines the wire protocol for streaming all harness activity to clients in real time. Every significant action emits a typed SSE frame, enabling frontends, CLIs, and HTTP clients to react to tokens, tool calls, permission requests, and errors as they happen rather than waiting for a completed response.

## Current Behavior

### Event Types

| Event | When emitted | Key fields |
|---|---|---|
| `session_start` | Top of every `run()` call | `session_id`, `active_agent` |
| `guardrail_block` | Input blocked by a guardrail rule | `rule`, `direction`, `message` |
| `token` | Each streaming text delta from the model | `text`, `agent` |
| `message_complete` | Model turn finished | `agent`, `text`, `has_tool_calls` |
| `tool_call` | Before a tool is dispatched | `tool_name`, `arguments`, `agent` |
| `tool_result` | After a tool returns | `tool_name`, `result`, `agent` |
| `permission_request` | CRITICAL/HIGH tool intercepted | `request_id`, `tool_name`, `arguments`, `level`, `reason` |
| `permission_resolved` | Human approved or denied | `request_id`, `approved`, `actor`, `tool_name` |
| `handoff` | Agent stack switch | `from_agent`, `to_agent`, `reason`, `context_payload` |
| `budget_warning` | Any budget dimension past `warn_ratio` | `dimension`, `ratio_used`, `snapshot` |
| `budget_exceeded` | Hard budget limit crossed | `dimension`, `used`, `limit`, `message` |
| `compaction` | Context window was compacted | `compaction_count`, `resulting_messages` |
| `error` | Any non-budget failure | `phase`, `message`, (optional) `tool_name` |
| `done` | Terminal event — always the last frame | `status`, (optional) `pending_permission` |

### Wire Format

```
event: token
data: {"id": "a1b2c3d4e5", "type": "token", "created_at": 1720000000.0, "text": "Hello", "agent": "root"}

```

Each frame is:
```
event: <event_type>\n
data: <json_payload>\n
\n
```

The JSON payload always includes `id` (10-char hex), `type`, and `created_at` (unix float), plus event-specific fields.

### SSEEvent Dataclass

```python
@dataclass
class SSEEvent:
    type: EventType
    data: Dict[str, Any]
    event_id: str          # uuid4 hex[:10], auto-generated
    created_at: float      # time.time(), auto-generated

    def to_sse(self) -> str: ...
```

`to_sse()` merges `event_id`, `type.value`, `created_at`, and `data` fields into a single JSON object on the `data:` line.

### `format_sse()` Helper

```python
format_sse(EventType.TOKEN, {"text": "hello"}) -> str
```

Convenience one-liner — creates a throwaway `SSEEvent` and returns its `to_sse()` string.

## Enhancement Ideas

- **Event versioning**: add a `spec_version` field (e.g. `"v1"`) to every payload so clients can handle breaking changes gracefully.
- **Sequence numbers**: add a monotonically increasing `seq` field so clients can detect dropped frames or replay from a specific point.
- **Event filtering**: allow clients to subscribe to a subset of event types (e.g. only `token` + `done`) via HTTP query params, reducing bandwidth for simple chat UIs.
- **Retry / reconnect support**: emit an SSE `id:` line (separate from `event_id` in the payload) compatible with `EventSource.lastEventId` so browsers can reconnect mid-stream without replaying from the start.
- **Binary transport option**: for high-throughput deployments, add a msgpack or CBOR codec alongside the text SSE format.
- **`thinking` event type**: expose model chain-of-thought / extended thinking tokens as a distinct event type, separate from `token`, so UIs can render them differently.
- **`subagent_start` / `subagent_end` events**: explicit scope markers when a handoff activates/deactivates a subagent frame.
- **Structured tool result events**: currently `result` is a raw string; add a `result_schema` field or typed variant for structured outputs.
