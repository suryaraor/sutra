# Feature: Harness State

**Module:** `src/sutra/core/state.py`  
**Classes:** `HarnessState`, `Message`, `ExecutionStep`, `PendingPermission`, `HarnessStatus`, `Role`

## Purpose

The single source of truth for everything that has happened in a harness run — conversation history, execution progress, and pause/resume metadata. Designed to be checkpointed and rehydrated so a run can survive a permission-gate pause, a process restart, or a migration to a different machine.

## Current Behavior

### `Role` Enum

| Value | Usage |
|---|---|
| `system` | Injected system notices (handoff briefings) |
| `user` | Human turn messages |
| `assistant` | Model-generated responses |
| `tool` | Tool execution results |
| `summary` | Synthetic messages produced by `ContextCompactor` |

### `HarnessStatus` Enum

| Value | Meaning |
|---|---|
| `idle` | Not yet started |
| `running` | Actively processing a turn |
| `paused_for_permission` | Halted waiting for HITL approval |
| `paused_for_handoff` | Reserved (not yet used) |
| `completed` | Turn finished successfully |
| `failed` | Error occurred (guardrail, tool, model, handoff) |
| `budget_exceeded` | A budget hard limit was crossed |

### `Message`

```python
@dataclass
class Message:
    role: Role
    content: str
    name: Optional[str] = None           # tool name for role=tool messages
    tool_call_id: Optional[str] = None   # correlation ID for tool call/result pairs
    tool_calls: Optional[List[Dict]] = None  # list of tool call dicts for role=assistant
    input_tokens: int = 0
    output_tokens: int = 0
    created_at: float                    # time.time()
    message_id: str                      # uuid4 hex[:12]
```

`approx_tokens()` returns `max(1, len(content) // 4) + input_tokens + output_tokens` — a fast heuristic used by `ContextCompactor` when `tiktoken` is unavailable.

### `ExecutionStep`

```python
@dataclass
class ExecutionStep:
    step_number: int
    action: str   # "model_call" | "tool_call" | "handoff" | "permission_wait" | "compaction" | "guardrail_block" | "budget_exceeded"
    detail: Dict[str, Any]
    created_at: float
```

An ordered audit log of every significant action in the run. Written by `HarnessState.record_step()`.

### `PendingPermission`

```python
@dataclass
class PendingPermission:
    request_id: str
    tool_name: str
    arguments: Dict[str, Any]
    reason: str
    tool_call_id: Optional[str] = None
    created_at: float
```

Set on `HarnessState` when a permission gate opens; cleared when `resume_after_permission()` is called. Included in the `DONE` event payload when status is `PAUSED_FOR_PERMISSION`.

### `HarnessState`

```python
@dataclass
class HarnessState:
    session_id: str              # uuid4 hex
    status: HarnessStatus        # current lifecycle status
    system_prompt: str           # root agent prompt
    messages: List[Message]      # full conversation history
    execution_steps: List[ExecutionStep]
    active_agent_id: str         # "root" or a subagent id
    agent_stack: List[str]       # e.g. ["root", "it_ops_agent"]
    pending_permission: Optional[PendingPermission]
    compaction_count: int        # how many times the window has been compacted
    step_count: int              # total model call steps across all hops
    created_at: float
    updated_at: float
```

### Mutation Helpers

| Method | Effect |
|---|---|
| `append_message(message)` | Appends to `messages`, updates `updated_at` |
| `record_step(action, **detail)` | Creates `ExecutionStep`, increments `step_count` |
| `push_agent(agent_id)` | Appends to `agent_stack`, sets `active_agent_id` |
| `pop_agent()` | Pops stack, sets `active_agent_id` to new top |
| `total_tokens()` | Sum of all `message.input_tokens + output_tokens` |

### Serialization

```python
state.to_dict()   -> Dict
state.to_json()   -> str
state.checkpoint() -> str   # alias for to_json()

HarnessState.from_dict(d)   -> HarnessState
HarnessState.from_json(s)   -> HarnessState
```

Full round-trip fidelity including nested `Message` and `ExecutionStep` objects, `PendingPermission`, and all enum values.

## Enhancement Ideas

- **Persistent checkpointing**: add a `checkpoint(path: str)` convenience method that writes the JSON to a file; add `HarnessState.load(path)` to rehydrate.
- **Selective message pruning**: add a method to drop specific messages by `message_id` (e.g. to remove a failed tool call and retry it).
- **State diffing**: `state.diff(other: HarnessState) -> List[str]` for debugging and test assertions.
- **Schema versioning**: add a `schema_version` field to `to_dict()` output so rehydration can handle backwards-incompatible changes gracefully.
- **Typed `execution_steps`**: currently `action` is a plain string; replace with an enum to enforce valid values and enable exhaustive pattern matching.
- **`paused_for_handoff` status**: currently defined in the enum but never set; wire it up if multi-hop handoffs need an async pause point.
- **Event sourcing**: instead of mutating state in place, record every mutation as an immutable event and derive current state by replaying them — enables time-travel debugging.
- **Token budget tracking on `HarnessState`**: expose `remaining_budget()` derived from `total_tokens()` vs. the `Budget.config` limits, so the state object is self-contained for monitoring purposes.
