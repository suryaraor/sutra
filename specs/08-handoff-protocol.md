# Feature: Handoff Protocol

**Module:** `src/sutra/core/handoff.py`  
**Classes:** `HandoffRouter`, `HandoffRequest`  
**Related:** `src/sutra/agents/subagent.py`

## Purpose

State-transfer mechanism for switching the active execution frame from one agent to another within the same harness run. Allows a root triage agent to delegate domain-specific tasks to specialist subagents with restricted tool subsets.

## Current Behavior

### `HandoffRequest`

```python
@dataclass
class HandoffRequest:
    target_agent_id: str         # must be registered in SubagentRegistry
    reason: str                  # why the root agent is handing off
    context_payload: Dict[str, Any]  # structured data to pass along
```

### `HandoffRouter`

```python
class HandoffRouter:
    def __init__(self, subagent_registry: SubagentRegistry): ...
    def validate(self, request: HandoffRequest) -> None: ...
    def build_briefing(self, request: HandoffRequest, *, from_agent_id: str) -> str: ...
```

`validate()` raises `HandoffError` if `request.target_agent_id` is not in `SubagentRegistry`.

`build_briefing()` produces a structured system message injected into the conversation:

```
[HANDOFF] Control transferred from 'root' to 'it_ops_agent'.
Reason: IT incident detected — routing to IT Operations.
Context payload:
- incident_id: INC-001
- severity: HIGH

You are now the active agent. Proceed immediately using your available tools to
resolve the request — do not just acknowledge this message.
```

### The `__handoff__` Pseudo-Tool

Handoffs are triggered by the model calling a reserved tool named `__handoff__` with this input schema:

```json
{
  "target_agent_id": "string (required)",
  "reason": "string (required)",
  "context_payload": "object (optional)"
}
```

This tool is **not** in the `ToolRegistry` — the harness intercepts calls to `__handoff__` in `_dispatch_tool_call()` before the registry lookup.

### Agent Stack

`HarnessState` maintains an `agent_stack: List[str]` (default `["root"]`) and `active_agent_id: str`.

- `state.push_agent(target_agent_id)` — activates the target; `active_agent_id = target_agent_id`
- `state.pop_agent()` — returns to the previous agent (used internally but no automatic trigger exists)

### Handoff Flow in Harness

1. Model calls `__handoff__` with `{target_agent_id, reason, context_payload}`.
2. `_handle_handoff()` creates a `HandoffRequest`, calls `validate()`.
3. `build_briefing()` generates the system notice string.
4. `state.push_agent(target_agent_id)` — the stack now shows `["root", "it_ops_agent"]`.
5. Briefing is appended as `Message(role=SYSTEM)`.
6. `HANDOFF` SSE event is emitted.
7. The inner loop continues — next model call uses the subagent's system prompt and restricted tool schemas.

### `HandoffError` Exception

Caught in `_handle_handoff()` → emits `error` SSE event → run terminates with `FAILED`.

## Enhancement Ideas

- **Return handoff (`__handoff_back__`)**: add a second pseudo-tool so a subagent can explicitly yield back to its caller; the harness calls `state.pop_agent()`.
- **Multi-hop handoffs**: currently only root → subagent is tested; validate that subagent → subagent chaining works and add a test.
- **Handoff timeout**: if a subagent doesn't produce a final answer within N steps, automatically pop back to the root agent.
- **Handoff rejection**: allow a subagent to reject an incoming handoff if the context payload is missing required fields — emit a `handoff_rejected` event and let root retry with a different specialist.
- **Parallel subagent dispatch**: for tasks with independent subtasks, allow root to fork to multiple subagents concurrently and merge their results.
- **`__handoff__` schema exposure**: optionally include the `__handoff__` pseudo-tool in `tool_schemas` so the model knows it exists without hardcoding its name in the system prompt.
- **Handoff history in memory**: record each handoff (from, to, reason, context) in session memory so future sessions can infer routing patterns.
- **Capability-based routing**: instead of root specifying `target_agent_id` by name, let it specify a required `capability` string and have `HandoffRouter` pick the best-matching registered subagent.
