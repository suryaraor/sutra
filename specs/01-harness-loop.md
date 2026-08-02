# Feature: Agent Harness Loop

**Module:** `src/sutra/core/harness.py`  
**Class:** `AsynchronousHarnessLoop`

## Purpose

The primary runtime environment. Orchestrates a single agent task end-to-end: guardrail-checks the user's input, streams model output as SSE events, dispatches tool calls (routing high-risk ones through the Permission Gate), applies context compaction when the token window grows too large, honors subagent handoffs, and enforces the Budget on every step.

## Current Behavior

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_client` | `ModelClient` | required | LLM provider adapter |
| `tool_registry` | `ToolRegistry` | required | Registered callable tools |
| `budget` | `Budget` | required | Hard spend/step limits |
| `guardrail` | `Guardrail` | default Guardrail() | Input/output sanitizer |
| `compactor` | `ContextCompactor` | default compactor | Token-window manager |
| `subagent_registry` | `SubagentRegistry` | empty registry | Known specialist agents |
| `permission_gate` | `PermissionGate` | default gate | HITL pause mechanism |
| `memory` | `MemoryManager` | None | Persistent memory (optional) |
| `user_id` | `str` | `"anonymous"` | Identity for memory scoping |
| `system_prompt` | `str` | default helpful agent | Root agent system prompt |
| `state` | `HarnessState` | new state | Resume from checkpoint |
| `max_tool_hops_per_turn` | `int` | `8` | Loop guard per turn |

### Public API

#### `async def run(user_input: str) -> AsyncIterator[SSEEvent]`

Drives one full user turn to completion. Yields SSE events as it progresses through each phase.

**Turn execution order:**
1. Emit `SESSION_START`
2. `Guardrail.sanitize_input()` — emit `GUARDRAIL_BLOCK` + `DONE` and return if blocked
3. Append `USER` message to state
4. **Inner loop** (up to `max_tool_hops_per_turn`):
   a. `Budget.check_step()` — raises `BudgetExceededError` if over limit
   b. `ContextCompactor.compact()` — emit `COMPACTION` if triggered
   c. Emit `BUDGET_WARNING` for any dimension past the warn ratio
   d. `ModelClient.stream()` — emit `TOKEN` events per delta
   e. `Budget.record_usage()` — raises `BudgetExceededError` if over limit
   f. Append `ASSISTANT` message, emit `MESSAGE_COMPLETE`
   g. If no tool calls → set `COMPLETED`, break
   h. Dispatch each tool call:
      - `__handoff__` → `HandoffRouter`, emit `HANDOFF`, push agent stack
      - CRITICAL/HIGH tool → `PermissionGate`, emit `PERMISSION_REQUEST`, emit `DONE`, return
      - Normal tool → execute, `sanitize_output`, append `TOOL_RESULT`, emit `TOOL_RESULT`
5. Emit `DONE` with final status
6. `finally:` always calls `_record_memory()`

#### `async def resume_after_permission(request_id, *, approved, actor) -> AsyncIterator[SSEEvent]`

External entry point to unblock a paused Permission Gate.

- Calls `PermissionGate.resolve()`
- Emits `PERMISSION_RESOLVED`
- If denied: appends denial tool message, emits `DONE (failed)`, returns
- If approved: executes the tool, sanitizes output, appends result, calls `self.run("")` to continue

### Key Internal Methods

- `_dispatch_tool_call()` — routes a single tool call to handoff, permission gate, or direct execution
- `_handle_handoff()` — validates target, builds briefing, pushes agent stack
- `_effective_system_prompt()` — composes root prompt + subagent overlay + memory context block
- `_messages_for_wire()` — converts `HarnessState.messages` to OpenAI/Ollama flat format
- `_stringify_tool_result()` — coerces any tool return value to a string for appending

### Exit Statuses

| Status | Trigger |
|---|---|
| `COMPLETED` | Model responded with no tool calls |
| `FAILED` | Guardrail block, model error, tool error, handoff to unknown agent, permission denied |
| `BUDGET_EXCEEDED` | Any budget dimension hard limit crossed |
| `PAUSED_FOR_PERMISSION` | CRITICAL/HIGH tool intercepted — waiting for `resume_after_permission` |

## Enhancement Ideas

- **Checkpointing**: persist `HarnessState.to_json()` to disk/Redis after each hop so restarts don't replay from scratch.
- **Parallel tool dispatch**: when a model returns multiple tool calls with no inter-dependency, dispatch them concurrently (currently sequential).
- **Retry on transient model errors**: wrap `ModelClient.stream()` with exponential backoff for rate-limit / 5xx responses.
- **Max turn limit**: add a `max_turns` parameter to cap multi-turn conversations (separate from `max_tool_hops_per_turn` which is per-turn).
- **Tool call timeout**: per-tool execution timeout so a slow tool doesn't block the loop indefinitely.
- **Agent stack pop on handoff completion**: currently agent is pushed but never automatically popped when subagent is done — add explicit `handoff_back` pseudo-tool or auto-pop when subagent has no more tool calls.
- **Streaming tool result summaries**: instead of injecting raw tool output as a user-visible `TOOL_RESULT` event, optionally pipe through a summarizer before feeding back to the model.
