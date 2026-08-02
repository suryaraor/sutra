# Feature: Agent-as-Tool (`__consult__`)

**Module:** `src/sutra/core/harness.py`
**Methods:** `AsynchronousHarnessLoop._handle_consult`, `_build_consult_prompt`, `_finish_consult`
**Related:** `src/sutra/core/handoff.py`, `src/sutra/agents/subagent.py`, `specs/08-handoff-protocol.md`

## Purpose

`__handoff__` (spec 08) transfers control to a subagent *permanently* — the calling agent's frame is gone; `active_agent_id` changes and stays changed until something hands back or the run ends.

`__consult__` is the complementary primitive: it lets the *currently active* agent ask a specialist subagent a single question and get an answer back as an ordinary tool result, without losing its own turn. Think "function call to another agent" rather than "transfer of control." After a consult, `self.state.active_agent_id` and `self.state.agent_stack` are exactly as they were before it — whether the consult succeeded or failed.

## The `__consult__` Pseudo-Tool

Like `__handoff__`, `__consult__` is **not** in the `ToolRegistry` in any functional sense — the harness intercepts calls to `__consult__` by name in `_dispatch_tool_call()` before the registry lookup ever happens. It's still registered as a real `Tool` (see `src/sutra/toolkits/contoso_demo.py::build_tool_registry()`) with a placeholder handler (`_consult_placeholder`) that raises if ever actually invoked — that registration exists purely so its JSON schema is visible to the model via `tool_registry.schemas_for(...)`.

Input schema:

```json
{
  "target_agent_id": "string (required)",
  "question": "string (required)",
  "context_payload": "object (optional)"
}
```

The tool's `description` (in `contoso_demo.py`) explicitly tells the model the difference from `__handoff__`, so it can choose correctly: consult when you just need a specialist's input and plan to keep working yourself; hand off when the specialist should take the request over entirely.

## Dispatch Flow

1. Model calls `__consult__` with `{target_agent_id, question, context_payload}`.
2. `_dispatch_tool_call()` intercepts the name (same spot as `HANDOFF_TOOL_NAME`) and routes to `_handle_consult(arguments, from_agent_id, tool_call_id)`.
3. `CONSULT_START` SSE event is emitted (`target_agent`, `question`).
4. `target_agent_id` is validated against `self.subagent_registry.has(...)`. If missing, a `ConsultError` is raised and caught **locally** (never propagates) — see "Failure handling" below.
5. The target's effective system prompt is built via the existing `_effective_system_prompt(target_agent_id)` (root prompt + `[Active subagent: ...]` overlay + memory context, unchanged from how the main loop builds it for a subagent turn).
6. A **private, local** wire-format message list is seeded with a single synthesized user turn combining `question` and (if present) `context_payload`, built by `_build_consult_prompt(...)`. This list lives only in a local variable — it is never written to `self.state.messages`.
7. A bounded nested loop (default cap: `self.max_consult_hops = 5`, a constructor param) repeats:
   - `await self.budget.check_step()` then `await self.budget.record_usage(...)` around `self.model_client.stream(...)`, called with the target agent's tool schemas (`tool_registry.schemas_for(target_agent_id, subagent_registry.allowed_tools_for(target_agent_id))`) — **exactly** the same two budget calls, in the same order, that the main loop makes (see "Budget sharing" below).
   - If the model call returns no tool calls, its text is the final answer — loop ends successfully.
   - If it returns tool calls, each is looked up in `self.tool_registry`, checked against `self.permission_gate.requires_gate(...)`, and (if not gated) executed via `self.tool_registry.execute(...)`. Results are appended to the *local* message list only, and the loop continues for another hop.
   - Hitting `self.max_consult_hops` without a final answer ends the loop as a **failure**.
8. `_finish_consult(...)` runs the outcome text (success answer or failure text) through `self.guardrail.sanitize_output(...)` (identically to how `_dispatch_tool_call` treats any other tool's result), appends it to `self.state.messages` as `Message(role=Role.TOOL, name=CONSULT_TOOL_NAME, tool_call_id=...)`, and emits `TOOL_RESULT` followed by `CONSULT_END` (`target_agent`, plus `answer` on success or `error` on failure).
9. The outer `run()` loop resumes exactly as after any other tool dispatch, feeding the new tool-result message back to the *calling* agent's model turn. `active_agent_id` / `agent_stack` were never touched.

## Design Decision 1: Consult Failure Handling

**A failed consult never fails the run.** Whether the failure is an unknown `target_agent_id`, a nested tool raising, or the hop limit being hit, `_handle_consult` always:

- emits an `EventType.ERROR` event (`phase: "consult"`) so the failure is observable in the event stream, and
- emits `CONSULT_END` with an `error` field, and
- appends a normal `Message(role=Role.TOOL, ...)` containing the error text to the *calling* agent's conversation,

but it deliberately does **not** set `self.state.status = HarnessStatus.FAILED`. The run keeps going; the calling model sees a tool result that happens to describe a failure and can react to it (retry with different arguments, try a different subagent, apologize to the user, whatever it decides) exactly the way it would react to, say, a `{"error": "unknown ticket"}` result from `check_ticket_status`. This treats "the specialist I asked couldn't answer" as *data for the calling agent*, not as a reason to kill the whole turn — consistent with `__consult__` being a lower-stakes, non-control-transferring primitive than `__handoff__` (whose `HandoffError` *does* still fail the run, since a bad handoff leaves the run with no sensible active agent to continue as).

## Design Decision 2: Permission-Gated Tools Inside a Consult Are Unsupported

If the nested subagent's model call requests a tool whose `permission_level` trips `self.permission_gate.requires_gate(...)` (HIGH/CRITICAL by default), the harness does **not** attempt to open a real permission gate from inside the nested loop.

Why: `PermissionGate` pauses the *whole run* on an `asyncio.Future` (see spec 09) — `_dispatch_tool_call` sets `state.status = PAUSED_FOR_PERMISSION`, `run()` sees that status and returns its generator entirely, and later `resume_after_permission()` is a completely separate external entry point that re-drives `run("")` from scratch. That pause/resume shape assumes the *outer* generator is what's suspended and later resumed. `_handle_consult`'s nested loop is a synchronous (from the outer loop's perspective — it's still `async`, but it doesn't yield control back to `run()` mid-consult) sub-conversation with no equivalent "come back later" hook: there is no sensible way to half-suspend a consult, return through `_dispatch_tool_call` and `run()`, and later resume a `for` loop buried three call-stack frames down inside a specific tool-call's nested loop.

So instead: reaching a gated tool inside a consult is treated as a **failure of that consult**, exactly like an unknown target agent or a hop-limit overrun. The error text explicitly tells the calling agent that the consultation required a permission-gated action and is unsupported here, and suggests using `__handoff__` instead — because `__handoff__` *does* compose cleanly with `PermissionGate`: it transfers control to the subagent for real, so when that subagent later calls the gated tool, the pause happens in the *main* loop where the existing `resume_after_permission()` mechanism already works correctly.

This is an intentional scope boundary, not an oversight: `__consult__` is for "ask a specialist a low-stakes question," not "let a specialist perform a dangerous action on my behalf without me even knowing it happened."

## Budget Sharing (load-bearing guarantee)

Every nested model call inside `_handle_consult` goes through the exact same `self.budget` instance the main loop uses — `await self.budget.check_step()` immediately before each `self.model_client.stream(...)` call, `await self.budget.record_usage(input_tokens=..., output_tokens=...)` immediately after, in the same order the main loop uses them. There is no separate budget, no higher limit, no bypass: a run that calls `__consult__` five times, each burning three nested hops, spends step/token/USD budget exactly as if those had been fifteen ordinary main-loop turns.

`BudgetExceededError` raised by either call is **not** caught anywhere inside `_handle_consult`. It propagates up through the `async for` in `_dispatch_tool_call` (which also does not catch it) into `run()`'s own `except BudgetExceededError` handler around the main `while True:` loop — the same handler that already existed for the main loop's own `check_step()`/`record_usage()` calls. The net effect: a budget overrun during a consult ends the whole run's turn with `HarnessStatus.BUDGET_EXCEEDED`, indistinguishable in outcome from an overrun that happened in the main loop. This is intentional — a consult must never be usable as a way to spend beyond the run's configured limits.

## State Isolation

- `local_messages` (the nested wire-format list) is a plain local `List[Dict[str, Any]]` inside `_handle_consult` — it is never assigned to or merged into `self.state.messages`. The calling agent's visible history gains exactly one new entry: the final `Message(role=Role.TOOL, name=CONSULT_TOOL_NAME, ...)` appended by `_finish_consult`.
- `self.state.active_agent_id` / `self.state.agent_stack` are never touched by `_handle_consult` — no `push_agent`/`pop_agent` call appears anywhere in the consult path. Contrast with `_handle_handoff`, which calls `state.push_agent(...)` unconditionally on success.
- `self.state.record_step(...)` *is* called (`"consult_model_call"` per nested hop, `"consult"` once at the end) — these go to `execution_steps`, a separate audit trail from `messages`, so recording them doesn't violate the "private aside" guarantee for the conversation itself.

## SSE Events

- `CONSULT_START` — emitted the moment `_handle_consult` begins. Fields: `target_agent`, `question`.
- `CONSULT_END` — emitted once, at the end of `_handle_consult`, on every exit path (success or failure). Fields: `target_agent`, plus `answer` (success) or `error` (failure).
- `ERROR` (`phase: "consult"`) — emitted additionally, right before `CONSULT_END`, only on failure paths.
- `TOOL_CALL` / `TOOL_RESULT` — the calling agent still sees the standard `TOOL_CALL` event (emitted generically for any tool name at the top of `_dispatch_tool_call`, before the `__consult__` interception check) and a `TOOL_RESULT` event once `_finish_consult` appends the outcome message — from the calling agent's point of view, `__consult__` looks exactly like any other tool call.

No rendering support was added to `src/sutra/cli/render.py` for `CONSULT_START`/`CONSULT_END` — out of scope for this change; the events are emitted correctly and available to any consumer of the SSE stream, but the CLI will currently just ignore the unrecognized event types.

## Enhancement Ideas

- **Consult depth limit across nested consults**: if a consulted subagent itself has `__consult__` in its `allowed_tools`, nothing currently stops it from consulting a third agent, which could consult a fourth, etc. Each level enforces its own `max_consult_hops`, but there's no global recursion-depth cap — add one if multi-level consults are ever wired up.
- **Streaming consult tokens**: currently the nested loop accumulates `delta_text` silently and never yields `TOKEN` events, so a consult looks "quiet" to any UI streaming the SSE feed until it resolves. A future version could yield token deltas tagged with the target agent id so a UI can show "IT ops agent is thinking..." live.
- **Consult result caching**: identical `(target_agent_id, question, context_payload)` consults within the same run currently always re-run the nested loop; caching would save budget on repeated questions.
- **Partial gated-tool support via auto-deny**: instead of always failing the consult outright when a gated tool is requested, optionally auto-deny with a synthetic tool result fed back into the nested loop, letting the specialist try a different (ungated) approach before giving up — currently it fails immediately on the first gated call.
- **`__consult__` schema exposure control**: right now any agent whose `allowed_tools` includes `__consult__` can consult *any* registered subagent, including itself (`from_agent_id == target_agent_id`, currently not disallowed and would just run a redundant nested loop). Consider validating against self-consultation or restricting which target ids a given consulting agent may name.
