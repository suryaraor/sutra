# Feature: Hooks

**Module:** `src/sutra/core/hooks.py`  
**Classes:** `HookRegistry`, `HookContext`, `HookResult`

## Purpose

A generalized lifecycle-interception mechanism. Where `Guardrail` (spec 07) is a fixed regex text sanitizer and `PermissionGate` (spec 09) is a fixed whole-tool-call approval pause, `Hooks` let an operator register arbitrary async callables against five named points in the harness loop — for audit logging, custom veto logic, argument rewriting, or metrics — **without modifying harness code**.

A harness built without `hooks=` behaves identically to a harness with no hooks feature at all: every call site in `harness.py` guards with `if self.hooks is not None:` before touching the registry, so `hooks=None` (the default) is a strict no-op.

## Current Behavior

### Hook Points

Named as plain string constants (not an `Enum`) so a hook can be registered against a bare string like `"pre_tool_call"` without importing a type — mirrors the string-literal `action` field already used on `ExecutionStep` in `state.py`.

| Point | Fires | Populated `HookContext` fields | Veto meaningful? | `modified_arguments` meaningful? |
|---|---|---|---|---|
| `pre_tool_call` | Start of `_dispatch_tool_call()`, for **every** tool call the harness sees — including the `__handoff__` pseudo-tool (see [Ordering Decision](#ordering-decision-pre_tool_call-vs-__handoff__) below) | `tool_name`, `arguments` | Yes — aborts the dispatch entirely | Yes — rewrites `arguments` before the tool (or handoff) runs |
| `post_tool_call` | After a normal (non-gated) tool call **successfully** executes, once its result is appended to state | `tool_name`, `arguments`, `result` | No — the tool already ran; a `veto=True` here is ignored | No |
| `pre_model_call` | Top of each model-call iteration of `run()`'s `while True:` loop, right after context compaction / budget-warning checks and immediately before `self.model_client.stream(...)` is invoked | (only `point`, `state`) | No — side-effect only, ignored | No |
| `on_handoff` | Inside `_handle_handoff()`, after the target agent id is validated but **before** `state.push_agent(...)` runs | `target_agent_id`, `reason` | Yes — blocks the handoff | No |
| `on_done` | The same `finally:` block that already calls `_record_memory()`, in both `run()` and `resume_after_permission()` — see [on_done firing rules](#on_done-firing-rules) | (only `point`, `state`) | No — the run has already ended, nothing left to veto | No |

### `HookContext`

One flexible dataclass with optional fields, not a class hierarchy — the fields populated depend on which point fired:

```python
@dataclass
class HookContext:
    point: str
    state: HarnessState
    tool_name: Optional[str] = None
    arguments: Optional[dict] = None
    result: Optional[Any] = None
    target_agent_id: Optional[str] = None
    reason: Optional[str] = None
```

### `HookResult`

```python
@dataclass
class HookResult:
    veto: bool = False
    veto_reason: Optional[str] = None
    modified_arguments: Optional[dict] = None
```

A hook returns `None` (no opinion) or a `HookResult`. `veto` / `veto_reason` only matter at `pre_tool_call` and `on_handoff` (see table above) — elsewhere they are read and discarded. `modified_arguments` only ever matters at `pre_tool_call`.

### `HookRegistry`

```python
registry = HookRegistry()
registry.register("pre_tool_call", my_audit_hook)
registry.register("pre_tool_call", my_veto_hook)   # multiple hooks per point are fine

harness = AsynchronousHarnessLoop(..., hooks=registry)
```

```python
async def register(self, point: str, hook: Callable[[HookContext], Awaitable[Optional[HookResult]]]) -> None
async def run(self, point: str, context: HookContext) -> Optional[HookResult]
```

`run()` semantics:

- No hooks registered for `point` → cheap no-op, returns `None` immediately (no allocation, no iteration).
- Hooks run **in registration order**.
- The moment a hook returns `HookResult(veto=True, ...)`, iteration stops — remaining hooks for that point do **not** run — and that result is returned immediately. This mirrors how a guardrail block halts further processing.
- If no hook vetoes, **every** hook still runs (for side effects like audit logging or metrics). The last non-`None` `HookResult` returned by a non-vetoing hook is returned to the caller — this is how a `pre_tool_call` hook communicates `modified_arguments` back to the harness even though nothing was vetoed. If every hook returns `None`, `run()` returns `None`.

### Ordering Decision: `pre_tool_call` vs. `__handoff__`

`_dispatch_tool_call()` fires `pre_tool_call` **before** the `tool_name == HANDOFF_TOOL_NAME` check. This means:

- `pre_tool_call` sees and can veto/rewrite handoff attempts too, not just "real" tool dispatches.
- A veto at `pre_tool_call` for a handoff call and a veto at the dedicated `on_handoff` point produce the same externally-observable result (`FAILED` status, run halted) but different SSE events (`hook_veto` vs. `error`) and fire at different points in the call stack — `pre_tool_call` fires in `_dispatch_tool_call()` before `_handle_handoff()` is even entered; `on_handoff` fires inside `_handle_handoff()`, after target validation.
- This is intentionally overlapping, not exclusive: an operator who wants one universal audit/veto point for "the model is about to do *anything*" can use `pre_tool_call` alone; an operator who only cares about handoffs specifically (and wants access to the validated `target_agent_id` / `reason`) can use `on_handoff` instead, or both together for defense in depth.

The alternative (firing `pre_tool_call` only for genuine tool dispatches, after the handoff check) was considered and rejected — it would mean an operator wanting a single choke point over every model-initiated action would have to register at two hook points instead of one.

### `on_done` Firing Rules

`on_done` is added to the exact same `finally:` block that already runs `_record_memory()`, in both `run()` and `resume_after_permission()`, via a shared `_fire_on_done()` helper — so it's structurally guaranteed to be *considered* on every exit path.

Unlike memory recording (an idempotent snapshot overwrite — recording twice is harmless), invoking `on_done` twice for what a caller perceives as *one* logical turn would double-fire a hook, which is a real, observable bug for hooks used for metrics or once-per-turn side effects. Two rules keep it to exactly one fire per logical run-to-completion:

1. **Skip while paused.** `PAUSED_FOR_PERMISSION` is not a terminal status — the turn hasn't concluded, it's waiting on an external decision — so `_fire_on_done()` is a no-op whenever `state.status == PAUSED_FOR_PERMISSION`. It fires later, when the turn actually resolves.
2. **Skip in `resume_after_permission()`'s own `finally` if it delegated into a nested `run()`.** On the approved path, `resume_after_permission()` executes the gated tool and then falls through into `async for evt in self.run(""): yield evt` to let the model keep going. That nested `run()` call has its own `finally` and will fire `on_done` itself once it reaches a terminal status. `resume_after_permission()` tracks this with a local `delegated_to_nested_run` flag and skips its own `on_done` call when set, to avoid a second fire for the same completion. On the denial path (and on a `ToolExecutionError` while re-executing the gated tool), there is no delegation, so `resume_after_permission()`'s own `finally` is the one that fires `on_done`.

Net effect, matching the test suite in `tests/test_hooks.py`:

- A simple run with no permission gate: `on_done` fires exactly once (normal completion path in `run()`).
- A run that hits a permission gate, pauses, and is later resolved via `resume_after_permission()` (approved or denied): `on_done` fires exactly once for the whole pause → resume sequence, not zero and not twice.

## API

```python
from sutra.core.hooks import HookRegistry, HookContext, HookResult, PRE_TOOL_CALL, ON_HANDOFF, ON_DONE

async def audit_every_tool_call(ctx: HookContext) -> None:
    print(f"[audit] {ctx.tool_name} called with {ctx.arguments}")

async def block_prod_db(ctx: HookContext) -> Optional[HookResult]:
    if ctx.arguments and ctx.arguments.get("target") == "prod-db":
        return HookResult(veto=True, veto_reason="prod-db is off-limits without a change ticket.")
    return None

async def redact_secret_arg(ctx: HookContext) -> Optional[HookResult]:
    if ctx.arguments and "api_key" in ctx.arguments:
        cleaned = dict(ctx.arguments)
        cleaned["api_key"] = "[stripped by hook]"
        return HookResult(modified_arguments=cleaned)
    return None

registry = HookRegistry()
registry.register(PRE_TOOL_CALL, audit_every_tool_call)
registry.register(PRE_TOOL_CALL, block_prod_db)
registry.register(PRE_TOOL_CALL, redact_secret_arg)

harness = AsynchronousHarnessLoop(..., hooks=registry)
```

### `EventType.HOOK_VETO`

Added to `src/sutra/streaming/sse.py`'s `EventType` enum (value `"hook_veto"`, matching the enum's existing snake_case-value naming convention) rather than reusing `EventType.ERROR`, so a UI or log consumer can distinguish "a hook operator explicitly vetoed this" from "something broke." Emitted only for a `pre_tool_call` veto. An `on_handoff` veto reuses `EventType.ERROR` with `phase="handoff"`, to stay consistent with how every other handoff failure (`HandoffError`) is already reported.

## Enhancement Ideas

- **`pre_model_call` veto**: currently side-effect only; add support for a hook to abort the turn before the model is even called (e.g. a rate-limiter or a "business hours only" gate), mirroring how `pre_tool_call` can already veto.
- **Priority / ordering control**: `register()` currently only supports append-order execution. Add an optional `priority` so, e.g., a security veto hook can be guaranteed to run before an audit-logging hook regardless of registration order.
- **Async fan-out / timeout**: `run()` awaits each hook sequentially; for hooks with I/O (e.g. calling out to a SIEM), add an optional per-hook timeout so a slow or hung hook can't stall the whole harness turn.
- **`unregister()` / scoped hooks**: no way to remove a hook once registered, or to scope a hook to a single subagent (`agent_id`) rather than globally.
- **Structured hook errors**: a hook callable that raises today will propagate straight out of `HookRegistry.run()` and crash the harness turn. Consider catching hook exceptions, emitting a `hook_error` SSE event, and continuing (or treating it as an implicit veto) rather than letting a buggy hook take down the run.
- **`pre_tool_call` veto for permission-gated tools**: currently `pre_tool_call` fires before the permission-gate check, so a vetoed tool never even reaches `PermissionGate`. Document/expose this ordering explicitly as a way to pre-filter what's even eligible for human approval.
- **Post-handoff hook**: add a matching `post_handoff` point (mirroring `post_tool_call`) that fires after `push_agent()` succeeds, for hooks that want to react to a completed handoff rather than gate it.
- **Metrics helper**: ship a small built-in hook (e.g. `sutra.core.hooks.counting_hook()`) that wraps a `collections.Counter` for the common "count events per point" use case, so operators don't have to hand-roll closures for basic metrics.
