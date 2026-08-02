# Feature: Permission Gates

**Module:** `src/sutra/core/permissions.py`  
**Classes:** `PermissionGate`, `PermissionRequest`, `PermissionLevel`

## Purpose

Human-in-the-loop (HITL) pause mechanism for high-risk tool calls. When an agent tries to call a tool at or above the approval threshold, the harness emits a `permission_request` SSE event and suspends the run on an `asyncio.Future` until an external actor (a human clicking "approve" in a UI, or an automated system) calls `resolve()`.

## Current Behavior

### `PermissionLevel`

```python
class PermissionLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

Ordered: `LOW(0) < MEDIUM(1) < HIGH(2) < CRITICAL(3)`.

### `PermissionGate`

```python
gate = PermissionGate(require_approval_from=PermissionLevel.HIGH)  # default
```

`requires_gate(level)` returns `True` if `level >= require_approval_from`.  
With defaults: `HIGH` and `CRITICAL` tools require approval; `LOW` and `MEDIUM` execute immediately.

#### Opening a request

```python
request = gate.open_request(
    tool_name="transfer_funds",
    arguments={"amount": 10000, "to": "ext-001"},
    level=PermissionLevel.CRITICAL,
    reason="Tool 'transfer_funds' is classified critical risk and requires human approval.",
    tool_call_id="tc_abc123",
)
```

Creates a `PermissionRequest` with a 12-char hex `request_id`, stores it in `_pending`, and creates an `asyncio.Future` in `_futures`. Returns the request immediately (does **not** await).

#### Resolving

```python
request = gate.resolve(request_id, approved=True, actor="alice")
```

Sets `request.resolved`, `approved`, `resolved_by`, `resolved_at`. Calls `future.set_result(approved)`. Removes from `_pending` and `_futures`.

Raises `KeyError` if `request_id` is not in `_pending`.

#### Awaiting (not used by current harness)

```python
approved = await gate.await_decision(request_id, timeout=300.0)
```

Blocks until `resolve()` is called or timeout expires. On timeout, raises `PermissionDeniedError`.

**Note:** The current harness does **not** use `await_decision`. Instead, `open_request()` is called synchronously, the harness emits `PERMISSION_REQUEST` + `DONE` and returns, leaving the `Future` open. `resume_after_permission()` is the external re-entry point that calls `resolve()` and then continues.

### `PermissionRequest` Fields

| Field | Type | Description |
|---|---|---|
| `request_id` | `str` | 12-char hex, unique per request |
| `tool_name` | `str` | Tool that triggered the gate |
| `arguments` | `Dict` | Arguments the model passed |
| `level` | `PermissionLevel` | Risk level of the tool |
| `reason` | `str` | Human-readable explanation |
| `tool_call_id` | `str` | Correlation ID back to the model's tool call |
| `resolved` | `bool` | Whether `resolve()` has been called |
| `approved` | `Optional[bool]` | Result of the decision |
| `resolved_by` | `Optional[str]` | Actor who resolved it |
| `resolved_at` | `Optional[float]` | Unix timestamp of resolution |

### Integration with `HarnessState`

When gated:
- `state.status = PAUSED_FOR_PERMISSION`
- `state.pending_permission = PendingPermission(request_id, tool_name, arguments, reason, tool_call_id)`
- `state.record_step("permission_wait", tool_name=..., request_id=...)`

This snapshot is included in the `DONE` event payload so any downstream system can re-hydrate the gate state.

## Enhancement Ideas

- **Timeout-based auto-denial**: expose a configurable `request_timeout_seconds` on `PermissionGate`; after the deadline, auto-deny and resume the run without waiting for `resolve()`.
- **Multi-approver support**: require approval from N-of-M designated actors before `resolve()` unblocks, for sensitive operations like large fund transfers.
- **Conditional approval**: allow the approving actor to modify the arguments (e.g. reduce the transfer amount) before approving — `resolve(request_id, approved=True, modified_arguments={...})`.
- **Approval escalation**: if no response within T seconds, route the request to a higher-tier approver (manager → CISO).
- **Persistent pending requests**: currently `_pending` and `_futures` are in-memory; if the process restarts, pending requests are lost. Persist them to Redis or a database so the resume endpoint can still function after a restart.
- **MEDIUM-level soft gate**: add a second threshold (`warn_approval_from`) that emits a `permission_advisory` event (logged but not blocking) for medium-risk tools.
- **Approval reason capture**: allow the actor to include a `reason` string when resolving; store it in `PermissionRequest.resolved_reason` for audit purposes.
- **Rate limiting on approvals**: prevent a single actor from approving more than N requests per minute to guard against social-engineering attacks on the approval UI.
- **Denial with retry**: on denial, instead of `FAILED`, allow the model to try a different (lower-risk) approach by injecting the denial as a tool result and continuing the turn.
