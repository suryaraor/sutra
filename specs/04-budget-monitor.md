# Feature: Budget Monitor

**Module:** `src/sutra/core/budget.py`  
**Classes:** `Budget`, `BudgetConfig`, `BudgetUsage`, `ModelPricing`

## Purpose

Enforces hard spending and step limits on a harness run. Prevents runaway agents from burning through API credits, looping forever, or consuming unbounded context by treating each resource dimension as a hard cap that terminates the run cleanly when breached.

## Current Behavior

### Configuration (`BudgetConfig`)

| Field | Default | Description |
|---|---|---|
| `max_usd` | `5.0` | Maximum USD spend across the entire run |
| `max_input_tokens` | `200_000` | Maximum input tokens consumed |
| `max_output_tokens` | `50_000` | Maximum output tokens generated |
| `max_steps` | `40` | Maximum model call steps (loop iterations) |
| `warn_ratio` | `0.75` | Fraction of any limit that triggers a soft `BUDGET_WARNING` SSE event |

### Pricing (`ModelPricing`)

```python
@dataclass
class ModelPricing:
    input_per_million: float   # USD per 1M input tokens
    output_per_million: float  # USD per 1M output tokens
```

Default: `$3.00 / 1M input`, `$15.00 / 1M output` (approximate Sonnet-tier pricing).

### Usage Tracking (`BudgetUsage`)

Accumulates `usd_spent`, `input_tokens`, `output_tokens`, and `steps` across all model calls in a run.

### Core Methods

#### `async check_step() -> None`

Called **before** each model invocation. Increments `usage.steps` and raises `BudgetExceededError(dimension="steps")` if `steps + 1 > max_steps`. Uses an `asyncio.Lock` for coroutine safety.

#### `async record_usage(*, input_tokens, output_tokens) -> float`

Called **after** each model call with actual token counts. Accumulates tokens, computes cost, adds to `usd_spent`. Raises `BudgetExceededError` for `usd`, `input_tokens`, or `output_tokens` if their respective limits are crossed.

Returns the cost of this call in USD.

#### `warnings() -> Dict[str, float]`

Returns `{dimension: ratio_used}` for every dimension whose `used / limit >= warn_ratio`. The harness emits one `BUDGET_WARNING` SSE event per entry before each model call.

#### `snapshot() -> Dict[str, float]`

Returns a serializable snapshot of current usage and limits, attached to `BUDGET_WARNING` events.

### Exception

`BudgetExceededError` (from `core/exceptions.py`) carries `dimension`, `used`, and `limit` fields so the harness can emit a structured `BUDGET_EXCEEDED` SSE event.

### Thread Safety

Both `check_step` and `record_usage` hold `self._lock` (an `asyncio.Lock`) during their read-modify-write cycle, making them safe for concurrent coroutines sharing a single `Budget` instance.

## Enhancement Ideas

- **Per-turn budget**: in addition to run-level limits, add turn-level caps (e.g. max $0.10 per user message) to guard against single expensive prompts.
- **Soft limit with graceful wrap-up**: at `warn_ratio`, give the model a system notice ("you are approaching your budget — wrap up") instead of hard-stopping mid-task.
- **Dynamic pricing table**: load model pricing from a config file or a remote endpoint so it stays accurate as providers change rates.
- **Separate budget per subagent**: allocate a fraction of the total budget to each subagent profile so no single specialist can exhaust the shared pool.
- **Budget reset across sessions**: the current `Budget` is per-harness-instance; add a session-scoped ledger that persists across `run()` calls for multi-turn sessions with a shared USD cap.
- **Cost estimation before execution**: before calling the model, estimate expected token count from the current context window and refuse to call if the estimate would breach the budget.
- **Real-time budget dashboard**: expose `Budget.snapshot()` on a `/budget/{session_id}` HTTP endpoint for monitoring dashboards.
- **Audit log**: record every `record_usage` call with timestamp and agent ID to an append-only log for billing reconciliation.
