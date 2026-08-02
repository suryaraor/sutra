# Sutra — Architecture Overview

Sutra is a **production-grade, LLM-agnostic asynchronous agent harness**. It threads together every step an agent takes into one coherent, governed run.

## The 10 Architectural Components

| # | Component | Module | Spec |
|---|---|---|---|
| 1 | Agent Harness Loop | `core/harness.py` | [01-harness-loop.md](01-harness-loop.md) |
| 2 | Server-Sent Events | `streaming/sse.py` | [02-sse-streaming.md](02-sse-streaming.md) |
| 3 | Context Compaction | `core/compaction.py` | [03-context-compaction.md](03-context-compaction.md) |
| 4 | Budget Monitor | `core/budget.py` | [04-budget-monitor.md](04-budget-monitor.md) |
| 5 | Subagents | `agents/subagent.py` | [05-subagents.md](05-subagents.md) |
| 6 | Tool Registry | `tools/registry.py` | [06-tool-registry.md](06-tool-registry.md) |
| 7 | Guardrails | `core/guardrails.py` | [07-guardrails.md](07-guardrails.md) |
| 8 | Handoff Protocol | `core/handoff.py` | [08-handoff-protocol.md](08-handoff-protocol.md) |
| 9 | Permission Gates | `core/permissions.py` | [09-permission-gates.md](09-permission-gates.md) |
| 10 | Memory System | `memory/` | [10-memory-system.md](10-memory-system.md) |

## Supporting Features

| Feature | Module | Spec |
|---|---|---|
| Model Client Interface | `models/base.py` | [11-model-client.md](11-model-client.md) |
| HTTP Server | `server/app.py` | [12-http-server.md](12-http-server.md) |
| CLI | `cli/` | [13-cli.md](13-cli.md) |
| Harness State | `core/state.py` | [14-harness-state.md](14-harness-state.md) |
| Hooks | `core/hooks.py` | [15-hooks.md](15-hooks.md) |

## Design Invariants

- **All state lives on `HarnessState`**, never in transient locals — a run can be paused, checkpointed, and resumed.
- **The harness never imports a provider SDK directly** — all LLM calls go through the `ModelClient` interface.
- **Safety layers run in a fixed order**: guardrail → budget check → compaction → model call → permission gate.
- **Every exit path** (guardrail block, budget exceeded, tool error, permission pause, normal completion) yields a terminal `DONE` SSE event.
- **Memory recording is always in `try/finally`** — it fires on every exit path exactly once.

## Data Flow (one turn)

```
user_input
  │
  ▼
Guardrail.sanitize_input()          → GUARDRAIL_BLOCK (if blocked)
  │
  ▼
HarnessState.append_message(USER)
  │
  ▼
loop:
  Budget.check_step()               → BUDGET_EXCEEDED (if over limit)
  ContextCompactor.compact()        → COMPACTION event (if triggered)
  Budget.warnings()                 → BUDGET_WARNING events
  ModelClient.stream()              → TOKEN events (streaming)
  Budget.record_usage()
  HarnessState.append_message(ASST) → MESSAGE_COMPLETE event
  │
  ├─ no tool calls → DONE (completed)
  │
  └─ tool calls:
       ├─ __handoff__ → HANDOFF event, push agent stack
       ├─ CRITICAL/HIGH tool → PERMISSION_REQUEST, halt (PAUSED_FOR_PERMISSION)
       └─ normal tool → Guardrail.sanitize_output() → TOOL_RESULT, loop
```
