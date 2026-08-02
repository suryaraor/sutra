# Getting Started with Sutra (सूत्र)

> *Sutra* (Sanskrit: "thread") is the string that holds beads in place on a mala. This harness is
> that thread for LLM agents: it takes a raw, non-deterministic model and threads every step it
> takes — thinking, calling tools, asking a human for permission, handing off to a specialist,
> running out of budget — into one coherent, observable, governed run.

This guide is a complete walkthrough of the Sutra harness: what it is, why it exists, every
architectural piece inside it, how to plug in *any* LLM provider (with a full worked example using
a local `gpt-oss:20b` model via Ollama), and how to build your own agent on top of it from scratch.

## Table of contents

1. [What Sutra is, and why it exists](#1-what-sutra-is-and-why-it-exists)
2. [Core agentic principles](#2-core-agentic-principles)
3. [Installation](#3-installation)
4. [5-minute quickstart](#4-5-minute-quickstart)
5. [Architecture tour — the 10 components](#5-architecture-tour--the-10-components)
6. [Connecting to any model](#6-connecting-to-any-model)
7. [Worked example: running gpt-oss:20b through Sutra](#7-worked-example-running-gpt-oss20b-through-sutra)
8. [Serving over HTTP (FastAPI + SSE)](#8-serving-over-http-fastapi--sse)
9. [Build your own agent, step by step](#9-build-your-own-agent-step-by-step)
10. [Testing your agent](#10-testing-your-agent)
11. [Design principles worth stealing](#11-design-principles-worth-stealing)
12. [Troubleshooting](#12-troubleshooting)
13. [Reference cheat-sheet](#13-reference-cheat-sheet)

---

## 1. What Sutra is, and why it exists

A raw call to an LLM is just text in, text out (or token stream in, token stream out). The moment
you want that model to *act* — call a tool, spend money, transfer funds, delete a record, hand a
task to a specialist, run for more than a few turns — you need infrastructure around it that the
model itself does not provide:

| Problem with a raw LLM loop | What happens without governance | Sutra's answer |
|---|---|---|
| The model can be prompt-injected | It silently follows attacker instructions embedded in tool output or user input | **Guardrails** — regex-based block/redact layer on both input and output |
| The model can call anything, blindly | A `DELETE` or `transfer_funds` call executes with no human in the loop | **Permission Gates** — high-risk tools pause the run until a human approves |
| The model can loop or spend forever | Runaway costs, infinite tool-call loops | **Budget Monitor** — hard USD / token / step ceilings, enforced mid-run |
| The conversation grows without bound | Context window overflow, degraded quality, higher cost | **Context Compaction** — token-aware summarization that preserves recent turns |
| One system prompt has to do everything | A single agent that knows "a bit of everything" is unsafe and unfocused | **Subagents + Handoff Protocol** — specialized agents with their own prompt and tool whitelist |
| State lives in local variables | A paused run (e.g. waiting on approval) can't survive a process restart | **HarnessState** — a single serializable object that is the sole source of truth |
| Swapping model providers means rewriting the agent | Vendor lock-in | **`ModelClient` interface** — the harness never imports a provider SDK directly |
| The frontend needs to show progress, not just a final answer | No visibility into what the agent is doing mid-turn | **SSE streaming** — every meaningful event (token, tool call, budget warning, handoff, permission request...) is emitted live |

Sutra is **not** a model, a prompt library, or an agent *framework* with opinions about what your
agent should say. It is the **runtime scaffolding** — the orchestration layer — that turns
"an LLM with tools" into "a production-grade software agent." It is deliberately small,
dependency-light (`pydantic` + `typing-extensions` in the core), and 100% LLM-agnostic.

---

## 2. Core agentic principles

Sutra is an implementation of a few principles that show up in every serious agent system:

1. **State is a first-class, serializable object — never scattered locals.**
   Everything about a run — messages, execution steps, active agent, pending permission,
   compaction count — lives on `HarnessState`. That's what makes "pause for 20 minutes waiting on
   a human, then resume" possible at all: `HarnessState.to_json()` / `from_json()` round-trips the
   entire run.

2. **Every unsafe action must be interceptable *before* it executes, not just logged after.**
   The Guardrail runs *before* the model ever sees a user's message. The Permission Gate runs
   *before* a high-risk tool executes, not after. Budget checks happen *before* the next step is
   allowed to start. Governance that only observes after the fact is not governance.

3. **The harness never trusts the model to self-regulate.**
   The model is not asked nicely to "be careful with tools" — the registry enforces required
   arguments, the permission gate enforces approval, the budget enforces hard ceilings regardless
   of what the model "intends" to do next.

4. **Specialization beats a single monolithic prompt.**
   A subagent only ever sees the tools its profile whitelists (`allowed_tools`). A finance
   subagent physically cannot call a tool that isn't in its allow-list — this is enforced by
   `ToolRegistry.schemas_for()` filtering, not by asking the model to behave.

5. **Provider-agnosticism is an architectural boundary, not an afterthought.**
   `AsynchronousHarnessLoop` only ever calls `model_client.stream(...)`. It has never heard of
   Anthropic, OpenAI, or Ollama. This is what lets you develop against a free, deterministic
   `MockModelClient` and flip a single constructor argument to go live against a real provider —
   local or hosted — with zero changes to orchestration code.

6. **Observability is built into the loop, not bolted on.**
   Every phase of a turn — guardrail block, token, tool call, tool result, budget warning,
   compaction, handoff, permission request/resolution, error, done — is a typed `SSEEvent`. You
   get a live, structured trace of the agent's reasoning process for free, which is what a
   frontend (or an audit log) needs.

7. **Fail closed, not open.**
   Guardrail input violations block by default. Budget overruns halt the run rather than let it
   silently continue. A tool exception is caught and turned into a controlled `FAILED` status
   rather than crashing the process.

---

## 3. Installation

Requires **Python 3.11+**.

```bash
cd sutra
python -m venv .venv
.venv/Scripts/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
# source .venv/bin/activate   # macOS / Linux

pip install -e ".[dev]"       # core + pytest
```

Optional extras, install only what you need:

| Extra | Installs | Needed for |
|---|---|---|
| `.[anthropic]` | `anthropic` SDK | `AnthropicModelClient` |
| `.[openai]` | `openai` SDK | `OpenAIModelClient` |
| `.[ollama]` | `httpx` | `OllamaModelClient` (local models, e.g. `gpt-oss:20b`) |
| `.[tokens]` | `tiktoken` | precise token counting in `ContextCompactor` (falls back to a `len(text)//4` heuristic without it) |
| `.[server]` | `fastapi`, `uvicorn` | serving the harness over HTTP/SSE |
| `.[dev]` | `pytest`, `pytest-asyncio` | running the test suite |

You can combine extras: `pip install -e ".[ollama,server,dev]"`.

Verify the install:

```bash
pytest
```

---

## 4. 5-minute quickstart

The fastest way to see every component fire in one run is the bundled finance-ops example, which
uses the network-free `MockModelClient` (a scripted, deterministic stand-in for a real model) so
it runs anywhere without credentials or a GPU:

```bash
python examples/finance_workflow.py
```

Watch for, in order:

1. A prompt-injection attempt **blocked by the Guardrail** before it reaches the model.
2. A legitimate request streaming as SSE `token` events.
3. A `budget_warning` event (soft threshold crossed) and a `compaction` event (context folded).
4. A **handoff** from the root triage agent to a `finance_ops_agent` subagent.
5. A CRITICAL-risk `transfer_funds` call **halted by the Permission Gate** — the run stops and
   emits `permission_request`.
6. A simulated human operator approving it via `resume_after_permission(...)`, completing the turn.

That one script is a tour of components #1, #2, #3, #4, #6, #8, #9, and #10 from the table below,
end to end.

---

## 5. Architecture tour — the 10 components

```
src/sutra/
├── core/
│   ├── state.py         # HarnessState, Message, ExecutionStep, PendingPermission
│   ├── budget.py         # Budget, BudgetConfig, ModelPricing
│   ├── compaction.py    # ContextCompactor, CompactionConfig
│   ├── guardrails.py    # Guardrail, GuardrailConfig, GuardrailRule
│   ├── permissions.py   # PermissionGate, PermissionLevel, PermissionRequest
│   ├── handoff.py       # HandoffRouter, HandoffRequest
│   ├── harness.py        # AsynchronousHarnessLoop  <- the engine
│   └── exceptions.py    # SutraError hierarchy
├── tools/registry.py     # Tool, ToolParameter, ToolRegistry
├── agents/subagent.py    # SubagentProfile, SubagentRegistry
├── models/                # ModelClient ABC + Anthropic/OpenAI/Ollama/Mock implementations
├── streaming/sse.py       # SSEEvent, EventType, format_sse
└── server/app.py         # FastAPI integration
```

| # | Component | Module | What it guarantees |
|---|---|---|---|
| 1 | **Agent Harness** | `core/harness.py` → `AsynchronousHarnessLoop` | Orchestrates one full turn: guardrail → budget → compaction → model stream → tool dispatch → loop or complete. All state lives on `self.state`. |
| 2 | **Server-Sent Events** | `streaming/sse.py` → `SSEEvent` | Every phase of a turn is a typed, timestamped, ID'd wire event ready for `text/event-stream`. |
| 3 | **Context Compaction** | `core/compaction.py` → `ContextCompactor` | Token-window monitor; folds everything except the system prompt and the last N turns into one summary message. |
| 4 | **Budget Monitor** | `core/budget.py` → `Budget` | Hard ceilings on USD, input tokens, output tokens, step count — checked *before* spend where possible. |
| 5 | **Subagents** | `agents/subagent.py` → `SubagentRegistry` | Named profiles with their own system prompt and a restricted tool whitelist. |
| 6 | **Custom Tools** | `tools/registry.py` → `ToolRegistry` | Schema generation (Anthropic/OpenAI-compatible), required-argument validation, async execution, per-agent visibility filtering. |
| 7 | **Streaming** | `models/base.py` + `harness.py` | Token-by-token `StreamChunk`s from the provider are piped straight into `TOKEN` SSE events as they arrive. |
| 8 | **Guardrails** | `core/guardrails.py` → `Guardrail` | Regex-based prompt-injection **blocking** on input, PII **redaction** on output. |
| 9 | **Handoff Protocol** | `core/handoff.py` → `HandoffRouter` | A reserved `__handoff__` tool lets any agent transfer control to a specialist, with a generated briefing message. |
| 10 | **Permission Gates** | `core/permissions.py` → `PermissionGate` | High-risk tool calls block on an `asyncio.Future` until an external actor calls `resolve()`. |

### 5.1 HarnessState — the single source of truth

```python
from sutra import HarnessState, Message

state = HarnessState(system_prompt="You are a helpful assistant.")
# state.session_id, state.status, state.messages, state.execution_steps,
# state.active_agent_id, state.agent_stack, state.pending_permission, ...

# Fully serializable — this is what makes a permission-gate pause durable:
blob = state.checkpoint()          # -> JSON string
restored = HarnessState.from_json(blob)
```

`HarnessStatus` is one of: `idle`, `running`, `paused_for_permission`, `paused_for_handoff`,
`completed`, `failed`, `budget_exceeded`.

### 5.2 AsynchronousHarnessLoop — the engine

```python
from sutra import AsynchronousHarnessLoop
from sutra.core.budget import Budget, BudgetConfig
from sutra.models.mock_client import MockModelClient

harness = AsynchronousHarnessLoop(
    model_client=MockModelClient(script=lambda messages: {"text": "hello", "tool_calls": []}),
    tool_registry=ToolRegistry(),
    budget=Budget(BudgetConfig(max_usd=1.0, max_steps=10)),
    system_prompt="You are a helpful assistant.",
)

async for event in harness.run("Hi there"):
    ...  # each `event` is an SSEEvent
```

One call to `harness.run(user_input)` drives **one full user turn to completion**, which may
internally loop several times if the model keeps calling tools (bounded by
`max_tool_hops_per_turn`, default 8). It yields an `SSEEvent` for every meaningful thing that
happens along the way and stops cleanly on: a normal answer (`COMPLETED`), a guardrail block
(`FAILED`), a budget breach (`BUDGET_EXCEEDED`), a tool error (`FAILED`), or a permission gate
(`PAUSED_FOR_PERMISSION` — call `resume_after_permission(...)` later to continue).

### 5.3 Streaming (SSE)

```python
from sutra import EventType

async for event in harness.run("What's 2+2?"):
    if event.type == EventType.TOKEN:
        print(event.data["text"], end="")
    elif event.type == EventType.DONE:
        print("\nfinal status:", event.data["status"])
```

Every `SSEEvent` has `.to_sse()`, which renders the exact wire format an `EventSource` in a
browser expects:

```
event: token
data: {"id": "a1b2c3d4e5", "type": "token", "created_at": 1234.5, "text": "Hello", "agent": "root"}

```

Event types: `session_start`, `guardrail_block`, `token`, `message_complete`, `tool_call`,
`tool_result`, `permission_request`, `permission_resolved`, `handoff`, `budget_warning`,
`budget_exceeded`, `compaction`, `error`, `done`.

### 5.4 Context Compaction

```python
from sutra.core.compaction import ContextCompactor, CompactionConfig

compactor = ContextCompactor(
    CompactionConfig(token_threshold=6000, preserve_last_n_turns=6)
)
```

When the running token count (via `tiktoken` if `.[tokens]` is installed, else a `len//4`
heuristic) crosses `token_threshold`, everything except the last `preserve_last_n_turns` messages
is folded into one `Role.SUMMARY` message. Repeated compactions merge into the existing summary
rather than losing earlier context. You can plug in your own LLM-backed summarizer instead of the
default extractive one:

```python
async def llm_summarizer(messages: list[Message]) -> str:
    ...  # call a (cheap/fast) model to produce a real abstractive summary

compactor = ContextCompactor(summarizer=llm_summarizer)
```

### 5.5 Budget Monitor

```python
from sutra.core.budget import Budget, BudgetConfig, ModelPricing

budget = Budget(
    BudgetConfig(max_usd=5.0, max_input_tokens=200_000, max_output_tokens=50_000,
                 max_steps=40, warn_ratio=0.75),
    pricing=ModelPricing(input_per_million=3.0, output_per_million=15.0),
)
```

- `check_step()` is called *before* every model call — the step counter is a true hard cap.
- `record_usage()` is called with actual token counts *after* a model call — since USD/token cost
  is only knowable post-hoc, breaching a limit here still raises `BudgetExceededError`, and the
  harness refuses to issue the *next* call rather than pretending the already-spent call didn't
  happen.
- `warnings()` returns soft-threshold crossings (`warn_ratio`, default 75%) as `budget_warning`
  SSE events — useful for UI cost meters before a hard stop.

### 5.6 Guardrails

```python
from sutra.core.guardrails import Guardrail, GuardrailConfig

guardrail = Guardrail(GuardrailConfig(
    enable_default_injection_rules=True,   # ignore-previous-instructions, DAN, etc. -> BLOCK
    enable_default_pii_redaction=True,     # credit card / SSN / email -> REDACT
    blocklist_terms=["internal-only-codeword"],
))

# add your own:
guardrail.add_input_rule("no_shell_commands", r"rm\s+-rf|sudo\s+", action=GuardrailAction.BLOCK)
guardrail.add_output_rule("mask_internal_urls", r"https://internal\.[\w./]+", action=GuardrailAction.REDACT)
```

Input rules default to **BLOCK** (a detected injection should never reach the model at all —
raises `GuardrailViolation`, caught by the harness and turned into a `guardrail_block` event).
Output rules default to **REDACT** (don't kill a whole turn over PII that slipped through — scrub
it and continue).

### 5.7 Subagents & the Handoff Protocol

```python
from sutra.agents.subagent import SubagentProfile, SubagentRegistry

subagents = SubagentRegistry()
subagents.register(SubagentProfile(
    agent_id="finance_ops_agent",
    name="Finance Operations Agent",
    system_prompt="You may check balances and execute transfers, which require approval.",
    allowed_tools={"get_account_balance", "transfer_funds"},
))
```

A subagent can only ever be offered the tools in `allowed_tools` — `ToolRegistry.schemas_for()`
filters by agent id, so this is an actual capability boundary, not just a prompt suggestion.

Any agent hands off control by calling the reserved `HANDOFF_TOOL_NAME` (`"__handoff__"`) pseudo-tool
— you must register a schema for it (see `examples/finance_workflow.py`) so the model knows it
exists, but never implement its handler; the harness loop intercepts and dispatches it internally,
pushes the target agent onto `state.agent_stack`, and injects a system briefing message so the new
agent knows why control was transferred.

### 5.8 Custom Tools

```python
from sutra.tools.registry import Tool, ToolRegistry, ToolParameter
from sutra.core.permissions import PermissionLevel

registry = ToolRegistry()

# Option A: decorator style
@registry.register(
    "get_weather",
    "Get current weather for a city.",
    [ToolParameter("city", "string", "City name")],
    permission_level=PermissionLevel.LOW,
)
async def get_weather(city: str) -> dict:
    return {"city": city, "condition": "sunny"}

# Option B: explicit Tool object (useful when building the registry programmatically)
registry.register_tool(Tool(
    name="delete_record",
    description="Permanently delete a record. Irreversible.",
    parameters=[ToolParameter("record_id", "string", "ID to delete")],
    handler=delete_record_handler,
    permission_level=PermissionLevel.CRITICAL,
    allowed_agents=["admin_agent"],   # None = any agent may use it
))
```

Handlers **must** be `async def`. `ToolRegistry.execute()` validates required arguments, filters
kwargs down to what the handler actually accepts, and converts any exception raised inside the
handler into a `ToolExecutionError` — a misbehaving tool can never crash the harness process.
`Tool.to_schema()` produces an Anthropic/OpenAI-compatible function-calling schema automatically.

### 5.9 Permission Gates (human-in-the-loop)

```python
from sutra.core.permissions import PermissionGate, PermissionLevel

gate = PermissionGate(require_approval_from=PermissionLevel.HIGH)
```

Any tool call whose `permission_level` is `HIGH` or `CRITICAL` (given the default threshold) halts
the run: the harness sets `state.status = PAUSED_FOR_PERMISSION`, records a `PendingPermission` on
state, emits a `permission_request` SSE event, and the async generator returns — no
`asyncio.Future` is left dangling across process boundaries, because the paused state is entirely
captured in `HarnessState` and can be checkpointed.

An external actor — a human clicking "approve" in a UI, or your own automation — resumes the run:

```python
async for event in harness.resume_after_permission(request_id, approved=True, actor="alice@example.com"):
    ...
```

If `approved=False`, the tool never executes and the turn ends `FAILED` with a `tool` message
recording the denial.

### 5.10 The `ModelClient` boundary

This is the seam that makes the harness LLM-agnostic — see [section 6](#6-connecting-to-any-model)
for the full walkthrough.

---

## 6. Connecting to any model

Everything the harness knows about "a model" is this abstract interface
(`src/sutra/models/base.py`):

```python
class ModelClient(ABC):
    @abstractmethod
    def stream(
        self, *, system_prompt: str, messages: list[dict], tools: list[dict]
    ) -> AsyncIterator[StreamChunk]:
        """Yield StreamChunks as they arrive; the final chunk must set finished=True."""
```

`StreamChunk` carries `delta_text` (incremental text), `tool_call_delta` (provider-specific partial
or `{"finalized": [...]}` tool calls), `finished`, `finish_reason`, and `usage` (populated on the
final chunk). `ModelClient.complete()` is provided for free — it just drains `stream()` into one
`ModelResponse` for callers that don't need token-by-token output.

**`AsynchronousHarnessLoop` never imports `anthropic`, `openai`, or `httpx` itself.** It only ever
calls `self.model_client.stream(...)`. This is why swapping providers is a one-line change:

```python
# Anthropic (hosted)
from sutra.models.anthropic_client import AnthropicModelClient
model_client = AnthropicModelClient(model="claude-sonnet-5")               # pip install -e ".[anthropic]"

# OpenAI (hosted)
from sutra.models.openai_client import OpenAIModelClient
model_client = OpenAIModelClient(model="gpt-4o")                           # pip install -e ".[openai]"

# Ollama (local, any pulled model — see gpt-oss:20b walkthrough below)
from sutra.models.ollama_client import OllamaModelClient
model_client = OllamaModelClient(model="gpt-oss:20b")                      # pip install -e ".[ollama]"

# Deterministic, network-free (tests / demos / CI)
from sutra.models.mock_client import MockModelClient
model_client = MockModelClient(script=lambda messages: {"text": "...", "tool_calls": []})
```

`AsynchronousHarnessLoop(model_client=model_client, ...)` — nothing else in your agent code
changes. `examples/finance_workflow.py` and `examples/ollama_live_demo.py` are literally the same
harness wiring with only the `model_client=` line different.

### 6.1 Built-in clients at a glance

| Client | Wraps | Auth | Tool-call format normalized from |
|---|---|---|---|
| `AnthropicModelClient` | `anthropic.AsyncAnthropic` | `ANTHROPIC_API_KEY` env var or `api_key=` | `content_block_start`/`content_block_delta` events, `tool_use` blocks |
| `OpenAIModelClient` | `openai.AsyncOpenAI` Chat Completions | `OPENAI_API_KEY` env var or `api_key=` | streamed `delta.tool_calls[].function` chunks, reassembled by index |
| `OllamaModelClient` | Local Ollama `/api/chat` via `httpx` | none (local server) | OpenAI-style `message.tool_calls[].function` |
| `MockModelClient` | Nothing — pure Python | none | a `script(messages) -> {"text", "tool_calls"}` callable you provide |

Every client normalizes provider-specific failures into one `ModelInvocationError`, so the harness
loop's error handling is provider-agnostic too.

### 6.2 Adding a provider that isn't built in

To connect a model that has no client yet (Gemini, Mistral, Cohere, another local runtime like
LM Studio or vLLM's OpenAI-compatible server, etc.), implement `ModelClient.stream()`:

```python
from sutra.models.base import ModelClient, StreamChunk

class MyProviderClient(ModelClient):
    def __init__(self, model: str, **kwargs):
        self.model = model
        # set up your SDK/HTTP client here

    async def stream(self, *, system_prompt, messages, tools):
        # 1. translate `messages` (Sutra's [{"role", "content"}, ...] wire format)
        #    and `tools` (Anthropic-style {"name","description","input_schema"})
        #    into your provider's request shape
        # 2. issue the streaming request
        # 3. yield StreamChunk(delta_text=...) for each text fragment
        # 4. yield StreamChunk(tool_call_delta={"finalized": [...]}) once tool
        #    calls are fully assembled — each entry: {"id", "name", "input"}
        # 5. yield a final StreamChunk(finished=True, finish_reason=..., usage={...})
        ...
```

Many local runtimes (LM Studio, vLLM, llama.cpp server, Ollama itself) expose an
**OpenAI-compatible** `/v1/chat/completions` endpoint — for those, the fastest path is usually to
point `OpenAIModelClient` at a custom `base_url` rather than writing a new client, e.g.:

```python
import openai
client = OpenAIModelClient.__new__(OpenAIModelClient)
client._client = openai.AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
client.model, client.max_tokens = "your-local-model", 4096
```

(or simply fork `OpenAIModelClient.__init__` to accept a `base_url` parameter — it's ~20 lines).

---

## 7. Worked example: running gpt-oss:20b through Sutra

`gpt-oss:20b` is an open-weight 20B model you run locally through [Ollama](https://ollama.com).
This is the harness proving its own core claim: **the exact same orchestration code — guardrails,
budget, permission gate, tool dispatch — works identically whether the model is a hosted frontier
model or a local open-weight model**, because none of that code ever talks to the model directly.

`examples/ollama_live_demo.py` is a ready-to-run demo of this. Here's the full walkthrough.

### 7.1 Prerequisites

```bash
# 1. Install Ollama: https://ollama.com/download
# 2. Start the server (or use the desktop app, which runs it for you)
ollama serve

# 3. In another terminal, pull the model (~13 GB download)
ollama pull gpt-oss:20b

# 4. Install the httpx dependency the OllamaModelClient needs
pip install -e ".[ollama]"
```

### 7.2 What the demo wires up

```python
from sutra.core.budget import Budget, BudgetConfig
from sutra.core.guardrails import Guardrail
from sutra.core.harness import AsynchronousHarnessLoop
from sutra.core.permissions import PermissionGate, PermissionLevel
from sutra.models.ollama_client import OllamaModelClient
from sutra.tools.registry import Tool, ToolParameter, ToolRegistry

MODEL_NAME = "gpt-oss:20b"

# Three real async tools: two low-risk (freely callable), one CRITICAL (gated)
registry = ToolRegistry()
registry.register_tool(Tool(
    name="get_current_weather",
    description="Get the current weather conditions for a location.",
    parameters=[ToolParameter("location", "string", "City name, e.g. 'Tokyo'.")],
    handler=get_current_weather,
    permission_level=PermissionLevel.LOW,
))
registry.register_tool(Tool(
    name="roll_dice",
    description="Roll an N-sided die and return the result.",
    parameters=[ToolParameter("sides", "integer", "Number of sides.", required=False)],
    handler=roll_dice,
    permission_level=PermissionLevel.LOW,
))
registry.register_tool(Tool(
    name="send_email",
    description="Send an email to a recipient. Sends real, externally-visible email.",
    parameters=[
        ToolParameter("to", "string", "Recipient email address."),
        ToolParameter("subject", "string", "Email subject line."),
        ToolParameter("body", "string", "Email body text."),
    ],
    handler=send_email,
    permission_level=PermissionLevel.CRITICAL,   # <- will be gated
))

harness = AsynchronousHarnessLoop(
    model_client=OllamaModelClient(model=MODEL_NAME),
    tool_registry=registry,
    budget=Budget(BudgetConfig(max_usd=1.0, max_input_tokens=50_000, max_output_tokens=8_000, max_steps=12)),
    guardrail=Guardrail(),
    permission_gate=PermissionGate(require_approval_from=PermissionLevel.HIGH),
    system_prompt=(
        "You are a helpful assistant with access to tools. Use get_current_weather and "
        "roll_dice freely. send_email is high-risk and requires human approval, but you "
        "should still call it when the user asks you to send an email."
    ),
)
```

Note the budget here is USD-based even though the model is free to run locally — `Budget` doesn't
know or care whether tokens are billed; it's still useful as a runaway-loop safety net (`max_steps`)
and a place to plug real pricing back in the moment you swap to a hosted model.

### 7.3 Driving one turn and handling the permission pause

```python
prompt = (
    "Check the current weather in Tokyo, then send an email to ops@example.com "
    "with subject 'Weather Update' summarizing the conditions you found."
)

request_id = None
async for event in harness.run(prompt):
    if event.type == EventType.TOKEN:
        print(event.data["text"], end="", flush=True)
    else:
        print(f"\n[{event.type.value}] {event.data}")
    if event.type == EventType.PERMISSION_REQUEST:
        request_id = event.data["request_id"]

print("status:", harness.state.status.value)   # -> "paused_for_permission"

if request_id:
    async for event in harness.resume_after_permission(request_id, approved=True, actor="demo-operator"):
        print(event.data)

print("final status:", harness.state.status.value)   # -> "completed"
```

Run it:

```bash
python examples/ollama_live_demo.py
```

Expected shape of the run (the exact model wording will vary — this is a *real* model making
*real* decisions, unlike the scripted `MockModelClient` in `finance_workflow.py`):

1. `session_start`
2. streamed `token` events as `gpt-oss:20b` reasons about the request
3. `tool_call` / `tool_result` for `get_current_weather("Tokyo")`
4. `tool_call` for `send_email(...)`, but because it's `CRITICAL` and the gate's threshold is
   `HIGH`, dispatch is intercepted → `permission_request` event, `state.status` becomes
   `paused_for_permission`, and `harness.run()` stops there
5. after `resume_after_permission(..., approved=True, ...)`: `permission_resolved`, `tool_result`
   for the now-executed `send_email` call, then the model's closing message and `done`

### 7.4 Swapping in a different local model

Nothing above is specific to `gpt-oss:20b` — it's just the string passed to `OllamaModelClient`:

```python
OllamaModelClient(model="llama3.1")       # default if you omit `model`
OllamaModelClient(model="qwen2.5:14b")
OllamaModelClient(model="gpt-oss:20b", base_url="http://192.168.1.50:11434")  # remote Ollama host
```

Whichever model you point at needs to support Ollama's `tools` field in `/api/chat` if you want it
to call tools reliably — `gpt-oss:20b`, `llama3.1`, and `qwen2.5` all do.

---

## 8. Serving over HTTP (FastAPI + SSE)

```bash
pip install -e ".[server]"
```

`src/sutra/server/app.py` exposes a small FastAPI app:

| Endpoint | Method | Purpose |
|---|---|---|
| `/chat` | `POST {session_id, message}` | Streams the turn back as `text/event-stream` |
| `/permissions/{request_id}/resolve?session_id=...` | `POST {approved, actor}` | Resumes a paused turn |
| `/state/{session_id}` | `GET` | Returns the session's serialized `HarnessState` |

You register a harness per session from your own bootstrap code (auth, session creation, whatever
your app needs) — the server module deliberately doesn't own that:

```python
from sutra.server.app import app, register_harness
from sutra.core.harness import AsynchronousHarnessLoop
# ... build a harness however section 9 describes ...
register_harness("session-123", harness)
```

```bash
uvicorn sutra.server.app:app --reload
```

A browser (or any SSE client) then does:

```js
const es = new EventSource(); // in practice you POST /chat and read the streamed response body
fetch("/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ session_id: "session-123", message: "Hello" }),
}).then(res => res.body.getReader() /* read text/event-stream frames */);
```

When a `permission_request` event arrives client-side, render an approve/deny UI, then:

```bash
curl -X POST "http://localhost:8000/permissions/<request_id>/resolve?session_id=session-123" \
     -H "Content-Type: application/json" \
     -d '{"approved": true, "actor": "alice@example.com"}'
```

---

## 9. Build your own agent, step by step

This mirrors what `examples/finance_workflow.py` and `examples/ollama_live_demo.py` do, distilled
into a template you can copy.

```python
import asyncio
from sutra.core.budget import Budget, BudgetConfig
from sutra.core.compaction import ContextCompactor, CompactionConfig
from sutra.core.guardrails import Guardrail, GuardrailConfig
from sutra.core.harness import AsynchronousHarnessLoop
from sutra.core.permissions import PermissionGate, PermissionLevel
from sutra.tools.registry import Tool, ToolParameter, ToolRegistry
from sutra.agents.subagent import SubagentProfile, SubagentRegistry
from sutra.models.ollama_client import OllamaModelClient   # or Anthropic/OpenAI/Mock
from sutra.streaming.sse import EventType

# 1. Define your tools as plain async functions.
async def search_docs(query: str) -> dict:
    return {"query": query, "results": ["..."]}

async def delete_index(index_name: str) -> dict:
    return {"status": "deleted", "index_name": index_name}

# 2. Register them with a risk level. LOW/MEDIUM run freely; HIGH/CRITICAL
#    are gated by the PermissionGate's threshold.
tools = ToolRegistry()
tools.register_tool(Tool(
    name="search_docs", description="Search the internal knowledge base.",
    parameters=[ToolParameter("query", "string", "Search text.")],
    handler=search_docs, permission_level=PermissionLevel.LOW,
))
tools.register_tool(Tool(
    name="delete_index", description="Permanently delete a search index.",
    parameters=[ToolParameter("index_name", "string", "Index to delete.")],
    handler=delete_index, permission_level=PermissionLevel.CRITICAL,
))

# 3. (Optional) Define specialized subagents with restricted tool access.
subagents = SubagentRegistry()
subagents.register(SubagentProfile(
    agent_id="admin_agent", name="Admin Agent",
    system_prompt="You handle destructive administrative actions. Confirm intent before acting.",
    allowed_tools={"delete_index"},
))

# 4. Assemble the harness.
harness = AsynchronousHarnessLoop(
    model_client=OllamaModelClient(model="gpt-oss:20b"),
    tool_registry=tools,
    budget=Budget(BudgetConfig(max_usd=2.0, max_steps=20)),
    guardrail=Guardrail(GuardrailConfig(blocklist_terms=["drop table"])),
    compactor=ContextCompactor(CompactionConfig(token_threshold=6000, preserve_last_n_turns=6)),
    subagent_registry=subagents,
    permission_gate=PermissionGate(require_approval_from=PermissionLevel.HIGH),
    system_prompt="You are a helpful knowledge-base assistant.",
)

# 5. Drive turns, handling the permission-pause branch.
async def main():
    request_id = None
    async for event in harness.run("Search for 'onboarding checklist'"):
        if event.type == EventType.PERMISSION_REQUEST:
            request_id = event.data["request_id"]
        print(event.type.value, event.data)

    if request_id:
        async for event in harness.resume_after_permission(request_id, approved=True, actor="you"):
            print(event.type.value, event.data)

asyncio.run(main())
```

Checklist for a production agent built this way:

- [ ] Every tool that mutates state or costs money is `HIGH` or `CRITICAL`.
- [ ] `PermissionGate(require_approval_from=...)` matches your actual risk tolerance.
- [ ] `Budget` limits are set to something you'd be comfortable losing unattended.
- [ ] `Guardrail` blocklist covers domain-specific injection phrases beyond the defaults.
- [ ] If a subagent exists, its `allowed_tools` is the *minimum* set it needs — not "everything."
- [ ] `HarnessState.checkpoint()` is persisted somewhere durable if a permission pause might
      outlive your process (e.g. waiting hours for a human).

---

## 10. Testing your agent

Sutra ships `MockModelClient` specifically so agent logic (tool wiring, guardrails, budgets,
handoffs, permission flows) can be tested deterministically, with zero network calls and zero API
cost:

```python
from sutra.models.mock_client import MockModelClient

def script(messages):
    # Inspect the wire-format conversation so far and decide the next turn.
    if len(messages) == 1:
        return {"text": "Let me check that.", "tool_calls": [
            {"id": "call_1", "name": "search_docs", "input": {"query": "onboarding"}}
        ]}
    return {"text": "Here's what I found.", "tool_calls": []}

model_client = MockModelClient(script=script)
```

Run the existing suite to see this pattern applied to every component:

```bash
pytest -v
```

- `tests/test_harness.py` — full-loop behavior (streaming, tool dispatch, handoff, permission pause)
- `tests/test_budget.py` — hard/soft threshold behavior
- `tests/test_guardrails.py` — block vs. redact rules
- `tests/test_compaction.py` — summarization and window preservation

Swap `MockModelClient` for a real client only in an explicit "live" test/example (like
`ollama_live_demo.py`) that you run manually or gate behind an integration-test marker — don't let
CI depend on a live model server or paid API credentials.

---

## 11. Design principles worth stealing

If you're building your *own* agent harness rather than using Sutra directly, these are the ideas
worth keeping regardless of implementation:

- **One state object, always serializable.** If your agent can't be checkpointed mid-run, it can't
  survive a crash, a deploy, or a multi-hour human approval wait.
- **Interception, not observation.** Guardrails and permission gates must run *before* the risky
  action, not log it after the fact.
- **The model proposes, the harness disposes.** The LLM decides *what* it wants to do; deterministic
  code (budget checks, permission levels, tool whitelists) decides what's actually *allowed* to
  happen.
- **One typed interface between orchestration and provider.** If your core loop imports a vendor
  SDK, you don't have an agent harness — you have a wrapper around one vendor.
- **Stream everything as structured events, not raw text.** A frontend, an audit log, and a test
  assertion all want the same thing: a typed record of what happened, in order.
- **Specialize instead of generalizing one giant prompt.** Narrow system prompts with a narrow
  tool whitelist are safer and more reliable than one agent that "can do everything."

---

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: The 'httpx' package is required for OllamaModelClient` | `.[ollama]` extra not installed | `pip install -e ".[ollama]"` |
| `ModelInvocationError: Ollama stream failed: ...Connection refused` | Ollama server not running | `ollama serve`, or open the desktop app |
| Model never calls tools even though tools are registered | Model doesn't support Ollama's `tools` field, or the prompt doesn't ask for tool use | Confirm the model supports function-calling (`gpt-oss:20b`, `llama3.1`, `qwen2.5` do); make the system prompt explicit about when to use each tool |
| Run stops abruptly with a `guardrail_block` event | Input matched a default injection pattern (e.g. "ignore previous instructions") or your `blocklist_terms` | Expected behavior — inspect `event.data["rule"]`; relax `GuardrailConfig` only if the block is a false positive |
| Run ends `budget_exceeded` sooner than expected | `BudgetConfig` limits are tight relative to model verbosity | Raise `max_output_tokens` / `max_usd`, or reduce `max_tool_hops_per_turn` |
| `harness.run()` returns after one `permission_request` and nothing else happens | This is correct — the generator halts on `PAUSED_FOR_PERMISSION` by design | Call `harness.resume_after_permission(request_id, approved=..., actor=...)` |
| `KeyError: No pending permission request with id ...` | Calling `resolve()`/`resume_after_permission()` twice, or after a process restart without restoring state | Persist `request_id` alongside `HarnessState.checkpoint()`; only resolve once |
| Context keeps growing / high token costs | `ContextCompactor` threshold too high, or not passed into the harness | Lower `CompactionConfig.token_threshold`; confirm `compactor=` is passed to `AsynchronousHarnessLoop` |

---

## 13. Reference cheat-sheet

```python
from sutra import (
    AsynchronousHarnessLoop, HarnessState, HarnessStatus, Message,
    Budget, BudgetConfig, ContextCompactor,
    Guardrail, GuardrailConfig,
    HandoffRouter, HandoffRequest,
    PermissionGate, PermissionLevel, PermissionRequest,
    ToolRegistry, Tool,
    SubagentProfile, SubagentRegistry,
    SSEEvent, EventType, format_sse,
)
```

| I want to... | Use |
|---|---|
| Run one user turn | `async for event in harness.run(text): ...` |
| Resume after a human approves/denies a tool | `async for event in harness.resume_after_permission(request_id, approved=bool, actor=str): ...` |
| Persist a paused/long-running session | `state.checkpoint()` / `HarnessState.from_json(blob)` |
| Register a tool | `registry.register_tool(Tool(...))` or `@registry.register(name, desc, params)` |
| Gate a tool behind human approval | Give it `permission_level=PermissionLevel.HIGH` or `.CRITICAL` |
| Create a specialist agent | `SubagentProfile(agent_id=..., allowed_tools={...})` + register on `SubagentRegistry` |
| Hand off to a specialist | Model calls the `__handoff__` tool (`HANDOFF_TOOL_NAME`) with `target_agent_id` |
| Block a prompt-injection pattern | `guardrail.add_input_rule(name, regex, GuardrailAction.BLOCK)` |
| Redact PII from output | `guardrail.add_output_rule(name, regex, GuardrailAction.REDACT)` |
| Cap spend | `Budget(BudgetConfig(max_usd=..., max_steps=...))` |
| Switch model providers | Change only the `model_client=` argument |
| Serve over HTTP | `pip install -e ".[server]"`, `register_harness(...)`, `uvicorn sutra.server.app:app` |

---

**Where to go next:**
- Read `examples/finance_workflow.py` for a scripted, deterministic tour of every safety mechanism.
- Read `examples/ollama_live_demo.py` and run it against `gpt-oss:20b` for a live, non-scripted tour.
- Read `tests/` for focused examples of each component's edge-case behavior.
- Read `README.md` at the repo root for the condensed architecture summary.
