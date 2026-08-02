# Sutra vs. the field: a comparison and a backlog

This document does two things: (1) audits the existing `specs/` folder against the actual code for accuracy, and (2) compares Sutra's architecture against four other agent harnesses/SDKs — **Claude Code**, the **OpenAI Agents SDK**, **OpenClaw**, and **Hermes Agent** (Nous Research) — to propose a prioritized backlog of new specs.

## Methodology and honesty about sources

- **Sutra** claims are verified directly against `src/sutra/` — every discrepancy below was confirmed by reading the actual code, not just the spec text.
- **Claude Code** claims come from direct operating knowledge — this document was written by an instance of Claude Code, running with this repo as its working directory, using its own tool list and system context as the primary source.
- **OpenAI Agents SDK** claims come from training knowledge of the `openai-agents` Python SDK's public API (`Agent`, `Runner`, handoffs, guardrails, sessions, tracing).
- **OpenClaw** and **Hermes Agent** claims come from a live web search plus a fetched comparison article ([mcplato.com/en/blog/ai-agent-harness-comparison-2026](https://mcplato.com/en/blog/ai-agent-harness-comparison-2026/), and the Hermes Function Calling reference at [github.com/NousResearch/Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling)), since I don't have reliable built-in knowledge of either. That source itself flags several OpenClaw/Hermes capabilities as "documented, not independently verified" — I've preserved that uncertainty below rather than presenting it as settled fact. Where a claim is soft, it's marked **[unverified]**.

---

## Part 1 — Specs folder accuracy audit

You asked me to review `specs/`. Overall it's a strong, well-organized set of 15 documents (`00-overview.md` through `14-harness-state.md`) — each has a consistent Purpose / Current Behavior / Enhancement Ideas structure, and most of it matches the code exactly. But a few files have drifted from what's actually implemented, likely written from an earlier design pass or without a final cross-check against the code:

| File | Claim | Actual code | Impact |
|---|---|---|---|
| `05-subagents.md` | `it_ops_agent` tools: `check_system_status`, `get_incident_details`, `escalate_incident`, `restart_service` | Actual (`toolkits/contoso_demo.py`): `check_ticket_status`, `search_knowledge_base`, `restart_service`, `escalate_to_oncall` | Someone following this spec would look for the wrong tool names |
| `05-subagents.md` | `finance_ops_agent` tools: `check_account_balance`, `get_transaction_history`, `validate_transfer`, `transfer_funds` | Actual: `get_account_balance`, `get_transaction_history`, `send_customer_notification`, `transfer_funds` | Same — `validate_transfer` doesn't exist; `send_customer_notification` isn't mentioned |
| `10-memory-system.md` | "Parse **YAML** frontmatter" | `MemoryStore`/`MemoryDocument` use a hand-rolled flat `key: value` parser — deliberately, to avoid a PyYAML dependency (see the module docstring in `memory/store.py`) | Minor, but "YAML" implies nested structures the format doesn't actually support |
| `10-memory-system.md` | `write_index()` refreshes `.memory/index.md` | Actual: `.memory/MEMORY.md` (`MemoryStore.index_path()`) | Someone would look in the wrong file |
| `11-model-client.md` | `ModelResponse` fields: `text`, `tool_calls`, `input_tokens`, `output_tokens` | Actual field is `content`, not `text`; there's also a `stop_reason` field not mentioned | `response.text` would raise `AttributeError` |
| `11-model-client.md` | `StreamChunk.delta_text: Optional[str] = None` | Actual default is `""`, not `None` | Minor, but affects falsy-check logic if someone codes against the spec |
| `11-model-client.md` | `StreamChunk` fields listed don't include `finish_reason` | Actual dataclass has `finish_reason: Optional[str] = None` | Missing field entirely |
| `11-model-client.md` | `MockModelClient(responses=[...])` — list of `ModelResponse` objects | Actual constructor: `MockModelClient(script: ScriptFn)` — a callable, not a list | **This example would crash with `TypeError` if run as written** |
| `11-model-client.md` | Example model string `claude-sonnet-4-6` | Not an error exactly, but doesn't match any model actually used in this codebase's examples/tests (which use `gpt-oss:20b` via Ollama, or `claude-sonnet-5` in the CLI's own default patterns elsewhere in this session) | Cosmetic |
| `13-cli.md` | "`sutra demo` — fully automated, narrated **three**-act tour" | Now **four** acts (Guardrails, Context Compaction, IT Incident, Finance Transfer) as of the most recent CLI change | Spec predates the compaction act |
| `13-cli.md` | Only documents `/demo all` | `/demo <act>` (e.g. `/demo compaction`), `sutra demo <act>`, and act aliases (`it`, `transfer`, `context`, etc.) now exist | Spec predates the per-act demo feature |

**Recommendation**: these are cheap, mechanical fixes (find-and-replace tool names, fix the two field names, fix one filename, update the act count). I'd suggest doing this as a follow-up pass before adding any of the new specs below, so the backlog builds on ground truth. I did not fix these files myself since you asked for a comparison doc, not a spec edit — say the word and I'll patch them.

---

## Part 2 — Capability matrix

| Capability | Sutra | Claude Code | OpenAI Agents SDK | OpenClaw | Hermes Agent |
|---|---|---|---|---|---|
| Core agent loop | ✅ `AsynchronousHarnessLoop` | ✅ | ✅ `Runner` | ✅ Gateway | ✅ |
| Tool/function registry | ✅ `ToolRegistry`, typed params | ✅ built-in + MCP | ✅ `function_tool` (auto schema from type hints) | ✅ shell/browser/canvas + skills | ✅ shell/browser/files/APIs + MCP |
| Subagents / multi-agent | ✅ restricted tool subsets, one-way handoff | ✅ rich (background, isolated worktrees, custom `.claude/agents/*.md`) | ✅ handoffs + **agent-as-tool** (retains control) | ✅ channel→agent routing **[unverified depth]** | ✅ **[unverified depth]** |
| Context compaction | ✅ token-threshold + preserve-last-N + pluggable summarizer | ✅ auto-compact (undocumented internals) | ➖ delegated to Sessions, no built-in compaction | ➖ not disclosed | ➖ not disclosed |
| Budget/cost control | ✅ hard USD/token/step caps, soft warnings | ➖ not exposed to the agent itself | ➖ `max_turns` only, no USD tracking | ➖ not disclosed | ➖ not disclosed |
| Guardrails | ✅ regex block/redact, input+output | ➖ (relies on permission system instead) | ✅ typed input/output guardrails, tripwire, run in parallel | ➖ not disclosed as a distinct layer | ➖ not disclosed |
| Permission/approval gates | ✅ `PermissionGate`, async Future, SSE-driven resume | ✅ per-tool permission modes, plan mode, allow/deny lists | ➖ no built-in HITL primitive (roll your own) | ✅ dangerous-command approvals, allowlists **[unverified]** | ✅ dangerous-command approvals **[unverified]** |
| Persistent memory | ✅ instance/user/session scopes, markdown files, injection-safe persona | ✅ `CLAUDE.md` + auto-memory (typed: user/feedback/project/reference) | ✅ `Session` (SQLite or pluggable) — raw conversation continuity, not fact extraction | ✅ "configurable memory patterns" **[unverified]** | ✅ session search, self-authored memory **[unverified]** |
| Cross-session memory search | ❌ | ✅ (memory files are searchable text) | ➖ | ➖ not disclosed | ✅ "search prior sessions" **[unverified]** |
| Hooks / lifecycle interception | ❌ | ✅ PreToolUse/PostToolUse/Stop/etc., shell-command based | ✅ `on_tool_start`/`on_tool_end`/`on_handoff` lifecycle callbacks | ➖ | ➖ |
| Structured task/todo tracking | ➖ internal `ExecutionStep` log only, not model-facing | ✅ `TaskCreate`/`TaskUpdate` with dependency graph, model-visible | ➖ | ➖ | ➖ |
| Background/async tool execution | ❌ tool calls block the loop | ✅ `run_in_background`, notification on completion | ➖ | ➖ | ➖ |
| Tool execution sandboxing | ❌ tools run in-process, full trust | ✅ optional Bash sandbox, isolated cloud VMs | ➖ (caller's responsibility) | ✅ opt-in Docker/SSH/OpenShell, dropped capabilities **[unverified]** | ✅ container/remote backends, "environment filtering" **[unverified]** |
| Plan mode (read-only-first) | ❌ | ✅ `EnterPlanMode`/`ExitPlanMode` | ➖ | ➖ | ➖ |
| Structured output enforcement | ❌ free-text only | ➖ (not applicable — Claude Code's own output is prose) | ✅ `output_type` as Pydantic model | ➖ | ➖ |
| Dynamic/contextual instructions | ❌ static string per profile | ✅ (memory context block is dynamic, but system prompts are static text) | ✅ instructions can be a function of run context | ➖ | ➖ |
| Tracing / observability spans | ➖ SSE events (live only, not queryable post-hoc) | ✅ session history, transcripts | ✅ built-in tracing, exportable | ➖ logging mentioned, no spans | ✅ tool-call recording **[unverified]** |
| Multi-channel ingress | ❌ CLI + HTTP only | ➖ (CLI/IDE/desktop/web, not chat platforms) | ➖ | ✅ 50+ integrations (WhatsApp, Slack, Discord, Telegram, etc.) — this is OpenClaw's headline feature | ➖ not primary focus |
| Proactive/scheduled triggers | ❌ purely reactive to `run(user_input)` | ✅ `/loop`, `ScheduleWakeup`, cron-backed scheduled agents | ➖ | ✅ implied by "24/7 autonomous" positioning **[unverified]** | ➖ |
| Self-authored tools/skills | ❌ | ➖ (Skills are curated/packaged, not self-written by the agent) | ➖ | ✅ headline feature — "autonomously writing code to create relevant new skills" | ✅ "create or improve skills" **[unverified]** |
| Model-provider agnosticism | ✅ `ModelClient` ABC, 4 implementations | N/A (Anthropic-only) | ✅ + LiteLLM (100+ providers) | ✅ "pick the models" | ✅ multi-model |
| Tool-call wire format flexibility | ➖ OpenAI/Ollama flat schema only; Anthropic needs its own adapter (documented gap in `harness.py`) | N/A | ✅ OpenAI-native | ➖ | Model-level: Hermes fine-tunes use a distinct `<tool_call>` XML-tag format many open-weight models expect |

Legend: ✅ present and solid · ➖ partial, indirect, or not disclosed · ❌ absent · **[unverified]** = sourced from public docs I couldn't independently confirm

---

## Part 3 — What each system does that Sutra doesn't, in one paragraph each

**Claude Code** generalizes *interception*. Where Sutra has exactly two intervention points (Guardrail for input/output text, PermissionGate for risky tool calls), Claude Code has arbitrary shell-command **hooks** at every lifecycle event, a **Plan Mode** that restricts the whole turn to read-only tools until a plan is approved, model-visible **task tracking** with dependencies, and **background execution** for anything that shouldn't block the loop. These aren't safety features so much as *extensibility* features — they let an operator bolt on custom policy without touching harness code.

**OpenAI Agents SDK** generalizes *orchestration*. Sutra's Handoff is one-way and permanent (the calling agent loses control). The SDK's **agent-as-tool** pattern lets an orchestrator call a specialist and get a result back without losing its own turn — useful for fan-out/consult patterns Sutra's handoff can't express. It also has typed **structured outputs** (an agent's final answer conforms to a schema, not just tool arguments) and a first-class **Sessions** abstraction that removes the boilerplate of manually holding a `HarnessState` across calls.

**OpenClaw** generalizes *reach and autonomy*. Its entire value proposition is meeting users where they already are (50+ chat/messaging integrations) and running unattended (proactive automation, not just reactive prompt→response). Sutra is currently CLI+HTTP only and 100% reactive — there's no way to trigger a turn from a schedule or a webhook without writing that plumbing yourself. OpenClaw's other headline feature — agents that write their own new tools at runtime — is powerful but exactly the kind of thing Sutra's whole design philosophy (explicit guardrails, explicit permission gates) should treat with real suspicion; see the caveats on spec 27 below.

**Hermes Agent** generalizes *memory depth*, and separately, the underlying **Hermes model family** has established its own tool-calling wire format (`<tools>`/`<tool_call>`/`<tool_response>` XML tags) that a meaningful slice of open-weight models are specifically fine-tuned against — a concrete, low-risk compatibility gap for Sutra's `ModelClient` layer, independent of how good the Hermes Agent *product* turns out to be.

---

## Part 4 — Proposed new specs

Numbered continuing from the existing `specs/` folder (`00`–`14`). Each entry: what it is, which system(s) it's inspired by, why it matters for Sutra specifically, a rough shape, and a priority.

### Group A — Safety & isolation

#### 15. Hooks: lifecycle event interception
*Inspired by: Claude Code (PreToolUse/PostToolUse/Stop hooks)*

Right now the only ways to intervene in a run are `Guardrail` (regex on text) and `PermissionGate` (block/allow a whole tool call). Neither lets an operator run arbitrary custom logic — log every tool call to a SIEM, veto a tool call based on business rules too complex for regex, inject a compliance banner before every response. Add a `HookRegistry` with named lifecycle points (`pre_tool_call`, `post_tool_call`, `pre_model_call`, `on_handoff`, `on_done`) that runs registered async callables (or shell commands, for parity with Claude Code's model) and can veto/modify the in-flight action.

```python
hooks = HookRegistry()
hooks.register("pre_tool_call", audit_log_hook)
hooks.register("pre_tool_call", custom_business_rule_veto)
harness = AsynchronousHarnessLoop(..., hooks=hooks)
```

**Priority: High** — this is the single biggest generalization gap versus Claude Code, and it's additive (doesn't change existing behavior for harnesses that don't register hooks).

#### 16. Plan Mode
*Inspired by: Claude Code (`EnterPlanMode`/`ExitPlanMode`)*

A turn-level mode where only read-only tools (`allowed_agents`/`permission_level` metadata already half-supports this distinction) are visible until the model produces an explicit plan and a human approves it — then mutating tools unlock for the rest of the turn. Different from `PermissionGate`, which gates individual calls; this gates an entire *phase* of the turn. Natural fit for `HarnessStatus` (`PLANNING` → `PLAN_APPROVED` → normal execution).

**Priority: Medium** — valuable for high-stakes domains (the finance/IT toolkits are exactly the kind of place a "show me the plan first" mode would matter), but `PermissionGate` already covers the worst-case (nothing CRITICAL executes without approval anyway).

#### 17. Tool execution sandboxing / pluggable backends
*Inspired by: Claude Code (Bash sandbox, isolated VMs), OpenClaw (Docker/SSH/OpenShell backends), Hermes (environment filtering)*

Every tool handler in Sutra currently runs in-process with the full trust and environment of the harness. Add an `ExecutionBackend` abstraction (`LocalBackend`, `DockerBackend`, `RestrictedShellBackend`) that a `Tool` can opt into, plus per-tool environment/credential scoping so a tool that shells out doesn't inherit every API key the process holds. This matters more as spec 27 (self-authored tools, see below) becomes even a remote possibility — tools nobody has manually reviewed absolutely should not run with ambient trust.

**Priority: Medium** — high value, but real engineering cost (needs a real sandboxing backend, not just an interface).

### Group B — Memory & session continuity

#### 18. Sessions API — automatic cross-call continuity
*Inspired by: OpenAI Agents SDK (`Session`, `SQLiteSession`)*

Today, a caller must manually hold onto a `HarnessState`/`AsynchronousHarnessLoop` instance across multiple `run()` calls to keep a conversation going (this is exactly what `sutra chat`'s REPL loop does by hand). Formalize this as a `SessionStore` protocol (`load(session_id) -> HarnessState`, `save(session_id, state)`) with a default file/SQLite-backed implementation, so `sutra chat --resume <id>` or a stateless HTTP deployment can pick up a conversation without the caller managing the object lifetime themselves. Complements — doesn't replace — the existing user/persona memory system, which is about *facts*, not raw conversation continuity.

**Priority: High** — directly enables `13-cli.md`'s own "Enhancement Ideas" item ("`sutra chat --resume session-abc`"), and is a clean, self-contained addition.

#### 19. Cross-session memory search
*Inspired by: Hermes Agent ("search prior sessions") — [unverified depth, but the primitive is cheap to build regardless]*

`MemoryStore.list_session_ids()` exists but there's no way to *search* — "when did we last discuss the Falcon project?" currently requires manually opening session files. Add `MemoryStore.search_sessions(query: str) -> List[SessionMatch]`, starting with simple substring/keyword matching over `session.md` bodies (no new dependency needed), with room to grow into embedding-based retrieval later (ties into `10-memory-system.md`'s own "vector-based retrieval" enhancement idea).

**Priority: Low-Medium** — nice-to-have, cheap first version, but not blocking anything else.

### Group C — Orchestration patterns

#### 20. Agent-as-tool (consult without handoff)
*Inspired by: OpenAI Agents SDK (`agent.as_tool()`)*

Sutra's `Handoff` is one-way and permanent — the moment the root agent hands off to `finance_ops_agent`, root's frame is gone until (if ever) something pops the stack. There's no way for root to *consult* a specialist and keep going. Add a pseudo-tool (`__consult__`, mirroring `__handoff__`'s interception pattern in `_dispatch_tool_call`) that runs a subagent's profile as a nested, bounded sub-run and returns its final text as a tool result — the calling agent's turn continues uninterrupted. This is the cleanest, most additive spec in this whole list: it reuses `SubagentRegistry`, `SubagentProfile`, and the existing tool-dispatch interception pattern almost unchanged.

**Priority: High** — high value, low implementation risk, directly extends an existing well-tested pattern (handoff interception) rather than introducing a new one.

#### 21. Structured output enforcement
*Inspired by: OpenAI Agents SDK (`output_type`)*

Add an optional `output_schema: Optional[Type[BaseModel]]` to `SubagentProfile` (or a new harness-level param) so a subagent's *final* text response (not just tool call arguments, which already have schemas via `ToolParameter`) is validated/parsed against a Pydantic model before `MESSAGE_COMPLETE` fires. Useful for a triage-style root agent whose whole job is to emit `{category, priority, target_agent}` for something else to consume programmatically.

**Priority: Medium** — valuable for programmatic/pipeline use of Sutra (vs. the chat-first use cases the current toolkits demonstrate), less valuable for the CLI/chat experience.

#### 22. Dynamic/contextual system prompts
*Inspired by: OpenAI Agents SDK (instructions as a function of context)*

`SubagentProfile.system_prompt` and `AsynchronousHarnessLoop`'s root `system_prompt` are static strings today. Allow either to be `Union[str, Callable[[HarnessState], str]]` so prompts can incorporate runtime facts (current date, caller's role, active budget remaining) without routing everything through the memory system. Small, mechanical change to `_effective_system_prompt()`.

**Priority: Low** — genuinely useful but the memory context block already covers the most common case (user identity); this closes a narrower gap.

### Group D — Observability

#### 23. Model-visible structured task tracking
*Inspired by: Claude Code (`TaskCreate`/`TaskUpdate`/`TaskList`)*

`HarnessState.execution_steps` is a good *internal* audit log but it's not something the model can read or write to plan its own multi-step work, and it has no dependency graph. For genuinely long multi-hop tasks (the kind `max_tool_hops_per_turn` is a safety valve against), give the model a `manage_tasks` tool backed by a `TaskList` on `HarnessState`, with `blocks`/`blockedBy` relationships, so both the model and any UI rendering the session can see real progress on a complex job instead of just a stream of tool calls.

**Priority: Medium** — genuinely improves UX for complex tasks, but the ROI is lower for Sutra's current toolkit-scale demos (2-5 hops) than it would be for a harness running much longer agentic sessions.

#### 24. Background/async tool execution
*Inspired by: Claude Code (`run_in_background`, notification-on-completion)*

Every tool call blocks the harness loop until it returns. For a tool that takes minutes (a deployment, a large data export), that's the whole run stalled. Add `Tool.supports_background: bool` and a harness-level pattern where a long tool call returns immediately with a handle, the loop continues (or pauses gracefully), and a `TOOL_COMPLETED_ASYNC` SSE event fires when the background operation finishes — requires a background task registry analogous to `PermissionGate`'s pending-request dict.

**Priority: Low-Medium** — real engineering lift (needs its own task-tracking infrastructure), valuable mainly once Sutra grows tools that are genuinely slow.

### Group E — Reach & autonomy

#### 25. Multi-channel ingress
*Inspired by: OpenClaw (headline feature — 50+ integrations)*

Sutra currently has two ways in: the CLI and the HTTP server. Add a thin adapter layer (`ChannelAdapter` protocol: receive external message → call `harness.run()` → route SSE events back to the channel) with a first implementation or two (Slack, or a generic webhook receiver) to prove the pattern. This is explicitly the single biggest reason OpenClaw grew as fast as it did — meeting users in tools they already have open beats requiring a dedicated CLI session.

**Priority: Medium** — high strategic value if Sutra wants to be a deployable assistant rather than a dev-focused harness library, but it's a genuinely new surface area (auth, per-channel permission policy, rate limiting) rather than an extension of existing code.

#### 26. Proactive/scheduled triggers
*Inspired by: OpenClaw ("24/7 autonomous" positioning), Claude Code (`/loop`, `ScheduleWakeup`, scheduled cloud agents)*

Sutra is purely reactive — `run()` requires a `user_input`. Add a scheduler (cron-like, or simple interval-based) that can initiate a turn with a *synthetic* prompt ("Check for new open tickets and triage any that are unassigned") on its own schedule, reusing the exact same `harness.run()` path an interactive user would take. Pairs naturally with spec 15 (hooks) for "run this hook every N minutes" style automation.

**Priority: Medium** — straightforward to build (it's just an external loop calling `run()` on a timer), but needs spec 25 or a persistent output sink to be useful (a proactive agent with nowhere to report to isn't very proactive).

#### 27. Self-authored tools (proceed with real caution)
*Inspired by: OpenClaw and Hermes Agent (both feature agents that write their own new skills/tools)*

Flagging this because it's a genuine capability gap versus two of the four comparison systems — but I want to be direct that this is the one item on this list I'd push back on building without a lot more design work first. Letting an LLM write and then *execute* its own new tool code is a fundamentally different trust model than everything else in Sutra, which is built entirely around the idea that tools are pre-registered, reviewed, and permission-leveled by a human before an agent ever sees them. If this gets built at all, it should require: spec 17's sandboxing as a hard prerequisite (never run agent-authored code in-process), CRITICAL permission level with no override, and a human-review step before a newly authored tool is added to any `ToolRegistry` a *future* session can call — i.e., treat it as a PR, not a runtime action.

**Priority: Low (by design)** — real capability gap, but the cost of getting it wrong (arbitrary code execution from an LLM) is high enough that I'd rather it stay explicitly deprioritized than get built quickly.

### Group F — Model compatibility

#### 28. Hermes-format tool-calling adapter
*Inspired by: the Hermes model family's `<tools>`/`<tool_call>`/`<tool_response>` XML-tag function-calling spec*

`11-model-client.md` already documents the flat OpenAI/Ollama-style wire format as the harness's native schema, with `AnthropicModelClient` needing its own adapter for Anthropic's block-structured format. A meaningful number of open-weight models (not just ones branded "Hermes") are specifically fine-tuned on Nous Research's XML-tag tool-calling format rather than OpenAI-style JSON schemas in the request body. Add a `HermesFormatAdapter` (or a mode flag on `OllamaModelClient`) that renders `tools`/`messages` into the `<tools>...</tools>` system-prompt block and parses `<tool_call>{...}</tool_call>` out of the model's raw text stream instead of relying on a structured tool-calling API response. This is the most concrete, lowest-ambiguity item in this whole document — it's a well-documented, stable spec (see [github.com/NousResearch/Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling)), not a guess about an unverified product feature.

**Priority: Medium-High** — cheap relative to its payoff (meaningfully improves tool-calling reliability on a real slice of local models beyond `gpt-oss`), and it's purely additive to `models/`.

---

## Part 5 — Recommended build order

If I were sequencing this backlog, in order:

1. **Fix the spec drift** (Part 1) — near-zero cost, restores trust in `specs/` as ground truth.
2. **Spec 20 (Agent-as-tool)** — highest value-to-risk ratio; reuses the existing, well-tested handoff-interception pattern.
3. **Spec 15 (Hooks)** — the single biggest extensibility gap versus Claude Code; unlocks a lot of the "we could bolt this on later" items elsewhere in the existing `specs/*/Enhancement Ideas` sections (audit logging, custom veto logic, metrics) without a bespoke spec for each.
4. **Spec 18 (Sessions API)** — closes a gap Sutra's own `13-cli.md` already flagged as wanted (`--resume`).
5. **Spec 28 (Hermes-format adapter)** — cheap, concrete, improves real local-model compatibility.
6. Everything else in Groups A/D on demand; **defer Group E** (multi-channel, scheduling) until there's a concrete deployment target that needs it, and **hold spec 27 (self-authored tools)** pending a dedicated safety design review — don't build it opportunistically.
