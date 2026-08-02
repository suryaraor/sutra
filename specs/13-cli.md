# Feature: CLI (`sutra` command)

**Module:** `src/sutra/cli/`  
**Entry point:** `sutra.cli.app:main` (registered via `pyproject.toml` `[project.scripts]`)  
**Renderer:** `src/sutra/cli/render.py` (`HarnessRenderer`)

## Purpose

Interactive terminal interface for the harness. Provides an REPL chat session, a library of pre-built scenarios, a fully automated demo tour, and a memory inspector — all rendered with Rich for live streaming display.

## Current Behavior

### Commands

```
sutra                                  # interactive chat, full Contoso IT+Finance toolkit
sutra chat --toolkit simple            # interactive chat, weather/dice/email toolkit
sutra list                             # list bundled one-shot scenarios
sutra run it-incident                  # run a scenario non-interactively
sutra run finance-transfer --auto-approve
sutra demo                             # fully automated, narrated tour of every component
sutra memory                           # show what's in .memory/ (same as /memory inside chat)
sutra --model llama3.1 chat            # point at a different Ollama model
sutra --user alice chat                # explicit identity instead of OS username
sutra --base-url http://server:11434 chat  # remote Ollama endpoint
```

### Global Flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `gpt-oss:20b` | Ollama model name |
| `--base-url` | `http://localhost:11434` | Ollama server URL |
| `--user` | OS username (`getpass.getuser()`) | Identity for memory scoping |

### Bundled Scenarios (`sutra list` / `sutra run <name>`)

| Name | Toolkit | Prompt summary |
|---|---|---|
| `it-incident` | contoso | Checkout broken; check ticket, escalate, restart service |
| `finance-transfer` | contoso | Check account balance, transfer $4500 to vendor |
| `injection-block` | contoso | Prompt injection attempt — triggers guardrail block |
| `weather-email` | simple | Check Tokyo weather, send summary email |

### `sutra demo`

Fully automated, narrated three-act tour:
1. **Injection block** — guardrail blocks the prompt before it reaches the model
2. **IT incident** — root → IT ops handoff → service restart behind permission gate
3. **Finance transfer** — root → finance ops handoff → fund transfer behind permission gate

Budget and compaction thresholds are tuned (`DEMO_BUDGET_CONFIG`, `DEMO_COMPACTION_CONFIG`) so the Budget Monitor warning and Context Compactor both fire visibly during the demo without cutting it off before the permission gate payoff.

Ends with a recap that only checks off components that were actually observed that run.

`/demo all` runs the same tour from inside an active chat session.

### Interactive Chat Slash Commands

| Command | Description |
|---|---|
| `/memory` | Show `.memory/` contents |
| `/demo all` | Run the automated demo tour |
| `/exit` or `/quit` | Leave the chat |

### `HarnessRenderer`

Rich-based live renderer in `cli/render.py`. Maps SSE event types to terminal output:

| Event | Rendered as |
|---|---|
| `session_start` | Dim session ID line |
| `token` | Streamed text to a live display panel |
| `message_complete` | End of live panel |
| `tool_call` | Spinner with tool name + arguments |
| `tool_result` | Result preview (truncated) |
| `permission_request` | Bordered panel with tool name, arguments, reason |
| `permission_resolved` | Approved/denied status line |
| `handoff` | Handoff banner (from → to, reason) |
| `guardrail_block` | Red warning panel |
| `budget_warning` | Yellow dim notice |
| `budget_exceeded` | Red error panel |
| `compaction` | Dim compaction notice |
| `error` | Red error panel |
| `done` | Status summary line |

### `build_harness()`

Factory function that wires together the full harness:

```python
build_harness(
    model="gpt-oss:20b",
    base_url="http://localhost:11434",
    toolkit="contoso",         # or "simple"
    user_id="alice",
    budget_config=None,        # uses DEFAULT_BUDGET_CONFIG
    compaction_config=None,    # uses default CompactionConfig
)
```

### `drive_turn()`

Orchestrates a full user turn including permission-gate pauses:
1. Runs `harness.run(prompt)` and renders each event
2. If paused for permission: prompts `y/n` (or auto-approves with `--auto-approve`)
3. Calls `harness.resume_after_permission()` and renders continuation
4. Repeats until no more permission requests
5. Prints budget summary (steps, tokens, USD)

### Memory in CLI

`sutra memory` (and `/memory` inside chat) calls `MemoryManager.summary()` and displays a Rich table showing:
- Persona operating principles
- Unreviewed persona directives
- Current user's identity facts
- Last 5 sessions with status

Identity defaults to the OS username and persists automatically — no login step required.

## Enhancement Ideas

- **Provider flag**: `--provider anthropic|openai|ollama` so users can switch providers without specifying a full model path.
- **`sutra run` output modes**: `--json` flag to emit raw JSON SSE events instead of Rich rendering, for pipeline integration.
- **Conversation history**: save and load chat sessions from JSON (`sutra chat --resume session-abc`) so users can pick up a conversation after exiting.
- **Toolkit plugin loading**: `--toolkit path/to/my_toolkit.py` that dynamically imports a module providing `build_tool_registry()` and optionally `build_subagent_registry()`.
- **Multi-turn scenario files**: YAML-based scenario files with multiple turns, expected events, and assertions — turning scenarios into a test harness.
- **`/tools` slash command**: inside chat, list all registered tools with their descriptions and permission levels.
- **`/budget` slash command**: show current budget snapshot mid-conversation.
- **`/clear` slash command**: reset the harness state (start a fresh session without restarting the process).
- **Tab completion**: add `argcomplete` support for `sutra run <tab>` to autocomplete scenario names.
- **Windows encoding fix generalization**: the `sys.stdout.reconfigure(encoding="utf-8")` workaround is done at module import time; document it and consider moving it to a `setup()` function to avoid side effects on import.
- **Rich theme customization**: let users set their preferred color scheme via a `.sutra/theme.json` config file.
