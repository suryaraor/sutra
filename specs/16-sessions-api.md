# Feature: Sessions API

**Module:** `src/sutra/core/sessions.py`  
**Classes:** `SessionStore`, `FileSessionStore`

## Purpose

Full-state checkpoint/resume for harness runs. Where `sutra.memory.store.MemoryStore` keeps a human-readable *summary* of a session (`.memory/sessions/<id>/session.md`) for context injection, this module persists the entire replayable `HarnessState` — every message, execution step, active agent, and pause/resume marker — as JSON. That's what makes `sutra chat --resume session-abc` possible: the harness doesn't just remember *that* a conversation happened, it can rehydrate exactly where the conversation left off and keep going.

This closes the gap `specs/13-cli.md` already flagged under Enhancement Ideas ("save and load chat sessions from JSON... so users can pick up a conversation after exiting").

## Current Behavior

### Layout

```
.sutra_sessions/
└── <session_id>.json          # full HarnessState.to_json() snapshot
```

Default root is `./.sutra_sessions` relative to CWD, same convention as `MemoryStore`'s `./.memory` default — configurable via the constructor.

### `SessionStore` (ABC)

```python
class SessionStore(ABC):
    def load(self, session_id: str) -> Optional[HarnessState]: ...
    def save(self, state: HarnessState) -> None: ...
    def list_sessions(self) -> List[str]: ...
```

`save()` takes no separate id parameter — the session id always comes from `state.session_id`, so a caller can never accidentally save a state object under the wrong key.

### `FileSessionStore`

```python
FileSessionStore(root: Optional[Path] = None)
```

| Method | Behavior |
|---|---|
| `session_path(session_id)` | `<root>/<session_id>.json` |
| `load(session_id)` | Reads and parses via `HarnessState.from_json()`; returns `None` (does not raise) if the file doesn't exist |
| `save(state)` | Writes `state.to_json()` to `session_path(state.session_id)`, creating parent directories as needed |
| `list_sessions()` | Returns session ids sorted newest-first by file mtime; `[]` if the root doesn't exist yet |

Pure I/O — no business logic, no validation of state contents. Round-trip fidelity (messages, execution steps, pending permissions, enums) is entirely delegated to `HarnessState.to_json()` / `HarnessState.from_json()`, which already guarantee it.

### CLI integration (`src/sutra/cli/app.py`)

| Flag/command | Effect |
|---|---|
| `--resume <session_id>` | Shared flag (`_build_common_parser()`). In `chat_loop()`, attempts `FileSessionStore().load(session_id)`; if found, passes it as `state=` into `build_harness()` so the resumed `AsynchronousHarnessLoop` picks up the existing conversation, agent stack, and step count. If not found, prints a friendly notice and starts a fresh session rather than crashing. |
| (every chat turn) | After each `drive_turn()` call inside `chat_loop()`'s main loop, `FileSessionStore().save(harness.state)` checkpoints automatically — not just on exit, so a crash or `Ctrl+C` mid-conversation only loses the in-flight turn, never prior ones. |
| Chat intro panel | Prints `harness.state.session_id` prominently so the user knows what to pass to `--resume` next time. |
| `sutra sessions` | Lists every session under `.sutra_sessions/`, newest-first, showing session id, status, active agent, and message count for each — enough to pick which one to resume. |

`build_harness()` grew a `state: Optional[HarnessState] = None` passthrough parameter to make this possible; it's forwarded straight to `AsynchronousHarnessLoop(..., state=state)`, which already supported resuming from an existing state object (nothing in `harness.py` needed to change). `sutra run <scenario>` and `sutra demo` are intentionally left without `--resume` wiring — they're one-shot scripted scenarios where resuming mid-run has no sensible meaning; only `chat_loop` threads it through.

## Enhancement Ideas

- **Session metadata index**: like `MemoryStore.write_index()`, maintain a lightweight `SESSIONS.md`/`.json` index alongside the per-session files so `list_sessions()` doesn't need to open and parse every file just to show a summary.
- **TTL / pruning**: add a `prune(older_than_days: int)` method so `.sutra_sessions/` doesn't grow unbounded across months of chat use.
- **Compression**: for long-running sessions with many messages, gzip the JSON on write (`<session_id>.json.gz`) — `HarnessState.to_json()` output compresses well given the repetitive dict shape.
- **Partial resume / branching**: `fork(session_id, at_message_id) -> HarnessState` to rewind to an earlier point and continue down a different path, useful for "what if" replays or retrying after a bad tool call.
- **Cross-store linkage**: record the `.sutra_sessions/<id>.json` path (or a boolean "full state available") in the `.memory/sessions/<id>/session.md` frontmatter, so `/memory` and `sutra sessions` can cross-reference instead of being two disconnected views of overlapping data.
- **Async I/O**: `save()` currently does a blocking `Path.write_text()` inside an async chat loop; swap to `aiofiles` or run it in a thread executor if session state grows large enough for the write to be noticeable turn-over-turn.
- **Session locking**: guard against two processes resuming and saving the same `session_id` concurrently (e.g. an advisory lock file, or a `updated_at` optimistic-concurrency check on save).
- **Export/import**: `sutra sessions export <id> > file.json` / `sutra sessions import file.json` for sharing a session or moving it between machines, distinct from the raw file already being plain JSON on disk.
