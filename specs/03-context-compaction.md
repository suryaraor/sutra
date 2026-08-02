# Feature: Context Compaction

**Module:** `src/sutra/core/compaction.py`  
**Class:** `ContextCompactor`

## Purpose

Keeps the conversation token window bounded regardless of conversation length by collapsing older messages into a single summary message. Without compaction, long-running sessions would eventually exceed the model's context window.

## Current Behavior

### Configuration (`CompactionConfig`)

| Field | Default | Description |
|---|---|---|
| `token_threshold` | `6000` | Trigger compaction when message window exceeds this token count |
| `preserve_last_n_turns` | `6` | Always keep the most recent N messages untouched |

### Token Counting

```python
count_tokens(text: str) -> int
```

- Uses `tiktoken` (`cl100k_base` encoding) if installed (`pip install -e ".[tokens]"`).
- Falls back to `len(text) // 4` heuristic (1 token ≈ 4 chars) if `tiktoken` is unavailable.

```python
count_message_tokens(messages: List[Message]) -> int
```

Sums `count_tokens(m.content) + 4` per message (the +4 accounts for role/metadata overhead).

### Compaction Algorithm

`needs_compaction(messages)` returns `True` when `count_message_tokens(messages) > token_threshold`.

`compact(messages)` when triggered:

1. Split: `head = messages[:-preserve_last_n_turns]`, `tail = messages[-preserve_last_n_turns:]`
2. If `len(messages) <= preserve_last_n_turns`: return unchanged.
3. If `head[0]` is already a `SUMMARY` message: peel it off as `already_summarized` so it's folded into the new summary (repeated compaction passes never lose information).
4. Call `await summarizer(head)` to get `summary_text`.
5. If `already_summarized`: prepend its content to `summary_text`.
6. Build a new `Message(role=SUMMARY, ...)` and return `[summary_message, *tail]`.

### Default Summarizer

An extractive fallback used when no LLM summarizer is wired in:

```
Summary of earlier turns:
[user] snippet up to 160 chars...
[assistant] snippet...
[tool] snippet...
```

Role label + first 160 characters of each message, one per line.

### Custom Summarizer

```python
async def my_summarizer(messages: List[Message]) -> str:
    # call LLM to produce a rich summary
    ...

compactor = ContextCompactor(summarizer=my_summarizer)
```

Any `async (List[Message]) -> str` callable works.

### Integration with Harness

Called in `harness.py` before each model call:

```python
if self.compactor.needs_compaction(self.state.messages):
    self.state.messages = await self.compactor.compact(self.state.messages)
    self.state.compaction_count += 1
    yield SSEEvent(EventType.COMPACTION, {...})
```

The `COMPACTION` SSE event carries `compaction_count` and `resulting_messages` for client-side display.

**Note:** The system prompt is held separately on `HarnessState.system_prompt` and is never touched by the compactor.

## Enhancement Ideas

- **LLM-powered summarizer**: wire in an `AnthropicModelClient` or `OllamaModelClient` call to produce semantically rich summaries instead of the extractive fallback.
- **Per-agent compaction thresholds**: subagents with small tool sets rarely need a large window — allow `CompactionConfig` overrides per `SubagentProfile`.
- **Importance-weighted preservation**: instead of a fixed "last N" rule, score messages by recency + whether they contain tool results or permission events, and preserve the highest-scored ones.
- **Compaction preview event**: before compacting, emit a `compaction_preview` event so the client can show which messages are about to be collapsed.
- **Async-safe compaction**: currently compact is called synchronously in the loop; for a production LLM summarizer this could become a bottleneck — consider background compaction with a swap-in on the next hop.
- **Multi-level summaries**: maintain a hierarchy (session summary → window summary) for very long-running agents so no single summary message grows unboundedly.
- **`tiktoken` auto-install**: emit a warning at startup if `tiktoken` is unavailable (currently silently falls back to the char heuristic, which can be off by 2-3x for code-heavy contexts).
