from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

from sutra.core.state import Message, Role

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - optional dependency
    _ENCODING = None


def count_tokens(text: str) -> int:
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return max(1, len(text) // 4)


def count_message_tokens(messages: List[Message]) -> int:
    return sum(count_tokens(m.content) + 4 for m in messages)


SummarizerFn = Callable[[List[Message]], Awaitable[str]]


async def _default_summarizer(messages: List[Message]) -> str:
    """Extractive fallback summarizer used when no LLM summarizer is wired in."""
    lines = []
    for m in messages:
        snippet = m.content.strip().replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        lines.append(f"[{m.role.value}] {snippet}")
    return "Summary of earlier turns:\n" + "\n".join(lines)


@dataclass
class CompactionConfig:
    token_threshold: int = 6000
    preserve_last_n_turns: int = 6


class ContextCompactor:
    """Monitors the conversation token window and compacts it in place.

    The system prompt (held outside `messages`, on `HarnessState`) and the
    most recent `preserve_last_n_turns` messages are never touched.
    Everything older is collapsed into a single synthetic `Role.SUMMARY`
    message so the window stays bounded regardless of conversation length.
    """

    def __init__(
        self,
        config: Optional[CompactionConfig] = None,
        summarizer: Optional[SummarizerFn] = None,
    ) -> None:
        self.config = config or CompactionConfig()
        self.summarizer = summarizer or _default_summarizer

    def needs_compaction(self, messages: List[Message]) -> bool:
        return count_message_tokens(messages) > self.config.token_threshold

    async def compact(self, messages: List[Message]) -> List[Message]:
        if not self.needs_compaction(messages):
            return messages

        preserve_n = self.config.preserve_last_n_turns
        if len(messages) <= preserve_n:
            return messages

        head, tail = messages[:-preserve_n], messages[-preserve_n:]

        # Fold any pre-existing summary at the front into the new one so
        # repeated compaction passes never lose information.
        already_summarized: Optional[Message] = None
        if head and head[0].role == Role.SUMMARY:
            already_summarized, head = head[0], head[1:]

        if not head:
            return messages

        summary_text = await self.summarizer(head)
        if already_summarized:
            summary_text = already_summarized.content + "\n\n" + summary_text

        summary_message = Message(
            role=Role.SUMMARY,
            content=summary_text,
            input_tokens=0,
            output_tokens=count_tokens(summary_text),
        )
        return [summary_message, *tail]
