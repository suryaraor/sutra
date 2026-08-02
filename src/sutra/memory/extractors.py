"""Heuristic, regex-based fact extraction from user messages — no extra
model call required, so memory can update on every turn for free. This is
a `MemoryManager`-internal implementation detail; swap in an LLM-backed
extractor later by implementing the same `extract(text) -> list[(label,
value)]` shape without touching the manager or harness integration.
"""

from __future__ import annotations

import re
from typing import List, Pattern, Tuple

# (label, compiled pattern with exactly one capture group)
_IDENTITY_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    ("name", re.compile(r"\bmy name is\s+([A-Z][a-zA-Z'-]{1,30})\b", re.IGNORECASE)),
    ("name", re.compile(r"\bcall me\s+([A-Z][a-zA-Z'-]{1,30})\b", re.IGNORECASE)),
    ("role", re.compile(r"\bi'?m an?\s+([a-zA-Z][\w\s-]{2,40}?)(?:[.,]|$| at | for | who )", re.IGNORECASE)),
    ("role", re.compile(r"\bi am an?\s+([a-zA-Z][\w\s-]{2,40}?)(?:[.,]|$| at | for | who )", re.IGNORECASE)),
    ("organization", re.compile(r"\bi work (?:at|for)\s+([\w&.,' -]{2,40}?)(?:[.,]|$)", re.IGNORECASE)),
    ("preference", re.compile(r"\bi prefer\s+(.{3,80}?)(?:[.]|$)", re.IGNORECASE)),
    ("preference", re.compile(r"\bplease always\s+(.{3,80}?)(?:[.]|$)", re.IGNORECASE)),
    ("note", re.compile(r"\bremember(?: that)?\s+(.{3,120}?)(?:[.]|$)", re.IGNORECASE)),
]

# Behavioral directives aimed at how the *agent itself* should act — these
# are instance-wide (persona) candidates, deliberately kept separate from
# per-user identity facts and NOT auto-applied to the live system prompt
# (see MemoryManager docstring for why).
_PERSONA_DIRECTIVE_PATTERNS: List[Pattern[str]] = [
    re.compile(r"\bfrom now on,?\s+(.{3,120}?)(?:[.]|$)", re.IGNORECASE),
    re.compile(r"\byou should always\s+(.{3,120}?)(?:[.]|$)", re.IGNORECASE),
    re.compile(r"\byou must always\s+(.{3,120}?)(?:[.]|$)", re.IGNORECASE),
    re.compile(r"\byou should never\s+(.{3,120}?)(?:[.]|$)", re.IGNORECASE),
]


class IdentityExtractor:
    """Pulls name/role/organization/preference/note facts out of raw user text."""

    def extract(self, text: str) -> List[Tuple[str, str]]:
        found: List[Tuple[str, str]] = []
        for label, pattern in _IDENTITY_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(1).strip().rstrip(".,")
                if value:
                    found.append((label, value))
        return found


class PersonaDirectiveExtractor:
    """Pulls candidate behavioral directives ("always...", "never...") out of
    raw user text. Output is recorded for human review, not auto-applied."""

    def extract(self, text: str) -> List[str]:
        found: List[str] = []
        for pattern in _PERSONA_DIRECTIVE_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(1).strip().rstrip(".,")
                if value:
                    found.append(value)
        return found
