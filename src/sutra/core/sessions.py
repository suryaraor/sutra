"""Full-state checkpoint/resume persistence for harness sessions.

This is a different concern from `sutra.memory.store.MemoryStore`: that
store keeps a human-readable *summary* of a session (`.memory/sessions/<id>/
session.md`) for context injection and inspection. This module persists the
*entire* replayable `HarnessState` — every message, execution step, and
pause/resume marker — as JSON, so a session can be resumed exactly where it
left off (`sutra chat --resume <session_id>`).

Layout under the store root (default `./.sutra_sessions`):

    .sutra_sessions/
    └── <session_id>.json          # full HarnessState.to_json() snapshot
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from sutra.core.state import HarnessState


class SessionStore(ABC):
    """Persists and retrieves full `HarnessState` snapshots by session id."""

    @abstractmethod
    def load(self, session_id: str) -> Optional[HarnessState]:
        """Return the stored state for `session_id`, or None if not found."""

    @abstractmethod
    def save(self, state: HarnessState) -> None:
        """Persist `state`, keyed by `state.session_id`."""

    @abstractmethod
    def list_sessions(self) -> List[str]:
        """Return known session ids, newest-first."""


class FileSessionStore(SessionStore):
    """Reads and writes `HarnessState` snapshots as one JSON file per session,
    under a root directory. Pure I/O — mirrors `MemoryStore`'s constructor
    style (`root: Optional[Path] = None`, default under CWD)."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or Path.cwd() / ".sutra_sessions"

    # -- path helpers ---------------------------------------------------------

    def session_path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    # -- I/O --------------------------------------------------------------------

    def load(self, session_id: str) -> Optional[HarnessState]:
        path = self.session_path(session_id)
        if not path.exists():
            return None
        return HarnessState.from_json(path.read_text(encoding="utf-8"))

    def save(self, state: HarnessState) -> None:
        path = self.session_path(state.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(state.to_json(), encoding="utf-8")

    def list_sessions(self) -> List[str]:
        if not self.root.exists():
            return []
        entries = [p for p in self.root.iterdir() if p.is_file() and p.suffix == ".json"]
        entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem for p in entries]
