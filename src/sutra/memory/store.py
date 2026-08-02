"""Filesystem persistence for Sutra memory: plain markdown files with a
small flat-key frontmatter block, no extra dependencies (no PyYAML) since
the metadata here is always a flat set of string key/value pairs.

Layout under the store root (default `./.memory`):

    .memory/
    ├── MEMORY.md                      # top-level index, regenerated on write
    ├── instance/
    │   └── persona.md                 # Sutra's own persona (instance-wide)
    ├── users/
    │   └── <slug(user_id)>/
    │       └── identity.md            # who this user is
    └── sessions/
        └── <session_id>/
            └── session.md             # what happened in this session
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class MemoryScope(str, Enum):
    INSTANCE = "instance"
    USER = "user"
    SESSION = "session"


def slugify(raw: str) -> str:
    """Filesystem-safe directory name for an arbitrary user/session id."""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw.strip()).strip("_")
    return slug.lower() or "unknown"


@dataclass
class MemoryDocument:
    """A single markdown memory file: flat frontmatter + free-text body."""

    frontmatter: Dict[str, str] = field(default_factory=dict)
    body: str = ""

    def to_markdown(self) -> str:
        fm_lines = "\n".join(f"{k}: {v}" for k, v in self.frontmatter.items())
        fm_block = f"---\n{fm_lines}\n---\n\n" if fm_lines else ""
        return f"{fm_block}{self.body.strip()}\n"

    @classmethod
    def from_markdown(cls, text: str) -> "MemoryDocument":
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                frontmatter: Dict[str, str] = {}
                for line in parts[1].strip().splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        frontmatter[key.strip()] = value.strip()
                return cls(frontmatter=frontmatter, body=parts[2].strip())
        return cls(frontmatter={}, body=text.strip())


class MemoryStore:
    """Reads and writes `MemoryDocument`s under a root directory, organized
    by scope. Pure I/O — no extraction/summarization logic lives here."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or Path.cwd() / ".memory"
        self.instance_dir = self.root / "instance"
        self.users_dir = self.root / "users"
        self.sessions_dir = self.root / "sessions"

    # -- path helpers -------------------------------------------------------

    def persona_path(self) -> Path:
        return self.instance_dir / "persona.md"

    def identity_path(self, user_id: str) -> Path:
        return self.users_dir / slugify(user_id) / "identity.md"

    def session_path(self, session_id: str) -> Path:
        return self.sessions_dir / session_id / "session.md"

    def index_path(self) -> Path:
        return self.root / "MEMORY.md"

    # -- I/O ------------------------------------------------------------------

    def read(self, path: Path) -> Optional[MemoryDocument]:
        if not path.exists():
            return None
        return MemoryDocument.from_markdown(path.read_text(encoding="utf-8"))

    def write(self, path: Path, document: MemoryDocument) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document.to_markdown(), encoding="utf-8")

    def list_user_ids(self) -> List[str]:
        if not self.users_dir.exists():
            return []
        return sorted(p.name for p in self.users_dir.iterdir() if p.is_dir())

    def list_session_ids(self) -> List[str]:
        if not self.sessions_dir.exists():
            return []
        entries = [p for p in self.sessions_dir.iterdir() if p.is_dir()]
        entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.name for p in entries]

    def write_index(self) -> None:
        """Regenerate the top-level MEMORY.md summary of everything on disk."""
        lines = ["# Sutra Memory Index", "", f"_Last updated: {iso_now()}_", ""]

        persona = self.read(self.persona_path())
        if persona:
            lines.append(f"- **Instance persona**: `instance/persona.md` (updated {persona.frontmatter.get('updated_at', '?')})")

        user_ids = self.list_user_ids()
        if user_ids:
            lines.append("- **Users**:")
            for uid in user_ids:
                doc = self.read(self.identity_path(uid))
                updated = doc.frontmatter.get("updated_at", "?") if doc else "?"
                lines.append(f"  - `{uid}` — `users/{uid}/identity.md` (updated {updated})")

        session_ids = self.list_session_ids()
        if session_ids:
            lines.append("- **Sessions** (most recent first):")
            for sid in session_ids[:20]:
                doc = self.read(self.session_path(sid))
                updated = doc.frontmatter.get("updated_at", "?") if doc else "?"
                user = doc.frontmatter.get("user_id", "?") if doc else "?"
                lines.append(f"  - `{sid}` (user `{user}`) — `sessions/{sid}/session.md` (updated {updated})")

        self.index_path().parent.mkdir(parents=True, exist_ok=True)
        self.index_path().write_text("\n".join(lines) + "\n", encoding="utf-8")


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
