from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EventType(str, Enum):
    SESSION_START = "session_start"
    GUARDRAIL_BLOCK = "guardrail_block"
    TOKEN = "token"
    MESSAGE_COMPLETE = "message_complete"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PERMISSION_REQUEST = "permission_request"
    PERMISSION_RESOLVED = "permission_resolved"
    HANDOFF = "handoff"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"
    COMPACTION = "compaction"
    ERROR = "error"
    HOOK_VETO = "hook_veto"
    DONE = "done"


@dataclass
class SSEEvent:
    """A single Server-Sent-Events frame.

    `to_sse()` renders the wire format expected by browsers'
    `EventSource` / any SSE-compatible frontend: an `event:` line naming
    the event type followed by a `data: {...}` JSON line and a blank-line
    terminator.
    """

    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created_at: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        payload = {
            "id": self.event_id,
            "type": self.type.value,
            "created_at": self.created_at,
            **self.data,
        }
        return f"event: {self.type.value}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def format_sse(event_type: EventType, data: Optional[Dict[str, Any]] = None) -> str:
    return SSEEvent(type=event_type, data=data or {}).to_sse()
