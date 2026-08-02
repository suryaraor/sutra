from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from sutra.core.exceptions import PermissionDeniedError


class PermissionLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PermissionRequest:
    request_id: str
    tool_name: str
    arguments: Dict[str, Any]
    level: PermissionLevel
    reason: str
    tool_call_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    approved: Optional[bool] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "level": self.level.value,
            "reason": self.reason,
            "tool_call_id": self.tool_call_id,
            "created_at": self.created_at,
            "resolved": self.resolved,
            "approved": self.approved,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at,
        }


class PermissionGate:
    """Human-in-the-loop pause point for high-risk tool calls.

    Tools carry a `PermissionLevel`. Any call at or above
    `require_approval_from` blocks the harness loop on an `asyncio.Future`
    until an external actor calls `resolve()` — typically from an HTTP
    endpoint reacting to a human clicking "approve" in a UI driven by the
    SSE `permission_request` event.
    """

    _ORDER = {
        PermissionLevel.LOW: 0,
        PermissionLevel.MEDIUM: 1,
        PermissionLevel.HIGH: 2,
        PermissionLevel.CRITICAL: 3,
    }

    def __init__(self, require_approval_from: PermissionLevel = PermissionLevel.HIGH) -> None:
        self.require_approval_from = require_approval_from
        self._pending: Dict[str, PermissionRequest] = {}
        self._futures: Dict[str, "asyncio.Future[bool]"] = {}

    def requires_gate(self, level: PermissionLevel) -> bool:
        return self._ORDER[level] >= self._ORDER[self.require_approval_from]

    def open_request(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        level: PermissionLevel,
        reason: str,
        *,
        tool_call_id: Optional[str] = None,
    ) -> PermissionRequest:
        request_id = uuid.uuid4().hex[:12]
        request = PermissionRequest(
            request_id=request_id,
            tool_name=tool_name,
            arguments=arguments,
            level=level,
            reason=reason,
            tool_call_id=tool_call_id,
        )
        self._pending[request_id] = request
        self._futures[request_id] = asyncio.get_running_loop().create_future()
        return request

    async def await_decision(self, request_id: str, *, timeout: Optional[float] = None) -> bool:
        future = self._futures[request_id]
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            tool_name = self._pending.pop(request_id, None)
            self._futures.pop(request_id, None)
            raise PermissionDeniedError(
                f"Permission request {request_id} timed out awaiting human approval.",
                tool_name=tool_name.tool_name if tool_name else "unknown",
                request_id=request_id,
            ) from exc

    def resolve(self, request_id: str, *, approved: bool, actor: str = "unknown") -> PermissionRequest:
        if request_id not in self._pending:
            raise KeyError(f"No pending permission request with id {request_id!r}.")
        request = self._pending.pop(request_id)
        request.resolved = True
        request.approved = approved
        request.resolved_by = actor
        request.resolved_at = time.time()
        future = self._futures.pop(request_id)
        if not future.done():
            future.set_result(approved)
        return request

    def get_pending(self, request_id: str) -> Optional[PermissionRequest]:
        return self._pending.get(request_id)

    def list_pending(self) -> Dict[str, PermissionRequest]:
        return dict(self._pending)
