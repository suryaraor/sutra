from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SUMMARY = "summary"


class HarnessStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED_FOR_PERMISSION = "paused_for_permission"
    PAUSED_FOR_HANDOFF = "paused_for_handoff"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class Message:
    role: Role
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    input_tokens: int = 0
    output_tokens: int = 0
    created_at: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        data = dict(data)
        data["role"] = Role(data["role"])
        return cls(**data)

    def approx_tokens(self) -> int:
        # 1 token ~= 4 chars is a reasonable, dependency-free heuristic.
        return max(1, len(self.content) // 4) + self.input_tokens + self.output_tokens


@dataclass
class ExecutionStep:
    step_number: int
    action: str  # "model_call" | "tool_call" | "handoff" | "permission_wait" | "compaction" | "guardrail_block" | ...
    detail: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PendingPermission:
    request_id: str
    tool_name: str
    arguments: Dict[str, Any]
    reason: str
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HarnessState:
    """Centralized, serializable state for a single harness run.

    This object — not instance attributes scattered across the loop — is the
    single source of truth for conversation history, execution progress, and
    pause/resume metadata. It can be checkpointed to JSON and rehydrated,
    which is what makes permission-gate pauses and process restarts safe.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: HarnessStatus = HarnessStatus.IDLE
    system_prompt: str = ""
    messages: List[Message] = field(default_factory=list)
    execution_steps: List[ExecutionStep] = field(default_factory=list)
    active_agent_id: str = "root"
    agent_stack: List[str] = field(default_factory=lambda: ["root"])
    pending_permission: Optional[PendingPermission] = None
    compaction_count: int = 0
    step_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # -- mutation helpers ---------------------------------------------------

    def append_message(self, message: Message) -> None:
        self.messages.append(message)
        self.updated_at = time.time()

    def record_step(self, action: str, **detail: Any) -> ExecutionStep:
        self.step_count += 1
        step = ExecutionStep(step_number=self.step_count, action=action, detail=detail)
        self.execution_steps.append(step)
        self.updated_at = time.time()
        return step

    def push_agent(self, agent_id: str) -> None:
        self.agent_stack.append(agent_id)
        self.active_agent_id = agent_id
        self.updated_at = time.time()

    def pop_agent(self) -> str:
        if len(self.agent_stack) > 1:
            self.agent_stack.pop()
        self.active_agent_id = self.agent_stack[-1]
        self.updated_at = time.time()
        return self.active_agent_id

    def total_tokens(self) -> int:
        return sum(m.input_tokens + m.output_tokens for m in self.messages)

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "system_prompt": self.system_prompt,
            "messages": [m.to_dict() for m in self.messages],
            "execution_steps": [s.to_dict() for s in self.execution_steps],
            "active_agent_id": self.active_agent_id,
            "agent_stack": list(self.agent_stack),
            "pending_permission": self.pending_permission.to_dict() if self.pending_permission else None,
            "compaction_count": self.compaction_count,
            "step_count": self.step_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarnessState":
        state = cls(
            session_id=data["session_id"],
            status=HarnessStatus(data["status"]),
            system_prompt=data.get("system_prompt", ""),
            active_agent_id=data.get("active_agent_id", "root"),
            agent_stack=list(data.get("agent_stack", ["root"])),
            compaction_count=data.get("compaction_count", 0),
            step_count=data.get("step_count", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )
        state.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        state.execution_steps = [
            ExecutionStep(
                step_number=s["step_number"],
                action=s["action"],
                detail=s.get("detail", {}),
                created_at=s.get("created_at", time.time()),
            )
            for s in data.get("execution_steps", [])
        ]
        pp = data.get("pending_permission")
        state.pending_permission = PendingPermission(**pp) if pp else None
        return state

    @classmethod
    def from_json(cls, blob: str) -> "HarnessState":
        return cls.from_dict(json.loads(blob))

    def checkpoint(self) -> str:
        """Return a durable snapshot string. Pair with `HarnessState.from_json`."""
        return self.to_json()
