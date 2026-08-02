from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

from sutra.core.exceptions import HandoffError

if TYPE_CHECKING:
    from sutra.agents.subagent import SubagentRegistry


@dataclass
class HandoffRequest:
    target_agent_id: str
    reason: str
    context_payload: Dict[str, Any] = field(default_factory=dict)


class HandoffRouter:
    """State-transfer mechanism for switching the active execution frame.

    A subagent signals a handoff by calling the reserved `__handoff__`
    pseudo-tool (wired up in the harness loop). The router validates the
    target exists in the `SubagentRegistry` and produces a briefing message
    so the newly active agent has explicit context about why control was
    transferred; the harness loop itself pushes/pops the agent stack.
    """

    def __init__(self, subagent_registry: "SubagentRegistry") -> None:
        self.subagent_registry = subagent_registry

    def validate(self, request: HandoffRequest) -> None:
        if not self.subagent_registry.has(request.target_agent_id):
            raise HandoffError(
                f"Cannot hand off to unknown agent '{request.target_agent_id}'. "
                f"Registered agents: {sorted(self.subagent_registry.ids())}."
            )

    def build_briefing(self, request: HandoffRequest, *, from_agent_id: str) -> str:
        payload_lines = "\n".join(f"- {k}: {v}" for k, v in request.context_payload.items())
        return (
            f"[HANDOFF] Control transferred from '{from_agent_id}' to '{request.target_agent_id}'.\n"
            f"Reason: {request.reason}\n"
            f"Context payload:\n{payload_lines if payload_lines else '(none)'}\n\n"
            "You are now the active agent. Proceed immediately using your available tools to "
            "resolve the request — do not just acknowledge this message."
        )
