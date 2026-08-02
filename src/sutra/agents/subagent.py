from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class SubagentProfile:
    agent_id: str
    name: str
    system_prompt: str
    allowed_tools: Set[str] = field(default_factory=set)
    model_override: Optional[str] = None
    description: str = ""


class SubagentRegistry:
    """Tracks specialized agent profiles the parent harness can instantiate.

    Each profile carries its own system instructions and a restricted tool
    subset, so a subagent can only ever see (and therefore only ever call)
    the tools its profile explicitly whitelists.
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, SubagentProfile] = {}

    def register(self, profile: SubagentProfile) -> None:
        self._profiles[profile.agent_id] = profile

    def get(self, agent_id: str) -> SubagentProfile:
        if agent_id not in self._profiles:
            raise KeyError(f"No subagent registered with id '{agent_id}'.")
        return self._profiles[agent_id]

    def has(self, agent_id: str) -> bool:
        return agent_id in self._profiles

    def ids(self) -> List[str]:
        return sorted(self._profiles.keys())

    def allowed_tools_for(self, agent_id: str) -> Optional[List[str]]:
        if agent_id == "root" or not self.has(agent_id):
            return None  # the root frame has no tool-name restriction by default
        return sorted(self.get(agent_id).allowed_tools)
