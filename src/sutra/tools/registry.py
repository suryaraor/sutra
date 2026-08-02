from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sutra.core.exceptions import ToolExecutionError, ToolNotFoundError
from sutra.core.permissions import PermissionLevel

ToolHandler = Callable[..., Awaitable[Any]]


@dataclass
class ToolParameter:
    name: str
    type: str  # JSON schema primitive: "string" | "number" | "integer" | "boolean" | "object" | "array"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None

    def to_json_schema(self) -> Dict[str, Any]:
        schema: Dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum:
            schema["enum"] = self.enum
        return schema


@dataclass
class Tool:
    name: str
    description: str
    parameters: List[ToolParameter]
    handler: ToolHandler
    permission_level: PermissionLevel = PermissionLevel.LOW
    allowed_agents: Optional[List[str]] = None  # None = usable by any agent

    def to_schema(self) -> Dict[str, Any]:
        """Anthropic/OpenAI-compatible function-calling schema."""
        properties = {p.name: p.to_json_schema() for p in self.parameters}
        required = [p.name for p in self.parameters if p.required]
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolRegistry:
    """Function-calling registry: schema generation, payload parsing, async execution."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Optional[List[ToolParameter]] = None,
        *,
        permission_level: PermissionLevel = PermissionLevel.LOW,
        allowed_agents: Optional[List[str]] = None,
    ) -> Callable[[ToolHandler], ToolHandler]:
        def decorator(fn: ToolHandler) -> ToolHandler:
            if not asyncio.iscoroutinefunction(fn):
                raise TypeError(f"Tool handler '{name}' must be an async function (async def).")
            self._tools[name] = Tool(
                name=name,
                description=description,
                parameters=parameters or [],
                handler=fn,
                permission_level=permission_level,
                allowed_agents=allowed_agents,
            )
            return fn

        return decorator

    def register_tool(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"No tool registered with name '{name}'.")
        return tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def schemas_for(self, agent_id: Optional[str] = None, allowed_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        out = []
        for tool in self._tools.values():
            if allowed_names is not None and tool.name not in allowed_names:
                continue
            if tool.allowed_agents is not None and agent_id is not None and agent_id not in tool.allowed_agents:
                continue
            out.append(tool.to_schema())
        return out

    def validate_arguments(self, tool: Tool, arguments: Dict[str, Any]) -> None:
        missing = [p.name for p in tool.parameters if p.required and p.name not in arguments]
        if missing:
            raise ToolExecutionError(
                f"Missing required arguments for tool '{tool.name}': {missing}",
                tool_name=tool.name,
                original_error="missing_required_arguments",
            )

    async def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        tool = self.get(name)
        self.validate_arguments(tool, arguments)
        try:
            sig = inspect.signature(tool.handler)
            accepted = set(sig.parameters.keys())
            filtered = {k: v for k, v in arguments.items() if k in accepted} if accepted else arguments
            return await tool.handler(**filtered)
        except ToolExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - intentional: a tool must never crash the master engine
            raise ToolExecutionError(
                f"Tool '{name}' raised during execution: {exc}",
                tool_name=name,
                original_error=repr(exc),
            ) from exc

    def list_tools(self) -> List[str]:
        return sorted(self._tools.keys())

    @staticmethod
    def parse_arguments(raw: str) -> Dict[str, Any]:
        """Parse a raw JSON argument payload as emitted by streaming tool-call deltas."""
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(
                f"Could not parse tool-call arguments as JSON: {raw!r}",
                tool_name="<unknown>",
                original_error=repr(exc),
            ) from exc
        if not isinstance(parsed, dict):
            raise ToolExecutionError(
                f"Tool-call arguments must decode to a JSON object, got {type(parsed).__name__}.",
                tool_name="<unknown>",
                original_error="non_object_arguments",
            )
        return parsed
