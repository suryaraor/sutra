"""Minimal single-agent toolkit (weather / dice / email) used for a quick,
low-ceremony live smoke test — shared by `examples/ollama_live_demo.py` and
the `sutra` CLI's `--toolkit simple` mode.
"""

from __future__ import annotations

import random

from sutra.core.permissions import PermissionLevel
from sutra.tools.registry import Tool, ToolParameter, ToolRegistry

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. Use get_current_weather and roll_dice "
    "freely. send_email is high-risk and requires human approval, but you should still call it "
    "when the user asks you to send an email."
)


async def get_current_weather(location: str) -> dict:
    # Canned data stand-in for a real weather API, deterministic for the demo.
    return {"location": location, "condition": "partly cloudy", "temp_c": 22, "humidity_pct": 58}


async def roll_dice(sides: int = 6) -> dict:
    return {"sides": sides, "result": random.randint(1, sides)}


async def send_email(to: str, subject: str, body: str) -> dict:
    # Simulated send: real deployments would call an actual mail API here.
    return {"status": "sent", "to": to, "subject": subject, "body_preview": body[:120]}


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(
        Tool(
            name="get_current_weather",
            description="Get the current weather conditions for a location.",
            parameters=[ToolParameter("location", "string", "City name, e.g. 'Tokyo'.")],
            handler=get_current_weather,
            permission_level=PermissionLevel.LOW,
        )
    )
    registry.register_tool(
        Tool(
            name="roll_dice",
            description="Roll an N-sided die and return the result.",
            parameters=[ToolParameter("sides", "integer", "Number of sides on the die.", required=False)],
            handler=roll_dice,
            permission_level=PermissionLevel.LOW,
        )
    )
    registry.register_tool(
        Tool(
            name="send_email",
            description="Send an email to a recipient. Sends real, externally-visible email.",
            parameters=[
                ToolParameter("to", "string", "Recipient email address."),
                ToolParameter("subject", "string", "Email subject line."),
                ToolParameter("body", "string", "Email body text."),
            ],
            handler=send_email,
            permission_level=PermissionLevel.CRITICAL,
        )
    )
    return registry
