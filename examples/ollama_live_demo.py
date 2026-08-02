"""Live integration demo: Sutra harness driven by a real local model via Ollama.

Unlike `examples/finance_workflow.py` (which uses the deterministic,
network-free `MockModelClient` so it can run anywhere without a GPU), this
script talks to an actual running Ollama server and a real downloaded model
(`gpt-oss:20b` by default). Tool-call behavior here is genuinely decided by
the model, not scripted — this is what proves the `OllamaModelClient` wiring
is real, not a stand-in.

Prerequisites:
    - Ollama installed and running (`ollama serve`, or the desktop app).
    - The model pulled: `ollama pull gpt-oss:20b`.
    - `pip install -e ".[ollama]"` (installs httpx).

Run with:  python examples/ollama_live_demo.py
"""

from __future__ import annotations

import asyncio
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sutra.core.budget import Budget, BudgetConfig
from sutra.core.guardrails import Guardrail
from sutra.core.harness import AsynchronousHarnessLoop
from sutra.core.permissions import PermissionGate, PermissionLevel
from sutra.models.ollama_client import OllamaModelClient
from sutra.streaming.sse import EventType
from sutra.tools.registry import Tool, ToolParameter, ToolRegistry

MODEL_NAME = "gpt-oss:20b"


async def get_current_weather(location: str) -> dict:
    # Canned data stand-in for a real weather API, deterministic for the demo.
    return {"location": location, "condition": "partly cloudy", "temp_c": 22, "humidity_pct": 58}


async def roll_dice(sides: int = 6) -> dict:
    import random

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


def print_event(event) -> None:
    if event.type == EventType.TOKEN:
        print(event.data["text"], end="", flush=True)
        return
    print(f"\n[{event.type.value}] {event.data}")


async def main() -> None:
    harness = AsynchronousHarnessLoop(
        model_client=OllamaModelClient(model=MODEL_NAME),
        tool_registry=build_tool_registry(),
        budget=Budget(BudgetConfig(max_usd=1.0, max_input_tokens=50_000, max_output_tokens=8_000, max_steps=12)),
        guardrail=Guardrail(),
        permission_gate=PermissionGate(require_approval_from=PermissionLevel.HIGH),
        system_prompt=(
            "You are a helpful assistant with access to tools. Use get_current_weather and "
            "roll_dice freely. send_email is high-risk and requires human approval, but you "
            "should still call it when the user asks you to send an email."
        ),
    )

    prompt = (
        "Check the current weather in Tokyo, then send an email to ops@example.com "
        "with subject 'Weather Update' summarizing the conditions you found."
    )

    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {prompt}\n")
    print("-" * 78)

    request_id = None
    async for event in harness.run(prompt):
        print_event(event)
        if event.type == EventType.PERMISSION_REQUEST:
            request_id = event.data["request_id"]

    print("\n" + "-" * 78)
    print(f"Status after first turn: {harness.state.status.value}")

    if request_id:
        print("\nHuman operator approves the send_email call...")
        async for event in harness.resume_after_permission(request_id, approved=True, actor="demo-operator"):
            print_event(event)

    print("\n" + "-" * 78)
    print(f"Final status: {harness.state.status.value}")
    print(f"Budget snapshot: {harness.budget.snapshot()}")


if __name__ == "__main__":
    asyncio.run(main())
