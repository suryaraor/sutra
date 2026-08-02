"""Live integration demo: Sutra harness driven by a real local model via Ollama.

Unlike `examples/finance_workflow.py` (which uses the deterministic,
network-free `MockModelClient` so it can run anywhere without a GPU), this
script talks to an actual running Ollama server and a real downloaded model
(`gpt-oss:20b` by default). Tool-call behavior here is genuinely decided by
the model, not scripted — this is what proves the `OllamaModelClient` wiring
is real, not a stand-in. Tools are the small weather/dice/email toolkit in
`sutra.toolkits.simple_demo`.

For a richer multi-subagent version of this same idea, see
`examples/live_workflow.py`, or run it interactively via the `sutra` CLI
(`sutra` for chat, `sutra run weather-email` for this exact scenario).

Prerequisites:
    - Ollama installed and running (`ollama serve`, or the desktop app).
    - The model pulled: `ollama pull gpt-oss:20b`.

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
from sutra.toolkits import simple_demo

MODEL_NAME = "gpt-oss:20b"


def print_event(event) -> None:
    if event.type == EventType.TOKEN:
        print(event.data["text"], end="", flush=True)
        return
    print(f"\n[{event.type.value}] {event.data}")


async def main() -> None:
    harness = AsynchronousHarnessLoop(
        model_client=OllamaModelClient(model=MODEL_NAME),
        tool_registry=simple_demo.build_tool_registry(),
        budget=Budget(BudgetConfig(max_usd=1.0, max_input_tokens=50_000, max_output_tokens=8_000, max_steps=12)),
        guardrail=Guardrail(),
        permission_gate=PermissionGate(require_approval_from=PermissionLevel.HIGH),
        system_prompt=simple_demo.SYSTEM_PROMPT,
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
