"""Hooks + Agent-as-Tool (`__consult__`) demo for Sutra.

Uses the same deterministic, network-free `MockModelClient` pattern as
`examples/finance_workflow.py`, driving the same Contoso IT/Finance toolkit
(`sutra.toolkits.contoso_demo`), to demonstrate two features that extend the
harness beyond the 10 core architectural components:

  1. **Lifecycle Hooks** (`core/hooks.py`): a `HookRegistry` of named
     interception points (`pre_tool_call`, `post_tool_call`, `pre_model_call`,
     `on_handoff`, `on_done`) that let an operator register custom async
     callables for audit logging, veto logic, or metrics WITHOUT modifying
     harness code. This demo registers:
       - an audit-log hook on `pre_tool_call` / `post_tool_call` that prints
         every tool invocation and its result, and
       - a policy-veto hook on `pre_tool_call` that unconditionally blocks
         `transfer_funds` for this demo environment — showing that hooks can
         enforce policy *upstream* of, and independently from, whatever the
         `PermissionGate`'s risk tier would otherwise allow.

  2. **Agent-as-Tool (`__consult__`)**: a second pseudo-tool alongside the
     existing one-way `__handoff__`. Unlike `__handoff__`, which permanently
     transfers control (pushes the agent stack, changes
     `harness.state.active_agent_id`), `__consult__` lets the active agent
     ask a specialist subagent a bounded question and get the answer back as
     an ordinary tool result — control returns to the asking agent
     immediately, `active_agent_id` never changes. This demo has the root
     triage agent consult `finance_ops_agent` for a quick balance check
     without handing the conversation off, then prints
     `harness.state.active_agent_id` before and after the consult to make
     that "control never moves" property visible. A later step *does* use a
     real `__handoff__` (for an actual fund transfer, which needs the
     subagent to retain control across multiple tool calls) so the two
     pseudo-tools can be contrasted side by side.

IMPORTANT — this example will not run successfully against every checkout:
`core/hooks.py` (the `HookRegistry` / `HookContext` / `HookResult` module) and
the harness's `__consult__` support are being implemented by other engineers
in parallel isolated worktrees and may not be present yet, depending on which
commit of the repo you're running this against. The imports and constructor
call below (`AsynchronousHarnessLoop(..., hooks=hooks, ...)`) are written
against the documented, agreed-upon contract for those two features, so this
file is expected to become runnable once those workstreams land and are
merged. Until then, treat this as a design reference / smoke-test target
rather than something you can `python examples/hooks_and_consult_demo.py`
today.

Run with (once `core/hooks.py` and `__consult__` support exist):
    python examples/hooks_and_consult_demo.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from sutra.core.budget import Budget, BudgetConfig, ModelPricing
from sutra.core.guardrails import Guardrail
from sutra.core.harness import HANDOFF_TOOL_NAME, AsynchronousHarnessLoop
from sutra.core.hooks import HookContext, HookRegistry, HookResult
from sutra.core.permissions import PermissionGate, PermissionLevel
from sutra.models.mock_client import MockModelClient
from sutra.toolkits import contoso_demo
from sutra.tools.registry import Tool, ToolParameter

# `__consult__` is the agent-as-tool counterpart to `HANDOFF_TOOL_NAME`
# (`core/harness.py`'s `__handoff__`). Once `core/harness.py` exports a
# `CONSULT_TOOL_NAME` constant of its own, prefer importing that over
# hardcoding the literal here.
CONSULT_TOOL_NAME = "__consult__"


# ---------------------------------------------------------------------------
# Hooks: audit logging + a policy veto, registered independently of the
# harness/tool code itself.
# ---------------------------------------------------------------------------

async def audit_log_hook(ctx: HookContext) -> None:
    """Print every tool call and result. Returning None means "no veto, continue"."""
    if ctx.point == "pre_tool_call":
        print(f"[AUDIT] {ctx.point}: tool={ctx.tool_name} args={ctx.arguments}")
    else:
        print(f"[AUDIT] {ctx.point}: tool={ctx.tool_name} result={ctx.result}")


async def block_transfer_funds_hook(ctx: HookContext) -> Optional[HookResult]:
    """Policy veto, independent of the PermissionGate's risk-tier approval flow.

    `transfer_funds` is CRITICAL-risk, which normally means the
    PermissionGate pauses the run and waits for a human to approve it (see
    `examples/finance_workflow.py`). This hook demonstrates a *stronger*
    control: it vetoes the call outright, before it ever reaches the
    PermissionGate, so no approval prompt is even generated. A real
    deployment might use this for a hard kill-switch, a maintenance freeze,
    or a policy that certain tools are never allowed from certain hook
    contexts.
    """
    if ctx.tool_name == "transfer_funds":
        return HookResult(
            veto=True,
            reason="blocked by policy hook: transfer_funds is disabled in this demo environment",
        )
    return None  # every other tool call proceeds untouched


# ---------------------------------------------------------------------------
# A deterministic scripted "model" standing in for a real LLM provider, same
# approach as examples/finance_workflow.py. Calls are counted in the order
# the harness issues them, including the extra model call(s) that happen
# *inside* a bounded __consult__ sub-conversation.
# ---------------------------------------------------------------------------

def make_script():
    call_count = {"n": 0}

    def script(_messages):
        call_count["n"] += 1
        n = call_count["n"]

        # --- Step 1: root agent consults finance_ops_agent for a quick,
        # non-committal balance check. Control never leaves "root".
        if n == 1:
            return {
                "text": (
                    "Before deciding whether this needs a full handoff, let me quickly consult "
                    "the finance operations specialist about the source account balance."
                ),
                "tool_calls": [
                    {
                        "id": "call_consult_1",
                        "name": CONSULT_TOOL_NAME,
                        "input": {
                            "target_agent_id": "finance_ops_agent",
                            "question": "Does account ACC-1001 currently have at least $500 available?",
                            "context_payload": {"account_id": "ACC-1001", "threshold_usd": 500},
                        },
                    }
                ],
            }
        # Nested turn, inside the bounded __consult__ sub-conversation: the
        # consulted agent uses its own tool to answer the question.
        if n == 2:
            return {
                "text": "Checking the balance on ACC-1001 to answer the consult question.",
                "tool_calls": [
                    {"id": "call_balance_1", "name": "get_account_balance", "input": {"account_id": "ACC-1001"}}
                ],
            }
        # Nested turn: the consulted agent has its tool result and gives its
        # final answer, which the harness returns to root as the __consult__
        # tool's result. This ends the bounded sub-conversation.
        if n == 3:
            return {
                "text": (
                    "Yes — ACC-1001 currently holds $18,250.42, comfortably above the $500 "
                    "threshold."
                ),
                "tool_calls": [],
            }
        # Back to root, which never left control: it now answers the user
        # directly using the consulted answer, still as "root".
        if n == 4:
            return {
                "text": (
                    "Good news: the source account has plenty of headroom for a $500 rush tip "
                    "($18,250.42 available). Since this was just a balance check, no handoff was "
                    "needed."
                ),
                "tool_calls": [],
            }
        # --- Step 2: a real fund transfer is requested. This time root does
        # a genuine one-way __handoff__ (not a consult), because the
        # transfer needs the subagent to retain control across the tool
        # call and its (blocked) aftermath.
        if n == 5:
            return {
                "text": "This is now an actual fund transfer request, handing off to finance operations.",
                "tool_calls": [
                    {
                        "id": "call_handoff_1",
                        "name": HANDOFF_TOOL_NAME,
                        "input": {
                            "target_agent_id": "finance_ops_agent",
                            "reason": "Request requires executing a real fund transfer.",
                            "context_payload": {
                                "from_account": "ACC-1001",
                                "to_account": "ACC-2002",
                                "amount_usd": 500,
                                "invoice": "INV-99001",
                            },
                        },
                    }
                ],
            }
        if n == 6:
            return {
                "text": "Initiating the $500 rush tip transfer for invoice INV-99001.",
                "tool_calls": [
                    {
                        "id": "call_transfer_1",
                        "name": "transfer_funds",
                        "input": {
                            "from_account": "ACC-1001",
                            "to_account": "ACC-2002",
                            "amount_usd": 500,
                            "memo": "INV-99001",
                        },
                    }
                ],
            }
        # The transfer_funds call above was vetoed by block_transfer_funds_hook
        # before it ever executed or reached the PermissionGate; the model
        # sees that as a failed tool result and reports it to the user.
        return {
            "text": (
                "I wasn't able to complete that transfer — it was blocked by a policy hook in "
                "this environment. Escalating to a human operator to run it manually."
            ),
            "tool_calls": [],
        }

    return script


def print_event(event) -> None:
    print(event.to_sse().rstrip())


async def main() -> None:
    tool_registry = contoso_demo.build_tool_registry()
    subagent_registry = contoso_demo.build_subagent_registry()

    # `__consult__` is registered the same way `__handoff__` is in
    # contoso_demo.build_tool_registry(): as a schema-only pseudo-tool whose
    # actual execution is intercepted by the harness loop, never by the
    # ToolRegistry's normal dispatch path.
    async def _consult_placeholder(**_kwargs: object) -> None:
        raise RuntimeError("The __consult__ pseudo-tool must be intercepted by the harness loop, not executed directly.")

    tool_registry.register_tool(
        Tool(
            name=CONSULT_TOOL_NAME,
            description=(
                "Ask a specialist subagent a bounded question and get the answer back as a tool "
                "result, WITHOUT transferring control. Use this instead of __handoff__ when you "
                "just need information from a specialist, not a permanent transfer."
            ),
            parameters=[
                ToolParameter("target_agent_id", "string", "The registered subagent id to consult."),
                ToolParameter("question", "string", "The specific question to ask the specialist."),
                ToolParameter(
                    "context_payload", "object", "Structured context to brief the specialist with.", required=False
                ),
            ],
            handler=_consult_placeholder,
            permission_level=PermissionLevel.LOW,
        )
    )

    hooks = HookRegistry()
    hooks.register("pre_tool_call", audit_log_hook)
    hooks.register("post_tool_call", audit_log_hook)
    hooks.register("pre_tool_call", block_transfer_funds_hook)

    budget = Budget(
        BudgetConfig(max_usd=1.00, max_input_tokens=20_000, max_output_tokens=500, max_steps=20, warn_ratio=0.8),
        pricing=ModelPricing(input_per_million=3.0, output_per_million=15.0),
    )
    guardrail = Guardrail()
    permission_gate = PermissionGate(require_approval_from=PermissionLevel.HIGH)

    harness = AsynchronousHarnessLoop(
        model_client=MockModelClient(script=make_script()),
        tool_registry=tool_registry,
        budget=budget,
        guardrail=guardrail,
        subagent_registry=subagent_registry,
        permission_gate=permission_gate,
        hooks=hooks,
        system_prompt=contoso_demo.ROOT_SYSTEM_PROMPT,
    )

    print("=" * 78)
    print("STEP 1 - agent-as-tool: root consults finance_ops_agent for a quick")
    print("          balance check, WITHOUT handing off control")
    print("=" * 78)
    print(f"active_agent_id BEFORE consult: {harness.state.active_agent_id!r}")

    quick_check = (
        "Before I decide whether to escalate this, can you quickly check whether account "
        "ACC-1001 has at least $500 available for a rush tip?"
    )
    async for event in harness.run(quick_check):
        print_event(event)

    print()
    print(f"active_agent_id AFTER consult:  {harness.state.active_agent_id!r}  (unchanged — no handoff occurred)")

    print()
    print("=" * 78)
    print("STEP 2 - real fund transfer: root does a genuine __handoff__ to")
    print("          finance_ops_agent, which attempts transfer_funds; the")
    print("          pre_tool_call policy hook vetoes it before the")
    print("          PermissionGate ever sees it")
    print("=" * 78)
    print(f"active_agent_id BEFORE handoff: {harness.state.active_agent_id!r}")

    transfer_request = (
        "Great — go ahead and transfer $500 to vendor account ACC-2002 as a rush tip for "
        "invoice INV-99001."
    )
    async for event in harness.run(transfer_request):
        print_event(event)

    print()
    print(f"active_agent_id AFTER handoff:  {harness.state.active_agent_id!r}  (changed — a real handoff occurred)")

    print()
    print(f"Final harness status: {harness.state.status.value}")
    print(f"Final budget snapshot: {json.dumps(harness.budget.snapshot())}")
    print(f"Final ACC-1001 balance (should be UNCHANGED, transfer was vetoed): "
          f"${contoso_demo.ACCOUNTS['ACC-1001']['balance_usd']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
