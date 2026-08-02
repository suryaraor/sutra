from typing import Optional

from sutra.agents.subagent import SubagentRegistry, SubagentProfile
from sutra.core.budget import Budget, BudgetConfig
from sutra.core.harness import AsynchronousHarnessLoop
from sutra.core.hooks import (
    ON_DONE,
    ON_HANDOFF,
    POST_TOOL_CALL,
    PRE_TOOL_CALL,
    HookContext,
    HookRegistry,
    HookResult,
)
from sutra.core.permissions import PermissionLevel
from sutra.core.state import HarnessStatus
from sutra.models.mock_client import MockModelClient
from sutra.streaming.sse import EventType
from sutra.tools.registry import Tool, ToolParameter, ToolRegistry


def simple_script(_messages):
    return {"text": "Hello! How can I help you today?", "tool_calls": []}


# -- HookRegistry unit behavior -----------------------------------------------


async def test_hook_registry_is_noop_when_nothing_registered():
    registry = HookRegistry()
    result = await registry.run(PRE_TOOL_CALL, HookContext(point=PRE_TOOL_CALL, state=None))
    assert result is None


async def test_hook_registry_short_circuits_on_first_veto():
    calls = []

    async def first(_ctx):
        calls.append("first")
        return HookResult(veto=True, veto_reason="stop right there")

    async def second(_ctx):
        calls.append("second")
        return None

    registry = HookRegistry()
    registry.register(PRE_TOOL_CALL, first)
    registry.register(PRE_TOOL_CALL, second)

    result = await registry.run(PRE_TOOL_CALL, HookContext(point=PRE_TOOL_CALL, state=None))

    assert calls == ["first"]  # second hook never ran
    assert result is not None
    assert result.veto is True
    assert result.veto_reason == "stop right there"


# -- pre_tool_call --------------------------------------------------------------


async def test_pre_tool_call_veto_blocks_the_tool_from_executing():
    executed = []
    tool_registry = ToolRegistry()

    async def do_thing(target: str) -> dict:
        executed.append(target)
        return {"status": "done"}

    tool_registry.register_tool(
        Tool(
            name="do_thing",
            description="does a thing",
            parameters=[ToolParameter("target", "string", "target")],
            handler=do_thing,
            permission_level=PermissionLevel.LOW,
        )
    )

    def script(_messages):
        return {
            "text": "Doing the thing.",
            "tool_calls": [{"id": "call_1", "name": "do_thing", "input": {"target": "prod-db"}}],
        }

    async def veto_prod_db(ctx: HookContext) -> Optional[HookResult]:
        if ctx.tool_name == "do_thing" and ctx.arguments and ctx.arguments.get("target") == "prod-db":
            return HookResult(veto=True, veto_reason="prod-db is off-limits.")
        return None

    hooks = HookRegistry()
    hooks.register(PRE_TOOL_CALL, veto_prod_db)

    harness = AsynchronousHarnessLoop(
        model_client=MockModelClient(script=script),
        tool_registry=tool_registry,
        budget=Budget(BudgetConfig()),
        hooks=hooks,
    )

    events = [event async for event in harness.run("Please do the thing on prod-db.")]

    assert executed == []  # the tool handler never ran
    assert harness.state.status == HarnessStatus.FAILED
    veto_events = [e for e in events if e.type == EventType.HOOK_VETO]
    assert len(veto_events) == 1
    assert veto_events[0].data["tool_name"] == "do_thing"
    assert veto_events[0].data["reason"] == "prod-db is off-limits."


async def test_pre_tool_call_modified_arguments_reach_the_tool_handler():
    received = []
    tool_registry = ToolRegistry()

    async def echo(value: str) -> dict:
        received.append(value)
        return {"received": value}

    tool_registry.register_tool(
        Tool(
            name="echo",
            description="echoes a value",
            parameters=[ToolParameter("value", "string", "value")],
            handler=echo,
            permission_level=PermissionLevel.LOW,
        )
    )

    calls = {"n": 0}

    def script(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "text": "Echoing.",
                "tool_calls": [{"id": "call_1", "name": "echo", "input": {"value": "original"}}],
            }
        return {"text": "Done echoing.", "tool_calls": []}

    async def rewrite_value(ctx: HookContext) -> Optional[HookResult]:
        if ctx.tool_name == "echo":
            return HookResult(modified_arguments={"value": "modified-by-hook"})
        return None

    hooks = HookRegistry()
    hooks.register(PRE_TOOL_CALL, rewrite_value)

    harness = AsynchronousHarnessLoop(
        model_client=MockModelClient(script=script),
        tool_registry=tool_registry,
        budget=Budget(BudgetConfig()),
        hooks=hooks,
    )

    events = [event async for event in harness.run("Echo something.")]

    assert received == ["modified-by-hook"]  # not "original"
    assert harness.state.status == HarnessStatus.COMPLETED
    tool_result_events = [e for e in events if e.type == EventType.TOOL_RESULT]
    assert any("modified-by-hook" in e.data["result"] for e in tool_result_events)


# -- post_tool_call ---------------------------------------------------------------


async def test_post_tool_call_hook_sees_tool_name_and_result():
    seen = []
    tool_registry = ToolRegistry()

    async def noop_tool() -> dict:
        return {"ok": True}

    tool_registry.register_tool(
        Tool(name="noop_tool", description="does nothing", parameters=[], handler=noop_tool, permission_level=PermissionLevel.LOW)
    )

    calls = {"n": 0}

    def script(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "text": "Running noop.",
                "tool_calls": [{"id": "call_1", "name": "noop_tool", "input": {}}],
            }
        return {"text": "Done.", "tool_calls": []}

    async def record_result(ctx: HookContext) -> None:
        seen.append((ctx.tool_name, ctx.result))

    hooks = HookRegistry()
    hooks.register(POST_TOOL_CALL, record_result)

    harness = AsynchronousHarnessLoop(
        model_client=MockModelClient(script=script),
        tool_registry=tool_registry,
        budget=Budget(BudgetConfig()),
        hooks=hooks,
    )

    events = [event async for event in harness.run("Run noop.")]

    assert seen == [("noop_tool", {"ok": True})]
    assert harness.state.status == HarnessStatus.COMPLETED
    assert any(e.type == EventType.TOOL_RESULT for e in events)


# -- on_handoff ------------------------------------------------------------------


async def test_on_handoff_veto_blocks_the_handoff():
    subagent_registry = SubagentRegistry()
    subagent_registry.register(
        SubagentProfile(agent_id="billing_agent", name="Billing Agent", system_prompt="You handle billing.")
    )

    def script(_messages):
        return {
            "text": "Handing off to billing.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "__handoff__",
                    "input": {"target_agent_id": "billing_agent", "reason": "billing question"},
                }
            ],
        }

    async def veto_all_handoffs(ctx: HookContext) -> Optional[HookResult]:
        return HookResult(veto=True, veto_reason=f"handoffs to {ctx.target_agent_id} are disabled right now.")

    hooks = HookRegistry()
    hooks.register(ON_HANDOFF, veto_all_handoffs)

    harness = AsynchronousHarnessLoop(
        model_client=MockModelClient(script=script),
        tool_registry=ToolRegistry(),
        budget=Budget(BudgetConfig()),
        subagent_registry=subagent_registry,
        hooks=hooks,
    )

    events = [event async for event in harness.run("I have a billing question.")]

    assert harness.state.agent_stack == ["root"]  # unchanged
    assert harness.state.active_agent_id == "root"
    assert harness.state.status == HarnessStatus.FAILED
    error_events = [e for e in events if e.type == EventType.ERROR]
    assert len(error_events) == 1
    assert error_events[0].data["phase"] == "handoff"
    assert "disabled" in error_events[0].data["message"]


# -- on_done ------------------------------------------------------------------------


async def test_on_done_fires_exactly_once_for_a_simple_completed_run():
    counter = {"n": 0}

    async def count_done(_ctx: HookContext) -> None:
        counter["n"] += 1

    hooks = HookRegistry()
    hooks.register(ON_DONE, count_done)

    harness = AsynchronousHarnessLoop(
        model_client=MockModelClient(script=simple_script),
        tool_registry=ToolRegistry(),
        budget=Budget(BudgetConfig()),
        hooks=hooks,
    )

    events = [event async for event in harness.run("Hi there")]

    assert harness.state.status == HarnessStatus.COMPLETED
    assert events[-1].type == EventType.DONE
    assert counter["n"] == 1


async def test_on_done_fires_exactly_once_across_permission_gate_pause_and_approved_resume():
    counter = {"n": 0}

    async def count_done(_ctx: HookContext) -> None:
        counter["n"] += 1

    hooks = HookRegistry()
    hooks.register(ON_DONE, count_done)

    tool_registry = ToolRegistry()

    async def risky_action(target: str) -> dict:
        return {"status": "done", "target": target}

    tool_registry.register_tool(
        Tool(
            name="risky_action",
            description="A high-risk action requiring approval.",
            parameters=[ToolParameter("target", "string", "target of the action")],
            handler=risky_action,
            permission_level=PermissionLevel.CRITICAL,
        )
    )

    calls = {"n": 0}

    def script(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "text": "Running the risky action.",
                "tool_calls": [{"id": "call_1", "name": "risky_action", "input": {"target": "prod-db"}}],
            }
        return {"text": "All done.", "tool_calls": []}

    harness = AsynchronousHarnessLoop(
        model_client=MockModelClient(script=script),
        tool_registry=tool_registry,
        budget=Budget(BudgetConfig()),
        hooks=hooks,
    )

    events = [event async for event in harness.run("Please run the risky action on prod-db.")]
    assert harness.state.status == HarnessStatus.PAUSED_FOR_PERMISSION
    # Paused, not done yet: on_done must NOT have fired during the pause.
    assert counter["n"] == 0

    request_id = next(e.data["request_id"] for e in events if e.type == EventType.PERMISSION_REQUEST)

    resume_events = [
        event async for event in harness.resume_after_permission(request_id, approved=True, actor="tester")
    ]

    assert any(e.type == EventType.TOOL_RESULT for e in resume_events)
    assert harness.state.status == HarnessStatus.COMPLETED
    # Exactly once total across the whole pause -> resume sequence: not zero
    # (it must fire once the turn truly concludes) and not twice (the nested
    # self.run("") inside resume_after_permission's approved path, and
    # resume_after_permission's own finally, must not both fire it).
    assert counter["n"] == 1


async def test_on_done_fires_exactly_once_across_permission_gate_pause_and_denied_resume():
    counter = {"n": 0}

    async def count_done(_ctx: HookContext) -> None:
        counter["n"] += 1

    hooks = HookRegistry()
    hooks.register(ON_DONE, count_done)

    tool_registry = ToolRegistry()

    async def risky_action(target: str) -> dict:
        return {"status": "done", "target": target}

    tool_registry.register_tool(
        Tool(
            name="risky_action",
            description="A high-risk action requiring approval.",
            parameters=[ToolParameter("target", "string", "target of the action")],
            handler=risky_action,
            permission_level=PermissionLevel.CRITICAL,
        )
    )

    def script(_messages):
        return {
            "text": "Running the risky action.",
            "tool_calls": [{"id": "call_1", "name": "risky_action", "input": {"target": "prod-db"}}],
        }

    harness = AsynchronousHarnessLoop(
        model_client=MockModelClient(script=script),
        tool_registry=tool_registry,
        budget=Budget(BudgetConfig()),
        hooks=hooks,
    )

    events = [event async for event in harness.run("Please run the risky action on prod-db.")]
    assert harness.state.status == HarnessStatus.PAUSED_FOR_PERMISSION
    assert counter["n"] == 0

    request_id = next(e.data["request_id"] for e in events if e.type == EventType.PERMISSION_REQUEST)

    resume_events = [
        event async for event in harness.resume_after_permission(request_id, approved=False, actor="tester")
    ]

    assert harness.state.status == HarnessStatus.FAILED
    assert any(e.type == EventType.DONE for e in resume_events)
    assert counter["n"] == 1


# -- regression: hooks=None must behave completely unchanged --------------------------


async def test_harness_without_hooks_behaves_unchanged_for_basic_turn():
    harness = AsynchronousHarnessLoop(
        model_client=MockModelClient(script=simple_script),
        tool_registry=ToolRegistry(),
        budget=Budget(BudgetConfig()),
        system_prompt="You are a test agent.",
    )
    assert harness.hooks is None

    events = [event async for event in harness.run("Hi there")]

    assert any(e.type == EventType.TOKEN for e in events)
    assert events[-1].type == EventType.DONE
    assert harness.state.status == HarnessStatus.COMPLETED


async def test_harness_without_hooks_behaves_unchanged_for_tool_and_permission_gate():
    tool_registry = ToolRegistry()

    async def risky_action(target: str) -> dict:
        return {"status": "done", "target": target}

    tool_registry.register_tool(
        Tool(
            name="risky_action",
            description="A high-risk action requiring approval.",
            parameters=[ToolParameter("target", "string", "target of the action")],
            handler=risky_action,
            permission_level=PermissionLevel.CRITICAL,
        )
    )

    calls = {"n": 0}

    def script(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "text": "Running the risky action.",
                "tool_calls": [{"id": "call_1", "name": "risky_action", "input": {"target": "prod-db"}}],
            }
        return {"text": "All done.", "tool_calls": []}

    harness = AsynchronousHarnessLoop(
        model_client=MockModelClient(script=script),
        tool_registry=tool_registry,
        budget=Budget(BudgetConfig()),
    )
    assert harness.hooks is None

    events = [event async for event in harness.run("Please run the risky action on prod-db.")]
    assert harness.state.status == HarnessStatus.PAUSED_FOR_PERMISSION
    request_id = next(e.data["request_id"] for e in events if e.type == EventType.PERMISSION_REQUEST)

    resume_events = [
        event async for event in harness.resume_after_permission(request_id, approved=True, actor="tester")
    ]
    assert any(e.type == EventType.TOOL_RESULT for e in resume_events)
    assert harness.state.status == HarnessStatus.COMPLETED
