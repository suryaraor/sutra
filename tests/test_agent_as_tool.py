from sutra.agents.subagent import SubagentProfile, SubagentRegistry
from sutra.core.budget import Budget, BudgetConfig
from sutra.core.harness import AsynchronousHarnessLoop
from sutra.core.permissions import PermissionLevel
from sutra.core.state import HarnessStatus, Role
from sutra.models.mock_client import MockModelClient
from sutra.streaming.sse import EventType
from sutra.tools.registry import Tool, ToolParameter, ToolRegistry


def _specialist_registry(tool_name: str) -> SubagentRegistry:
    registry = SubagentRegistry()
    registry.register(
        SubagentProfile(
            agent_id="specialist",
            name="Specialist",
            system_prompt="You are a narrow domain specialist.",
            allowed_tools={tool_name},
        )
    )
    return registry


async def test_consult_returns_answer_without_changing_active_agent():
    tool_registry = ToolRegistry()

    async def lookup_fact(topic: str) -> dict:
        return {"topic": topic, "fact": "The answer is 42."}

    tool_registry.register_tool(
        Tool(
            name="lookup_fact",
            description="Look up a fact.",
            parameters=[ToolParameter("topic", "string", "topic to look up")],
            handler=lookup_fact,
            permission_level=PermissionLevel.LOW,
            allowed_agents=["specialist"],
        )
    )

    calls = {"n": 0}

    def script(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "text": "Let me consult the specialist.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "__consult__",
                        "input": {"target_agent_id": "specialist", "question": "What is the answer?"},
                    }
                ],
            }
        if calls["n"] == 2:
            return {
                "text": "",
                "tool_calls": [{"id": "n1", "name": "lookup_fact", "input": {"topic": "the answer"}}],
            }
        if calls["n"] == 3:
            return {"text": "The answer is 42.", "tool_calls": []}
        return {"text": "Root: per the specialist, the answer is 42.", "tool_calls": []}

    harness = AsynchronousHarnessLoop(
        model_client=MockModelClient(script=script),
        tool_registry=tool_registry,
        budget=Budget(BudgetConfig()),
        subagent_registry=_specialist_registry("lookup_fact"),
        system_prompt="You are root.",
    )

    events = [event async for event in harness.run("Ask the specialist what the answer is.")]

    # Control never left root.
    assert harness.state.active_agent_id == "root"
    assert harness.state.agent_stack == ["root"]
    assert harness.state.status == HarnessStatus.COMPLETED

    assert any(e.type == EventType.CONSULT_START for e in events)
    consult_end_events = [e for e in events if e.type == EventType.CONSULT_END]
    assert len(consult_end_events) == 1
    assert "42" in consult_end_events[0].data["answer"]

    tool_result_events = [e for e in events if e.type == EventType.TOOL_RESULT]
    assert any("42" in e.data["result"] for e in tool_result_events)

    # Only root's own messages + exactly one consult tool-result are top-level;
    # the nested specialist's own tool call/result never leaked in.
    tool_messages = [m for m in harness.state.messages if m.role == Role.TOOL]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "__consult__"
    assert all(m.name != "lookup_fact" for m in harness.state.messages)
    assert len(harness.state.messages) == 4  # user, assistant(consult call), tool(consult result), assistant(final)

    # Budget reflects all 4 nested+outer model calls (1 root + 2 nested + 1 root).
    assert harness.budget.usage.steps == 4
    assert harness.budget.snapshot()["steps"] == 4


async def test_consult_unknown_target_agent_returns_error_without_crashing():
    calls = {"n": 0}

    def script(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "text": "Consulting.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "__consult__",
                        "input": {"target_agent_id": "ghost_agent", "question": "Are you there?"},
                    }
                ],
            }
        return {"text": "The specialist isn't registered, handling it myself.", "tool_calls": []}

    harness = AsynchronousHarnessLoop(
        model_client=MockModelClient(script=script),
        tool_registry=ToolRegistry(),
        budget=Budget(BudgetConfig()),
        system_prompt="You are root.",
    )

    events = [event async for event in harness.run("Consult the ghost agent.")]

    # A failed consult does not fail the run -- it comes back as a tool result.
    assert harness.state.status == HarnessStatus.COMPLETED
    assert harness.state.active_agent_id == "root"

    error_events = [e for e in events if e.type == EventType.ERROR]
    assert any(e.data.get("phase") == "consult" for e in error_events)

    consult_end = next(e for e in events if e.type == EventType.CONSULT_END)
    assert "error" in consult_end.data
    assert "ghost_agent" in consult_end.data["error"]

    tool_messages = [m for m in harness.state.messages if m.role == Role.TOOL]
    assert len(tool_messages) == 1
    assert "ghost_agent" in tool_messages[0].content

    # Harness is still usable afterward: a normal follow-up turn still works.
    calls["n"] = 0

    def follow_up_script(_messages):
        return {"text": "All good.", "tool_calls": []}

    harness.model_client = MockModelClient(script=follow_up_script)
    more_events = [event async for event in harness.run("Thanks.")]
    assert harness.state.status == HarnessStatus.COMPLETED
    assert any(e.type == EventType.DONE for e in more_events)


async def test_consult_budget_usage_reflects_nested_model_calls():
    tool_registry = ToolRegistry()

    async def echo(value: str) -> dict:
        return {"value": value}

    tool_registry.register_tool(
        Tool(
            name="echo",
            description="Echo a value.",
            parameters=[ToolParameter("value", "string", "value to echo")],
            handler=echo,
            permission_level=PermissionLevel.LOW,
            allowed_agents=["specialist"],
        )
    )

    calls = {"n": 0}

    def script(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "text": "Consulting.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "__consult__",
                        "input": {"target_agent_id": "specialist", "question": "Echo hi."},
                    }
                ],
            }
        if calls["n"] == 2:
            return {"text": "", "tool_calls": [{"id": "n1", "name": "echo", "input": {"value": "hi"}}]}
        if calls["n"] == 3:
            return {"text": "hi", "tool_calls": []}
        return {"text": "Done.", "tool_calls": []}

    harness = AsynchronousHarnessLoop(
        model_client=MockModelClient(script=script),
        tool_registry=tool_registry,
        budget=Budget(BudgetConfig()),
        subagent_registry=_specialist_registry("echo"),
        system_prompt="You are root.",
    )

    steps_before = harness.budget.usage.steps
    tokens_before = harness.budget.usage.input_tokens + harness.budget.usage.output_tokens

    [event async for event in harness.run("Ask the specialist to echo hi.")]

    steps_after = harness.budget.usage.steps
    tokens_after = harness.budget.usage.input_tokens + harness.budget.usage.output_tokens

    # 1 root call + 2 nested consult calls + 1 root follow-up call = 4 steps.
    assert steps_after - steps_before == 4
    assert tokens_after > tokens_before


async def test_consult_with_gated_tool_fails_gracefully_instead_of_hanging():
    tool_registry = ToolRegistry()

    async def dangerous_action(target: str) -> dict:
        return {"status": "done", "target": target}

    tool_registry.register_tool(
        Tool(
            name="dangerous_action",
            description="A critical-risk action.",
            parameters=[ToolParameter("target", "string", "target of the action")],
            handler=dangerous_action,
            permission_level=PermissionLevel.CRITICAL,
            allowed_agents=["specialist"],
        )
    )

    calls = {"n": 0}

    def script(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "text": "Consulting the specialist.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "__consult__",
                        "input": {"target_agent_id": "specialist", "question": "Handle this for me."},
                    }
                ],
            }
        if calls["n"] == 2:
            return {
                "text": "",
                "tool_calls": [{"id": "n1", "name": "dangerous_action", "input": {"target": "prod"}}],
            }
        return {"text": "Cannot complete via consult; would need a real handoff.", "tool_calls": []}

    harness = AsynchronousHarnessLoop(
        model_client=MockModelClient(script=script),
        tool_registry=tool_registry,
        budget=Budget(BudgetConfig()),
        subagent_registry=_specialist_registry("dangerous_action"),
        system_prompt="You are root.",
    )

    events = [event async for event in harness.run("Please handle this via the specialist.")]

    # No permission gate was opened, no hang, no crash -- the run simply finishes.
    assert harness.state.status == HarnessStatus.COMPLETED
    assert harness.state.active_agent_id == "root"
    assert harness.state.pending_permission is None
    assert not any(e.type == EventType.PERMISSION_REQUEST for e in events)

    consult_end = next(e for e in events if e.type == EventType.CONSULT_END)
    assert "error" in consult_end.data
    assert "not supported inside __consult__" in consult_end.data["error"]
    assert "__handoff__" in consult_end.data["error"]

    tool_messages = [m for m in harness.state.messages if m.role == Role.TOOL]
    assert len(tool_messages) == 1
    assert "__handoff__" in tool_messages[0].content
