from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from sutra.agents.subagent import SubagentRegistry
from sutra.core.budget import Budget
from sutra.core.compaction import ContextCompactor
from sutra.core.exceptions import (
    BudgetExceededError,
    GuardrailViolation,
    HandoffError,
    ModelInvocationError,
    ToolExecutionError,
)
from sutra.core.guardrails import Guardrail
from sutra.core.handoff import HandoffRequest, HandoffRouter
from sutra.core.hooks import (
    ON_DONE,
    ON_HANDOFF,
    POST_TOOL_CALL,
    PRE_MODEL_CALL,
    PRE_TOOL_CALL,
    HookContext,
    HookRegistry,
)
from sutra.core.permissions import PermissionGate
from sutra.core.state import HarnessState, HarnessStatus, Message, PendingPermission, Role
from sutra.memory.manager import MemoryManager
from sutra.models.base import ModelClient
from sutra.streaming.sse import EventType, SSEEvent
from sutra.tools.registry import ToolRegistry

HANDOFF_TOOL_NAME = "__handoff__"


class AsynchronousHarnessLoop:
    """The primary runtime environment.

    Orchestrates a single agent task end-to-end: guardrail-checks the
    user's input, streams model output as SSE events, dispatches tool
    calls (routing high-risk ones through the `PermissionGate`), applies
    context compaction when the token window grows too large, honors
    subagent handoffs, and enforces the `Budget` on every step. All state
    lives on `self.state` (a `HarnessState`), never in transient locals, so
    a run can be paused, checkpointed, and resumed.
    """

    def __init__(
        self,
        *,
        model_client: ModelClient,
        tool_registry: ToolRegistry,
        budget: Budget,
        guardrail: Optional[Guardrail] = None,
        compactor: Optional[ContextCompactor] = None,
        subagent_registry: Optional[SubagentRegistry] = None,
        permission_gate: Optional[PermissionGate] = None,
        memory: Optional[MemoryManager] = None,
        hooks: Optional[HookRegistry] = None,
        user_id: str = "anonymous",
        system_prompt: str = "You are a helpful, safety-conscious autonomous agent.",
        state: Optional[HarnessState] = None,
        max_tool_hops_per_turn: int = 8,
    ) -> None:
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.budget = budget
        self.guardrail = guardrail or Guardrail()
        self.compactor = compactor or ContextCompactor()
        self.subagent_registry = subagent_registry or SubagentRegistry()
        self.permission_gate = permission_gate or PermissionGate()
        self.handoff_router = HandoffRouter(self.subagent_registry)
        self.memory = memory
        # Optional: named lifecycle hooks (pre_tool_call, post_tool_call,
        # pre_model_call, on_handoff, on_done). Every call site guards with
        # `if self.hooks is not None`, so leaving this `None` (the default)
        # is a strict no-op — behavior is identical to a harness built before
        # hooks existed. See specs/15-hooks.md.
        self.hooks = hooks
        self.user_id = user_id
        self.max_tool_hops_per_turn = max_tool_hops_per_turn

        self.state = state or HarnessState(system_prompt=system_prompt)
        if not self.state.system_prompt:
            self.state.system_prompt = system_prompt

    # -- public API -----------------------------------------------------------

    async def run(self, user_input: str) -> AsyncIterator[SSEEvent]:
        """Drive one full user turn to completion, yielding SSE events as it goes.

        The whole body runs inside a `try/finally` so memory gets recorded
        exactly once per call regardless of which of the several exit paths
        below fires — guardrail block, model error, budget exceeded, tool
        error, permission pause, or normal completion.
        """
        try:
            self.state.status = HarnessStatus.RUNNING
            yield SSEEvent(
                EventType.SESSION_START,
                {"session_id": self.state.session_id, "active_agent": self.state.active_agent_id},
            )

            # 1. Guardrail the raw input before it touches state or the model.
            try:
                clean_input = self.guardrail.sanitize_input(user_input)
            except GuardrailViolation as violation:
                self.state.record_step("guardrail_block", rule=violation.rule_name, direction="input")
                self.state.status = HarnessStatus.FAILED
                yield SSEEvent(
                    EventType.GUARDRAIL_BLOCK,
                    {"rule": violation.rule_name, "direction": "input", "message": str(violation)},
                )
                yield SSEEvent(EventType.DONE, {"status": self.state.status.value})
                return

            if clean_input:
                self.state.append_message(Message(role=Role.USER, content=clean_input))

            hops = 0
            try:
                while True:
                    hops += 1
                    if hops > self.max_tool_hops_per_turn:
                        raise ToolExecutionError(
                            f"Exceeded max_tool_hops_per_turn={self.max_tool_hops_per_turn} without a final answer.",
                            tool_name="<loop_guard>",
                            original_error="tool_hop_limit",
                        )

                    # 2. Budget: reserve a step before spending anything on it.
                    await self.budget.check_step()
                    self.state.record_step("model_call", agent=self.state.active_agent_id, hop=hops)

                    # 3. Context compaction, applied in place on state.messages.
                    if self.compactor.needs_compaction(self.state.messages):
                        compacted = await self.compactor.compact(self.state.messages)
                        self.state.messages = compacted
                        self.state.compaction_count += 1
                        yield SSEEvent(
                            EventType.COMPACTION,
                            {"compaction_count": self.state.compaction_count, "resulting_messages": len(compacted)},
                        )

                    for dim, ratio in self.budget.warnings().items():
                        yield SSEEvent(
                            EventType.BUDGET_WARNING,
                            {"dimension": dim, "ratio_used": round(ratio, 3), "snapshot": self.budget.snapshot()},
                        )

                    # 4. Call the model, streaming tokens straight out as SSE.
                    agent_id = self.state.active_agent_id
                    allowed_tools = self.subagent_registry.allowed_tools_for(agent_id)
                    tool_schemas = self.tool_registry.schemas_for(agent_id, allowed_tools)
                    system_prompt = self._effective_system_prompt(agent_id)
                    wire_messages = self._messages_for_wire()

                    # 3.5. pre_model_call hook: fires once per model-call
                    # iteration of this loop, right before the streaming
                    # request goes out. Side-effect only — a returned
                    # HookResult (veto or otherwise) is intentionally ignored
                    # here; see specs/15-hooks.md.
                    if self.hooks is not None:
                        await self.hooks.run(PRE_MODEL_CALL, HookContext(point=PRE_MODEL_CALL, state=self.state))

                    assistant_text = ""
                    tool_calls: List[Dict[str, Any]] = []
                    input_tokens = output_tokens = 0

                    try:
                        async for chunk in self.model_client.stream(
                            system_prompt=system_prompt, messages=wire_messages, tools=tool_schemas
                        ):
                            if chunk.delta_text:
                                assistant_text += chunk.delta_text
                                yield SSEEvent(EventType.TOKEN, {"text": chunk.delta_text, "agent": agent_id})
                            if chunk.tool_call_delta and "finalized" in chunk.tool_call_delta:
                                tool_calls = chunk.tool_call_delta["finalized"]
                            if chunk.finished and chunk.usage:
                                input_tokens = chunk.usage.get("input_tokens", 0)
                                output_tokens = chunk.usage.get("output_tokens", 0)
                    except ModelInvocationError as exc:
                        self.state.status = HarnessStatus.FAILED
                        yield SSEEvent(EventType.ERROR, {"phase": "model_call", "message": str(exc)})
                        yield SSEEvent(EventType.DONE, {"status": self.state.status.value})
                        return

                    await self.budget.record_usage(input_tokens=input_tokens, output_tokens=output_tokens)

                    assistant_message = Message(
                        role=Role.ASSISTANT,
                        content=assistant_text,
                        tool_calls=tool_calls or None,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                    self.state.append_message(assistant_message)
                    yield SSEEvent(
                        EventType.MESSAGE_COMPLETE,
                        {"agent": agent_id, "text": assistant_text, "has_tool_calls": bool(tool_calls)},
                    )

                    if not tool_calls:
                        self.state.status = HarnessStatus.COMPLETED
                        break

                    for call in tool_calls:
                        async for evt in self._dispatch_tool_call(call, agent_id):
                            yield evt
                            if self.state.status == HarnessStatus.PAUSED_FOR_PERMISSION:
                                yield SSEEvent(
                                    EventType.DONE,
                                    {
                                        "status": self.state.status.value,
                                        "pending_permission": self.state.pending_permission.to_dict(),
                                    },
                                )
                                return
                            if self.state.status == HarnessStatus.FAILED:
                                yield SSEEvent(EventType.DONE, {"status": self.state.status.value})
                                return
                    # Fall through: loop back up and give the model another turn
                    # with the freshly appended tool result(s) / handoff briefing.

            except BudgetExceededError as exc:
                self.state.status = HarnessStatus.BUDGET_EXCEEDED
                self.state.record_step("budget_exceeded", dimension=exc.dimension, used=exc.used, limit=exc.limit)
                yield SSEEvent(
                    EventType.BUDGET_EXCEEDED,
                    {"dimension": exc.dimension, "used": exc.used, "limit": exc.limit, "message": str(exc)},
                )
                yield SSEEvent(EventType.DONE, {"status": self.state.status.value})
                return
            except ToolExecutionError as exc:
                self.state.status = HarnessStatus.FAILED
                yield SSEEvent(EventType.ERROR, {"phase": "tool_execution", "message": str(exc)})
                yield SSEEvent(EventType.DONE, {"status": self.state.status.value})
                return

            yield SSEEvent(EventType.DONE, {"status": self.state.status.value})
        finally:
            await self._record_memory()
            await self._fire_on_done()

    async def resume_after_permission(self, request_id: str, *, approved: bool, actor: str) -> AsyncIterator[SSEEvent]:
        """External entry point: unblock a paused Permission Gate and continue the turn.

        Wrapped in `try/finally` like `run()`, for the same reason: on the
        approved path this falls through into a nested `self.run("")` call
        that already records memory itself, so this ends up recording twice
        (harmless — each write is an idempotent overwrite of the latest
        snapshot) but every exit path, including denial, is guaranteed to
        record exactly once at minimum.

        The `on_done` hook is handled slightly differently than memory
        (see `_fire_on_done`): unlike an idempotent memory snapshot
        overwrite, invoking a hook twice for the same logical completion is
        a real, observable duplicate (e.g. double-counted metrics), so we
        track whether this call delegated into a nested `self.run("")` — if
        it did, that nested call's own `finally` already fired `on_done` for
        us and we must not fire it again here.
        """
        delegated_to_nested_run = False
        try:
            request = self.permission_gate.resolve(request_id, approved=approved, actor=actor)
            yield SSEEvent(
                EventType.PERMISSION_RESOLVED,
                {"request_id": request_id, "approved": approved, "actor": actor, "tool_name": request.tool_name},
            )
            self.state.pending_permission = None

            if not approved:
                self.state.status = HarnessStatus.FAILED
                self.state.append_message(
                    Message(
                        role=Role.TOOL,
                        content=f"Permission denied for tool '{request.tool_name}' by {actor}.",
                        name=request.tool_name,
                        tool_call_id=request.tool_call_id,
                    )
                )
                yield SSEEvent(EventType.DONE, {"status": self.state.status.value})
                return

            try:
                result = await self.tool_registry.execute(request.tool_name, request.arguments)
            except ToolExecutionError as exc:
                self.state.status = HarnessStatus.FAILED
                yield SSEEvent(EventType.ERROR, {"phase": "tool_execution", "message": str(exc)})
                yield SSEEvent(EventType.DONE, {"status": self.state.status.value})
                return

            result_text = self._stringify_tool_result(result)
            try:
                result_text = self.guardrail.sanitize_output(result_text)
            except GuardrailViolation as violation:
                result_text = f"[output redacted by guardrail: {violation.rule_name}]"

            self.state.append_message(
                Message(role=Role.TOOL, content=result_text, name=request.tool_name, tool_call_id=request.tool_call_id)
            )
            yield SSEEvent(EventType.TOOL_RESULT, {"tool_name": request.tool_name, "result": result_text})

            # Continue the turn: feed the tool result back into the model.
            delegated_to_nested_run = True
            async for evt in self.run(""):
                yield evt
        finally:
            await self._record_memory()
            if not delegated_to_nested_run:
                await self._fire_on_done()

    # -- internals --------------------------------------------------------------

    async def _record_memory(self) -> None:
        if self.memory is not None:
            await self.memory.record_turn(user_id=self.user_id, state=self.state)

    async def _fire_on_done(self) -> None:
        """Fire the `on_done` hook, but only once the run has truly concluded.

        `PAUSED_FOR_PERMISSION` is not a terminal status — the turn hasn't
        finished, it's waiting on an external decision — so `on_done` is
        deliberately skipped in that case. It fires later, exactly once, when
        `resume_after_permission()` (directly, on denial/error, or via its
        nested `self.run("")` call, on approval) drives the state to an
        actual terminal status (`COMPLETED`, `FAILED`, `BUDGET_EXCEEDED`).
        """
        if self.hooks is None:
            return
        if self.state.status == HarnessStatus.PAUSED_FOR_PERMISSION:
            return
        await self.hooks.run(ON_DONE, HookContext(point=ON_DONE, state=self.state))

    async def _dispatch_tool_call(self, call: Dict[str, Any], agent_id: str) -> AsyncIterator[SSEEvent]:
        tool_name = call.get("name", "")
        arguments = call.get("input", {}) or {}
        tool_call_id = call.get("id") or ""

        yield SSEEvent(EventType.TOOL_CALL, {"tool_name": tool_name, "arguments": arguments, "agent": agent_id})

        # pre_tool_call fires BEFORE the __handoff__ name-check, i.e. for
        # every tool call this method sees, handoff pseudo-tool included.
        # This gives a single audit/veto point visibility into (and veto
        # power over) handoff attempts too, overlapping with the dedicated
        # on_handoff hook below by design — see specs/15-hooks.md for the
        # rationale. If a hook vetoes, the whole dispatch (handoff or real
        # tool) is aborted here.
        if self.hooks is not None:
            hook_result = await self.hooks.run(
                PRE_TOOL_CALL,
                HookContext(point=PRE_TOOL_CALL, state=self.state, tool_name=tool_name, arguments=arguments),
            )
            if hook_result is not None and hook_result.veto:
                self.state.status = HarnessStatus.FAILED
                yield SSEEvent(
                    EventType.HOOK_VETO,
                    {
                        "point": PRE_TOOL_CALL,
                        "tool_name": tool_name,
                        "reason": hook_result.veto_reason or f"pre_tool_call hook vetoed '{tool_name}'.",
                    },
                )
                return
            if hook_result is not None and hook_result.modified_arguments is not None:
                arguments = hook_result.modified_arguments

        if tool_name == HANDOFF_TOOL_NAME:
            async for evt in self._handle_handoff(arguments, agent_id):
                yield evt
            return

        try:
            tool = self.tool_registry.get(tool_name)
        except Exception as exc:  # noqa: BLE001 - any lookup failure must not crash the engine
            self.state.status = HarnessStatus.FAILED
            yield SSEEvent(EventType.ERROR, {"phase": "tool_lookup", "message": str(exc)})
            return

        if self.permission_gate.requires_gate(tool.permission_level):
            request = self.permission_gate.open_request(
                tool_name=tool_name,
                arguments=arguments,
                level=tool.permission_level,
                reason=f"Tool '{tool_name}' is classified {tool.permission_level.value} risk and requires human approval.",
                tool_call_id=tool_call_id,
            )
            self.state.status = HarnessStatus.PAUSED_FOR_PERMISSION
            self.state.pending_permission = PendingPermission(
                request_id=request.request_id,
                tool_name=tool_name,
                arguments=arguments,
                reason=request.reason,
                tool_call_id=tool_call_id,
            )
            self.state.record_step("permission_wait", tool_name=tool_name, request_id=request.request_id)
            yield SSEEvent(
                EventType.PERMISSION_REQUEST,
                {
                    "request_id": request.request_id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "level": tool.permission_level.value,
                    "reason": request.reason,
                },
            )
            return  # Halts here — run() sees PAUSED_FOR_PERMISSION and stops the generator.

        try:
            result = await self.tool_registry.execute(tool_name, arguments)
        except ToolExecutionError as exc:
            self.state.status = HarnessStatus.FAILED
            yield SSEEvent(EventType.ERROR, {"phase": "tool_execution", "tool_name": tool_name, "message": str(exc)})
            return

        result_text = self._stringify_tool_result(result)
        try:
            result_text = self.guardrail.sanitize_output(result_text)
        except GuardrailViolation as violation:
            result_text = f"[output redacted by guardrail: {violation.rule_name}]"

        self.state.append_message(Message(role=Role.TOOL, content=result_text, name=tool_name, tool_call_id=tool_call_id))
        yield SSEEvent(EventType.TOOL_RESULT, {"tool_name": tool_name, "result": result_text, "agent": agent_id})

        # post_tool_call: side-effect only. The tool has already run and its
        # result is already appended to state, so a HookResult(veto=True)
        # returned here is intentionally ignored — there is nothing left to
        # abort.
        if self.hooks is not None:
            await self.hooks.run(
                POST_TOOL_CALL,
                HookContext(
                    point=POST_TOOL_CALL, state=self.state, tool_name=tool_name, arguments=arguments, result=result
                ),
            )

    async def _handle_handoff(self, arguments: Dict[str, Any], from_agent_id: str) -> AsyncIterator[SSEEvent]:
        request = HandoffRequest(
            target_agent_id=arguments.get("target_agent_id", ""),
            reason=arguments.get("reason", ""),
            context_payload=arguments.get("context_payload", {}) or {},
        )
        try:
            self.handoff_router.validate(request)
        except HandoffError as exc:
            self.state.status = HarnessStatus.FAILED
            yield SSEEvent(EventType.ERROR, {"phase": "handoff", "message": str(exc)})
            return

        # on_handoff fires after the target is validated but BEFORE the
        # agent stack actually changes, so a veto can still prevent the
        # transfer. A veto is treated exactly like a HandoffError: FAILED
        # status, an `error` SSE event, and no push_agent().
        if self.hooks is not None:
            hook_result = await self.hooks.run(
                ON_HANDOFF,
                HookContext(
                    point=ON_HANDOFF,
                    state=self.state,
                    target_agent_id=request.target_agent_id,
                    reason=request.reason,
                ),
            )
            if hook_result is not None and hook_result.veto:
                self.state.status = HarnessStatus.FAILED
                message = hook_result.veto_reason or f"Handoff to '{request.target_agent_id}' vetoed by hook."
                yield SSEEvent(EventType.ERROR, {"phase": "handoff", "message": message})
                return

        briefing = self.handoff_router.build_briefing(request, from_agent_id=from_agent_id)
        self.state.push_agent(request.target_agent_id)
        self.state.append_message(Message(role=Role.SYSTEM, content=briefing))
        self.state.record_step("handoff", **{"from": from_agent_id, "to": request.target_agent_id, "reason": request.reason})

        yield SSEEvent(
            EventType.HANDOFF,
            {
                "from_agent": from_agent_id,
                "to_agent": request.target_agent_id,
                "reason": request.reason,
                "context_payload": request.context_payload,
            },
        )

    def _effective_system_prompt(self, agent_id: str) -> str:
        if agent_id == "root" or not self.subagent_registry.has(agent_id):
            base = self.state.system_prompt
        else:
            profile = self.subagent_registry.get(agent_id)
            base = f"{self.state.system_prompt}\n\n[Active subagent: {profile.name}]\n{profile.system_prompt}"

        if self.memory is None:
            return base
        context = self.memory.context_block(self.user_id)
        return f"{context}\n\n{base}" if context else base

    def _messages_for_wire(self) -> List[Dict[str, Any]]:
        # Targets the OpenAI/Ollama-style flat tool-calling schema (assistant
        # messages carry a `tool_calls` array; results come back as
        # role="tool" + tool_call_id). Anthropic's Messages API instead wants
        # block-structured content (tool_use/tool_result content blocks) and
        # only allows user/assistant roles — AnthropicModelClient needs its
        # own adapter over this wire format rather than consuming it as-is.
        wire: List[Dict[str, Any]] = []
        for m in self.state.messages:
            if m.role == Role.SUMMARY:
                wire.append({"role": "user", "content": f"[CONTEXT SUMMARY]\n{m.content}"})
            elif m.role == Role.TOOL:
                entry: Dict[str, Any] = {"role": "tool", "content": m.content}
                if m.tool_call_id:
                    entry["tool_call_id"] = m.tool_call_id
                if m.name:
                    entry["name"] = m.name
                wire.append(entry)
            elif m.role == Role.SYSTEM:
                wire.append({"role": "user", "content": f"[SYSTEM NOTICE]\n{m.content}"})
            elif m.role == Role.ASSISTANT and m.tool_calls:
                wire.append(
                    {
                        "role": "assistant",
                        "content": m.content,
                        "tool_calls": [
                            {
                                "id": c.get("id", ""),
                                "type": "function",
                                "function": {"name": c.get("name", ""), "arguments": c.get("input", {})},
                            }
                            for c in m.tool_calls
                        ],
                    }
                )
            else:
                wire.append({"role": m.role.value, "content": m.content})
        return wire

    @staticmethod
    def _stringify_tool_result(result: Any) -> str:
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except TypeError:
            return str(result)
