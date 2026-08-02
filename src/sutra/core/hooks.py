from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sutra.core.state import HarnessState

# -- hook points --------------------------------------------------------------
# Named lifecycle points an operator can attach custom async callables to.
# Plain string constants (not an Enum) so a hook can be registered against a
# bare string like "pre_tool_call" without importing an enum type — mirrors
# the string-literal `action` field already used on `ExecutionStep` in
# state.py. Use the constants below (or the equivalent literal string) when
# calling `HookRegistry.register()` / `HookRegistry.run()`.

PRE_TOOL_CALL = "pre_tool_call"
POST_TOOL_CALL = "post_tool_call"
PRE_MODEL_CALL = "pre_model_call"
ON_HANDOFF = "on_handoff"
ON_DONE = "on_done"

ALL_HOOK_POINTS = (PRE_TOOL_CALL, POST_TOOL_CALL, PRE_MODEL_CALL, ON_HANDOFF, ON_DONE)


@dataclass
class HookContext:
    """Everything a hook needs to know about the lifecycle event it's observing.

    One flexible dataclass with optional fields covers all five hook points
    rather than a class hierarchy — most fields are only populated for the
    hook point(s) they're relevant to:

    - `tool_name` / `arguments` / `result`: populated for `pre_tool_call`
      (name + arguments), `post_tool_call` (name + arguments + result).
    - `target_agent_id` / `reason`: populated for `on_handoff`.
    - `pre_model_call` and `on_done` only ever populate `point` and `state`.
    """

    point: str
    state: HarnessState
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    target_agent_id: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class HookResult:
    """What a hook callable may return to influence the harness.

    `veto` / `veto_reason` halt processing at hook points where vetoing is
    meaningful (`pre_tool_call`, `on_handoff`); it is ignored at points where
    it isn't (`post_tool_call`, `pre_model_call`, `on_done` — see
    `specs/15-hooks.md` for the full table). `modified_arguments` is only
    ever consulted at `pre_tool_call`, letting a hook rewrite a tool call's
    arguments before it executes.
    """

    veto: bool = False
    veto_reason: Optional[str] = None
    modified_arguments: Optional[Dict[str, Any]] = None


HookCallable = Callable[[HookContext], Awaitable[Optional[HookResult]]]


class HookRegistry:
    """Ordered collection of lifecycle hooks, keyed by hook point.

    Operators register plain async callables against one of the five named
    points (`PRE_TOOL_CALL`, `POST_TOOL_CALL`, `PRE_MODEL_CALL`, `ON_HANDOFF`,
    `ON_DONE`) for audit logging, custom veto logic, or metrics, without
    touching harness code.
    """

    def __init__(self) -> None:
        self._hooks: Dict[str, List[HookCallable]] = {}

    def register(self, point: str, hook: HookCallable) -> None:
        self._hooks.setdefault(point, []).append(hook)

    async def run(self, point: str, context: HookContext) -> Optional[HookResult]:
        """Run every hook registered for `point`, in registration order.

        - If no hooks are registered for `point`, this is a cheap no-op that
          returns `None` immediately without allocating anything.
        - Hooks run in registration order. The moment a hook returns a
          `HookResult` with `veto=True`, iteration stops (remaining hooks do
          NOT run) and that result is returned immediately — exactly like a
          guardrail block halts further processing.
        - If no hook vetoes, every hook still runs (for side effects such as
          audit logging or metrics). The last non-`None` `HookResult`
          returned by a non-vetoing hook is returned to the caller — this is
          what lets a `pre_tool_call` hook communicate `modified_arguments`
          back to the harness even though nothing was vetoed. If every hook
          returns `None`, `run()` returns `None`.
        """
        hooks = self._hooks.get(point)
        if not hooks:
            return None

        final_result: Optional[HookResult] = None
        for hook in hooks:
            result = await hook(context)
            if result is None:
                continue
            if result.veto:
                return result
            final_result = result
        return final_result
