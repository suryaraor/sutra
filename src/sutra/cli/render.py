"""Rich-based live renderer for Sutra `SSEEvent`s.

Turns the harness's raw event stream into a Claude-Code-style terminal
view: streamed tokens print live, tool calls show a spinner that resolves
into a checkmark line, handoffs get a banner, and permission requests get
a bordered panel. This module only renders — it never decides what to do
about a permission request; that's the CLI app's job (see `app.py`).
"""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.status import Status

from sutra.streaming.sse import EventType, SSEEvent

_STATUS_COLOR = {
    "completed": "green",
    "failed": "red",
    "budget_exceeded": "red",
    "paused_for_permission": "yellow",
    "paused_for_handoff": "yellow",
    "running": "cyan",
}


def _format_args(args: dict) -> str:
    parts = [f"{k}={v!r}" for k, v in args.items()]
    text = ", ".join(parts)
    return text if len(text) <= 90 else text[:87] + "..."


def _truncate(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


class HarnessRenderer:
    """Consumes `SSEEvent`s and renders them to a Rich console as they arrive."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console(legacy_windows=False)
        self._current_agent: Optional[str] = None
        self._tool_status: Optional[Status] = None
        self._streamed_any_token = False

    def render(self, event: SSEEvent) -> None:
        handler = getattr(self, f"_on_{event.type.value}", None)
        if handler is not None:
            handler(event)

    def _stop_tool_status(self) -> None:
        if self._tool_status is not None:
            self._tool_status.stop()
            self._tool_status = None

    def _ensure_fresh_line(self) -> None:
        # Errors/budget-exceeded can interrupt a turn mid-stream, before a
        # MESSAGE_COMPLETE event ever fires to close out the line — without
        # this, the next panel prints jammed onto the tail of the streamed
        # text instead of starting on its own line.
        if self._streamed_any_token:
            self.console.print()
            self._streamed_any_token = False

    # -- event handlers, one per SSE event type ----------------------------

    def _on_session_start(self, event: SSEEvent) -> None:
        self._current_agent = event.data.get("active_agent")
        self._streamed_any_token = False

    def _on_token(self, event: SSEEvent) -> None:
        agent = event.data.get("agent")
        if agent != self._current_agent or not self._streamed_any_token:
            self._current_agent = agent
            self.console.print(f"\n[bold cyan]● {agent}[/bold cyan]")
        self._streamed_any_token = True
        self.console.print(event.data["text"], end="", highlight=False, markup=False)

    def _on_message_complete(self, event: SSEEvent) -> None:
        if self._streamed_any_token:
            self.console.print()
        self._streamed_any_token = False

    def _on_tool_call(self, event: SSEEvent) -> None:
        name = event.data["tool_name"]
        args = _format_args(event.data.get("arguments", {}))
        self._tool_status = self.console.status(f"[yellow]⚙ {name}({args})[/yellow]", spinner="dots")
        self._tool_status.start()

    def _on_tool_result(self, event: SSEEvent) -> None:
        self._stop_tool_status()
        name = event.data["tool_name"]
        preview = _truncate(str(event.data.get("result", "")))
        self.console.print(f"  [green]✓[/green] [yellow]{name}[/yellow] [dim]→ {preview}[/dim]")

    def _on_handoff(self, event: SSEEvent) -> None:
        self._stop_tool_status()
        self.console.print(
            f"\n[bold magenta]⇄ handoff[/bold magenta] "
            f"{event.data['from_agent']} → [bold]{event.data['to_agent']}[/bold] "
            f"[dim]({event.data['reason']})[/dim]"
        )

    def _on_permission_request(self, event: SSEEvent) -> None:
        self._stop_tool_status()
        body = (
            f"Tool:      [bold]{event.data['tool_name']}[/bold]\n"
            f"Arguments: {event.data['arguments']}\n"
            f"Risk:      [bold red]{event.data['level'].upper()}[/bold red]\n\n"
            f"{event.data['reason']}"
        )
        self.console.print(Panel(body, title="\U0001f512 Permission required", border_style="red", expand=False))

    def _on_permission_resolved(self, event: SSEEvent) -> None:
        approved = event.data["approved"]
        color = "green" if approved else "red"
        label = "approved" if approved else "denied"
        self.console.print(f"[{color}]● {label}[/{color}] by {event.data['actor']}")

    def _on_guardrail_block(self, event: SSEEvent) -> None:
        self._stop_tool_status()
        self.console.print(
            Panel(
                f"Rule: [bold]{event.data['rule']}[/bold]\n{event.data['message']}",
                title="⛔ Blocked by guardrail",
                border_style="red",
                expand=False,
            )
        )

    def _on_budget_warning(self, event: SSEEvent) -> None:
        ratio = event.data["ratio_used"] * 100
        self.console.print(f"[dim yellow]⚠ budget: {event.data['dimension']} at {ratio:.0f}%[/dim yellow]")

    def _on_budget_exceeded(self, event: SSEEvent) -> None:
        self._stop_tool_status()
        self._ensure_fresh_line()
        self.console.print(Panel(event.data["message"], title="\U0001f6d1 Budget exceeded", border_style="red", expand=False))

    def _on_compaction(self, event: SSEEvent) -> None:
        self.console.print(
            f"[dim]\U0001f5dc context compacted "
            f"(pass {event.data['compaction_count']}, now {event.data['resulting_messages']} messages)[/dim]"
        )

    def _on_error(self, event: SSEEvent) -> None:
        self._stop_tool_status()
        self._ensure_fresh_line()
        phase = event.data.get("phase", "")
        self.console.print(Panel(event.data["message"], title=f"✗ Error ({phase})", border_style="red", expand=False))

    def _on_done(self, event: SSEEvent) -> None:
        self._stop_tool_status()
        self._ensure_fresh_line()
        status = event.data["status"]
        color = _STATUS_COLOR.get(status, "white")
        self.console.print(f"[{color}]── {status} ──[/{color}]")
