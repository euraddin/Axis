"""Pure Rich rendering for structured Axis TUI state."""

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from axis_coding.tui.state import (
    TuiMessageItem,
    TuiNoticeItem,
    TuiState,
    TuiToolItem,
)

_TOOL_PREVIEW_LINES = 30
_TOOL_PREVIEW_CHARS = 4_000


def render_tui_state(state: TuiState) -> Group:
    """Render committed and live state without mutating it."""
    renderables: list[RenderableType] = [_render_item(item) for item in state.items]
    if state.thinking_buffer:
        renderables.append(_message_panel("Thinking…", state.thinking_buffer, "dim cyan"))
    if state.assistant_buffer:
        renderables.append(_message_panel("Axis…", state.assistant_buffer, "green"))
    if not renderables:
        renderables.append(
            Text("Axis is ready. Describe a coding task below.", style="dim", justify="center")
        )
    return Group(*renderables)


def format_tui_status(state: TuiState, *, model: str, cwd: str) -> str:
    """Build the compact status line shown above the prompt."""
    if state.running:
        activity = (
            f"Running · turn {state.current_turn}" if state.current_turn is not None else "Running"
        )
    elif state.cancelled:
        activity = "Cancelled"
    elif state.error:
        activity = "Error"
    else:
        activity = "Ready"
    queue = f" · queued {state.queued_message_count}" if state.queued_message_count else ""
    return f"Axis · {model} · {cwd} · {activity}{queue}"


def _render_item(item: TuiMessageItem | TuiToolItem | TuiNoticeItem) -> RenderableType:
    if isinstance(item, TuiMessageItem):
        title, style = {
            "user": ("You", "blue"),
            "assistant": ("Axis", "green"),
            "thinking": ("Thinking", "dim cyan"),
        }[item.role]
        return _message_panel(title, item.text, style)
    if isinstance(item, TuiToolItem):
        return _tool_panel(item)
    title, style = {
        "status": ("Status", "dim"),
        "retry": ("Retry", "cyan"),
        "error": ("Error", "red"),
    }[item.level]
    text = item.text
    if item.level == "retry" and item.attempt is not None and item.max_attempts is not None:
        text = f"[{item.attempt}/{item.max_attempts}] {text}"
    return _message_panel(title, text, style)


def _message_panel(title: str, content: str, style: str) -> Panel:
    return Panel(Text(content), title=title, border_style=style, padding=(0, 1))


def _tool_panel(item: TuiToolItem) -> Panel:
    icon, style = {
        "running": ("…", "yellow"),
        "succeeded": ("✓", "green"),
        "failed": ("✗", "red"),
        "cancelled": ("■", "dim yellow"),
    }[item.status]
    body = Text(f"{icon} {item.summary}")
    if item.updates:
        body.append(f"\n{item.updates[-1]}", style="dim")
    if item.result is not None:
        result_text = item.result.content
        if item.result.error and item.result.error not in result_text:
            result_text = f"{result_text}\nError: {item.result.error}".strip()
        if result_text:
            preview = _preview_tool_result(
                result_text,
                keep_tail=item.tool_call.name == "bash",
            )
            body.append(f"\n\n{preview}")
    return Panel(body, title="Tool", border_style=style, padding=(0, 1))


def _preview_tool_result(content: str, *, keep_tail: bool) -> str:
    lines = content.splitlines()
    omitted_lines = max(0, len(lines) - _TOOL_PREVIEW_LINES)
    if omitted_lines:
        if keep_tail:
            lines = [f"[… {omitted_lines} earlier lines omitted …]", *lines[-_TOOL_PREVIEW_LINES:]]
        else:
            lines = [*lines[:_TOOL_PREVIEW_LINES], f"[… {omitted_lines} more lines omitted …]"]
    preview = "\n".join(lines)
    if len(preview) <= _TOOL_PREVIEW_CHARS:
        return preview
    if keep_tail:
        return f"[… earlier output omitted …]\n{preview[-_TOOL_PREVIEW_CHARS:]}"
    return f"{preview[:_TOOL_PREVIEW_CHARS]}\n[… more output omitted …]"
