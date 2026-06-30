"""Pure rendering helpers shared by Axis TUI tests and status chrome."""

from rich.console import Group, RenderableType
from rich.text import Text

from axis_coding.tui.config import AXIS_DARK_THEME, TuiTheme
from axis_coding.tui.state import ChatItem, TuiState
from axis_coding.tui.widgets import render_chat_item


def render_tui_state(
    state: TuiState,
    *,
    theme: TuiTheme = AXIS_DARK_THEME,
) -> Group:
    """Render semantic state without requiring a mounted Textual app."""
    renderables: list[RenderableType] = []
    hidden_thinking = False
    for item in state.items:
        if item.role == "thinking" and not state.show_thinking:
            if not hidden_thinking:
                renderables.append(
                    render_chat_item(
                        ChatItem(
                            role="thinking",
                            text="Thinking… Press Ctrl+T to show thinking tokens.",
                        ),
                        theme=theme,
                    )
                )
                hidden_thinking = True
            continue
        hidden_thinking = False
        renderables.append(
            render_chat_item(
                item,
                theme=theme,
                show_tool_results=state.show_tool_results or item.always_show_tool_result,
            )
        )
    if state.assistant_buffer:
        renderables.append(
            render_chat_item(
                ChatItem(role="assistant", text=state.assistant_buffer),
                theme=theme,
            )
        )
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
