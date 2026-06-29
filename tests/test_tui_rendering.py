"""Tests for pure Rich rendering of Axis TUI state."""

from io import StringIO

from rich.console import Console, RenderableType

from axis_agent import AgentToolResult, ToolCall
from axis_coding.tui import TuiMessageItem, TuiNoticeItem, TuiState, TuiToolItem
from axis_coding.tui.rendering import format_tui_status, render_tui_state


def render_text(renderable: RenderableType) -> str:
    output = StringIO()
    console = Console(file=output, width=100, color_system=None)
    console.print(renderable)
    return output.getvalue()


def test_renderer_includes_committed_live_tool_and_notice_state() -> None:
    call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    state = TuiState(
        items=[
            TuiMessageItem(role="user", text="Inspect the project"),
            TuiMessageItem(role="thinking", text="I should read first."),
            TuiToolItem(
                tool_call=call,
                summary="read README.md",
                status="succeeded",
                updates=["reading"],
                result=AgentToolResult(
                    tool_call_id="call-1",
                    name="read",
                    ok=True,
                    content="README contents",
                ),
            ),
            TuiNoticeItem(
                level="retry",
                text="Retrying provider request",
                attempt=2,
                max_attempts=3,
            ),
        ],
        assistant_buffer="Streaming answer",
    )

    text = render_text(render_tui_state(state))

    for expected in (
        "You",
        "Inspect the project",
        "Thinking",
        "I should read first.",
        "✓ read README.md",
        "README contents",
        "[2/3] Retrying provider request",
        "Axis…",
        "Streaming answer",
    ):
        assert expected in text


def test_tool_preview_keeps_bash_tail_and_read_head() -> None:
    content = "\n".join(f"line {number}" for number in range(1, 36))
    bash_call = ToolCall(id="bash", name="bash", arguments={"command": "test"})
    read_call = ToolCall(id="read", name="read", arguments={"path": "file"})
    state = TuiState(
        items=[
            TuiToolItem(
                tool_call=bash_call,
                summary="$ test",
                status="failed",
                result=AgentToolResult(
                    tool_call_id="bash",
                    name="bash",
                    ok=False,
                    content=content,
                ),
            ),
            TuiToolItem(
                tool_call=read_call,
                summary="read file",
                status="succeeded",
                result=AgentToolResult(
                    tool_call_id="read",
                    name="read",
                    ok=True,
                    content=content,
                ),
            ),
        ]
    )

    text = render_text(render_tui_state(state))

    assert "earlier lines omitted" in text
    assert "more lines omitted" in text
    assert "line 35" in text
    assert "line 1" in text


def test_empty_state_and_status_are_human_readable() -> None:
    state = TuiState()

    assert "Axis is ready" in render_text(render_tui_state(state))
    assert format_tui_status(state, model="deepseek-v4-pro", cwd="/repo") == (
        "Axis · deepseek-v4-pro · /repo · Ready"
    )

    state.running = True
    state.current_turn = 2
    state.queued_follow_up = ("next",)
    assert format_tui_status(state, model="model", cwd="/repo").endswith(
        "Running · turn 2 · queued 1"
    )
