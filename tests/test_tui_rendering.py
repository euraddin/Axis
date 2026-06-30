"""Tests for pure rendering of semantic Axis transcript state."""

from io import StringIO

from rich.console import Console, RenderableType

from axis_coding.tui import ChatItem, TuiState, format_terminal_command_result_block
from axis_coding.tui.rendering import format_tui_status, render_tui_state


def render_text(renderable: RenderableType) -> str:
    output = StringIO()
    Console(file=output, width=100, color_system=None).print(renderable)
    return output.getvalue()


def test_renderer_collapses_thinking_and_tool_results_by_default() -> None:
    state = TuiState(
        items=[
            ChatItem(role="user", text="Inspect **literally**"),
            ChatItem(role="thinking", text="internal plan"),
            ChatItem(
                role="tool",
                text="→ read README.md",
                tool_result_text="✓ read\nREADME contents",
            ),
            ChatItem(role="assistant", text="## Done\n\n- one\n- two"),
        ]
    )

    text = render_text(render_tui_state(state))

    assert "Inspect **literally**" in text
    assert "Thinking… Press Ctrl+T" in text
    assert "internal plan" not in text
    assert "→ read README.md" in text
    assert "README contents" not in text
    assert "Done" in text
    assert "one" in text


def test_renderer_expands_thinking_and_tool_results() -> None:
    state = TuiState(
        items=[
            ChatItem(role="thinking", text="internal plan"),
            ChatItem(
                role="tool",
                text="$ pytest",
                tool_result_text="✓ bash\n2 passed",
            ),
        ],
        show_thinking=True,
        show_tool_results=True,
    )

    text = render_text(render_tui_state(state))

    assert "internal plan" in text
    assert "2 passed" in text


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


def test_terminal_output_preview_is_bounded_but_keeps_context_label() -> None:
    output = "\n".join(f"line {index}" for index in range(130))

    rendered = format_terminal_command_result_block(
        ok=False,
        added_to_context=False,
        output=output,
    )

    assert rendered.startswith("✗ bash · not added to context")
    assert "line 119" in rendered
    assert "line 120" not in rendered
    assert "10 more lines" in rendered
