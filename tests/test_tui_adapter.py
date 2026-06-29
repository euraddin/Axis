"""Tests for the UI-framework-independent AgentEvent adapter."""

from axis_agent import (
    AgentEndEvent,
    AgentStartEvent,
    AgentToolResult,
    AssistantMessage,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    QueueUpdateEvent,
    RetryEvent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
    UserMessage,
)
from axis_coding.tui import (
    TuiEventAdapter,
    TuiMessageItem,
    TuiNoticeItem,
    TuiState,
    TuiToolItem,
    format_tool_call_summary,
)


def test_adapter_tracks_run_turn_and_queue_state() -> None:
    adapter = TuiEventAdapter()

    state = adapter.apply(AgentStartEvent())
    adapter.apply(TurnStartEvent(turn=2))
    adapter.apply(QueueUpdateEvent(steering=("adjust",), follow_up=("after",)))

    assert state is adapter.state
    assert state.running is True
    assert state.current_turn == 2
    assert state.queued_message_count == 2
    adapter.apply(TurnEndEvent(turn=2))
    adapter.apply(AgentEndEvent())
    assert state.current_turn is None
    assert state.running is False


def test_streaming_thinking_and_text_commit_without_duplication() -> None:
    adapter = TuiEventAdapter()

    adapter.apply(MessageStartEvent())
    adapter.apply(ThinkingDeltaEvent(delta="inspect "))
    adapter.apply(ThinkingDeltaEvent(delta="first"))
    assert adapter.state.thinking_buffer == "inspect first"
    adapter.apply(MessageDeltaEvent(delta="Hel"))
    adapter.apply(MessageDeltaEvent(delta="lo"))
    assert adapter.state.assistant_buffer == "Hello"
    adapter.apply(MessageEndEvent(message=AssistantMessage(content="Hello")))

    assert adapter.state.assistant_buffer == ""
    assert adapter.state.thinking_buffer == ""
    assert adapter.state.items == [
        TuiMessageItem(role="thinking", text="inspect first"),
        TuiMessageItem(role="assistant", text="Hello"),
    ]


def test_complete_assistant_message_is_authoritative_over_partial_buffer() -> None:
    state = TuiState(assistant_buffer="Hel")
    adapter = TuiEventAdapter(state)

    adapter.apply(MessageEndEvent(message=AssistantMessage(content="Hello")))

    assert state.items == [TuiMessageItem(role="assistant", text="Hello")]
    assert state.assistant_buffer == ""


def test_user_message_is_committed_in_event_order() -> None:
    adapter = TuiEventAdapter()

    adapter.apply(MessageStartEvent(message_role="user"))
    adapter.apply(MessageEndEvent(message=UserMessage(content="Hello Axis")))

    assert adapter.state.items == [TuiMessageItem(role="user", text="Hello Axis")]


def test_tool_lifecycle_keeps_structured_call_updates_and_result() -> None:
    adapter = TuiEventAdapter()
    call = ToolCall(
        id="call-1",
        name="read",
        arguments={"path": "src/app.py", "offset": 10, "limit": 20},
    )
    result = AgentToolResult(
        tool_call_id="call-1",
        name="read",
        ok=True,
        content="contents",
        data={"path": "src/app.py"},
    )

    adapter.apply(ToolExecutionStartEvent(tool_call=call))
    adapter.apply(ToolExecutionUpdateEvent(tool_call_id="call-1", message="reading"))
    adapter.apply(ToolExecutionEndEvent(result=result))

    assert adapter.state.active_tool_count == 0
    assert len(adapter.state.items) == 1
    item = adapter.state.items[0]
    assert isinstance(item, TuiToolItem)
    assert item.tool_call is call
    assert item.summary == "read src/app.py:10-29"
    assert item.updates == ["reading"]
    assert item.status == "succeeded"
    assert item.result is result


def test_tool_start_flushes_partial_assistant_text_first() -> None:
    state = TuiState(assistant_buffer="Before tool")
    adapter = TuiEventAdapter(state)

    adapter.apply(
        ToolExecutionStartEvent(
            tool_call=ToolCall(id="call-1", name="bash", arguments={"command": "pytest"})
        )
    )

    assert isinstance(state.items[0], TuiMessageItem)
    assert state.items[0].text == "Before tool"
    assert isinstance(state.items[1], TuiToolItem)
    assert state.items[1].summary == "$ pytest"


def test_orphan_failed_and_cancelled_tool_results_remain_visible() -> None:
    adapter = TuiEventAdapter()

    adapter.apply(
        ToolExecutionEndEvent(
            result=AgentToolResult(
                tool_call_id="missing",
                name="unknown",
                ok=False,
                content="Unknown tool",
                error="Unknown tool",
            )
        )
    )
    adapter.apply(
        ToolExecutionEndEvent(
            result=AgentToolResult(
                tool_call_id="cancelled",
                name="bash",
                ok=False,
                content="Tool call cancelled",
                error="Tool call cancelled",
            )
        )
    )

    tools = [item for item in adapter.state.items if isinstance(item, TuiToolItem)]
    assert [tool.status for tool in tools] == ["failed", "cancelled"]


def test_orphan_tool_update_becomes_status_notice() -> None:
    adapter = TuiEventAdapter()

    adapter.apply(ToolExecutionUpdateEvent(tool_call_id="missing", message="still running"))

    assert adapter.state.items == [TuiNoticeItem(level="status", text="still running")]


def test_retry_and_recoverable_error_remain_in_transcript() -> None:
    state = TuiState(running=True)
    adapter = TuiEventAdapter(state)

    adapter.apply(
        RetryEvent(
            attempt=2,
            max_attempts=3,
            delay_seconds=1,
            message="Retrying after HTTP 503",
        )
    )
    adapter.apply(ErrorEvent(message="Reached max turns", recoverable=True))

    assert state.running is True
    assert state.error == "Reached max turns"
    retry, error = state.items
    assert isinstance(retry, TuiNoticeItem)
    assert retry.level == "retry"
    assert (retry.attempt, retry.max_attempts) == (2, 3)
    assert isinstance(error, TuiNoticeItem)
    assert error.level == "error"
    assert error.recoverable is True


def test_non_recoverable_error_flushes_text_and_stops_run() -> None:
    state = TuiState(running=True, assistant_buffer="partial")
    adapter = TuiEventAdapter(state)

    adapter.apply(ErrorEvent(message="provider failed", recoverable=False))

    assert state.running is False
    assert state.error == "provider failed"
    assert state.items == [
        TuiMessageItem(role="assistant", text="partial"),
        TuiNoticeItem(level="error", text="provider failed", recoverable=False),
    ]


def test_cancellation_is_status_not_terminal_error() -> None:
    state = TuiState(running=True)
    adapter = TuiEventAdapter(state)

    adapter.apply(ErrorEvent(message="Agent run cancelled", recoverable=True))

    assert state.running is True
    assert state.cancelled is True
    assert state.error is None
    assert state.items == [TuiNoticeItem(level="status", text="Agent run cancelled.")]


def test_agent_end_flushes_unfinished_live_buffers() -> None:
    state = TuiState(running=True, thinking_buffer="thought", assistant_buffer="answer")
    adapter = TuiEventAdapter(state)

    adapter.apply(AgentEndEvent())

    assert state.items == [
        TuiMessageItem(role="thinking", text="thought"),
        TuiMessageItem(role="assistant", text="answer"),
    ]
    assert state.running is False


def test_tool_summary_formats_coding_tools_and_fallbacks() -> None:
    assert (
        format_tool_call_summary(ToolCall(id="1", name="edit", arguments={"path": "src/app.py"}))
        == "edit src/app.py"
    )
    assert (
        format_tool_call_summary(
            ToolCall(id="2", name="bash", arguments={"command": "pytest", "timeout": 30})
        )
        == "$ pytest (timeout 30s)"
    )
    assert (
        format_tool_call_summary(ToolCall(id="3", name="custom", arguments={"value": 1}))
        == "custom {'value': 1}"
    )
