"""Tests for Axis's portable agent event contract."""

import pytest
from pydantic import ValidationError

from axis_agent import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AgentToolResult,
    AssistantMessage,
    ContextCompactionEvent,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    QueueUpdateEvent,
    RetryEvent,
    ThinkingDeltaEvent,
    ToolApprovalRequestEvent,
    ToolApprovalResolvedEvent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)


def test_event_types_form_a_stable_runtime_vocabulary() -> None:
    tool_call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    result = AgentToolResult(
        tool_call_id="call-1",
        name="read",
        ok=True,
        content="contents",
    )
    assistant = AssistantMessage(content="Done")
    events: list[AgentEvent] = [
        AgentStartEvent(),
        TurnStartEvent(turn=1),
        MessageStartEvent(),
        MessageDeltaEvent(delta="Do"),
        ThinkingDeltaEvent(delta="inspect"),
        MessageEndEvent(message=assistant),
        ToolApprovalRequestEvent(tool_call=tool_call),
        ToolApprovalResolvedEvent(tool_call_id="call-1", decision="allow_once"),
        ToolExecutionStartEvent(tool_call=tool_call),
        ToolExecutionUpdateEvent(tool_call_id="call-1", message="running"),
        ToolExecutionEndEvent(result=result),
        RetryEvent(
            attempt=1,
            max_attempts=3,
            delay_seconds=0.25,
            message="temporary failure",
        ),
        QueueUpdateEvent(steering=("adjust",), follow_up=("then test",)),
        ContextCompactionEvent(
            before_tokens=100,
            after_tokens=40,
            trigger_tokens=80,
            compacted_entries=4,
            retained_entries=2,
        ),
        ErrorEvent(message="cancelled", recoverable=True),
        TurnEndEvent(turn=1),
        AgentEndEvent(),
    ]

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_delta",
        "thinking_delta",
        "message_end",
        "tool_approval_request",
        "tool_approval_resolved",
        "tool_execution_start",
        "tool_execution_update",
        "tool_execution_end",
        "retry",
        "queue_update",
        "context_compaction",
        "error",
        "turn_end",
        "agent_end",
    ]


def test_message_delta_is_ephemeral_but_message_end_carries_complete_message() -> None:
    delta = MessageDeltaEvent(delta="Hel")
    complete = AssistantMessage(content="Hello")
    end = MessageEndEvent(message=complete)

    assert delta.model_dump() == {"type": "message_delta", "delta": "Hel"}
    assert end.message == complete
    assert end.model_dump() == {
        "type": "message_end",
        "message": {"role": "assistant", "content": "Hello", "tool_calls": []},
    }


def test_event_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MessageDeltaEvent(delta="hello", unexpected=True)  # type: ignore[call-arg]


def test_queue_update_defaults_to_empty_immutable_snapshots() -> None:
    event = QueueUpdateEvent()

    assert event.steering == ()
    assert event.follow_up == ()
