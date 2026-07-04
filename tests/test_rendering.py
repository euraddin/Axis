"""Tests for Axis print-mode event renderers."""

import json
from io import StringIO

from axis_agent import (
    AgentEndEvent,
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
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from axis_coding.rendering import (
    FinalTextRenderer,
    JsonEventRenderer,
    PrintOutputMode,
    TranscriptRenderer,
    create_event_renderer,
)


def test_renderer_factory_selects_all_output_modes() -> None:
    assert isinstance(create_event_renderer(PrintOutputMode.TEXT), FinalTextRenderer)
    assert isinstance(create_event_renderer(PrintOutputMode.JSON), JsonEventRenderer)
    assert isinstance(
        create_event_renderer(PrintOutputMode.TRANSCRIPT),
        TranscriptRenderer,
    )


def test_json_renderer_emits_strict_jsonl_and_tracks_failure() -> None:
    stdout = StringIO()
    renderer = JsonEventRenderer(stdout=stdout)

    renderer.render(MessageStartEvent())
    renderer.render(QueueUpdateEvent(steering=("adjust",), follow_up=("after",)))
    renderer.render(ThinkingDeltaEvent(delta="private reasoning"))
    renderer.render(
        ContextCompactionEvent(
            before_tokens=100,
            after_tokens=40,
            trigger_tokens=80,
            compacted_entries=4,
            retained_entries=2,
        )
    )
    renderer.render(ErrorEvent(message="provider failed", recoverable=False))

    lines = stdout.getvalue().splitlines()
    assert len(lines) == 5
    assert json.loads(lines[0]) == {
        "type": "message_start",
        "message_role": "assistant",
    }
    assert json.loads(lines[1]) == {
        "type": "queue_update",
        "steering": ["adjust"],
        "follow_up": ["after"],
    }
    assert json.loads(lines[2]) == {
        "type": "thinking_delta",
        "delta": "private reasoning",
    }
    assert json.loads(lines[3]) == {
        "type": "context_compaction",
        "automatic": True,
        "before_tokens": 100,
        "after_tokens": 40,
        "trigger_tokens": 80,
        "compacted_entries": 4,
        "retained_entries": 2,
        "replays_current_prompt": False,
    }
    assert json.loads(lines[4]) == {
        "type": "error",
        "message": "provider failed",
        "recoverable": False,
        "data": None,
    }
    assert renderer.finish() is False


def test_transcript_renderer_streams_text_and_tool_activity() -> None:
    stdout = StringIO()
    stderr = StringIO()
    renderer = TranscriptRenderer(stdout=stdout, stderr=stderr)

    renderer.render(MessageStartEvent())
    renderer.render(ThinkingDeltaEvent(delta="private reasoning"))
    renderer.render(MessageDeltaEvent(delta="Hel"))
    renderer.render(MessageDeltaEvent(delta="lo"))
    renderer.render(MessageEndEvent(message=AssistantMessage(content="Hello")))
    renderer.render(
        ContextCompactionEvent(
            before_tokens=100,
            after_tokens=40,
            trigger_tokens=80,
            compacted_entries=4,
            retained_entries=2,
        )
    )
    renderer.render(
        RetryEvent(
            attempt=2,
            max_attempts=3,
            delay_seconds=0,
            message="Retrying provider request 2/3 after HTTP 503.",
        )
    )
    renderer.render(
        ToolExecutionStartEvent(
            tool_call=ToolCall(
                id="call-1",
                name="read",
                arguments={"path": "你好.py"},
            )
        )
    )
    renderer.render(ToolExecutionUpdateEvent(tool_call_id="call-1", message="reading"))
    renderer.render(
        ToolExecutionEndEvent(
            result=AgentToolResult(
                tool_call_id="call-1",
                name="read",
                ok=True,
                content="line one\nline two",
            )
        )
    )
    renderer.render(AgentEndEvent())

    assert renderer.finish() is True
    assert stdout.getvalue() == "Hello\n"
    assert "private reasoning" not in stdout.getvalue()
    assert "private reasoning" not in stderr.getvalue()
    assert stderr.getvalue() == (
        "… Auto-compacted context (100 → 40 tokens; kept 2 entries).\n"
        "… Retrying provider request 2/3 after HTTP 503.\n"
        '→ read {"path":"你好.py"}\n'
        "… reading\n"
        "✓ read\n"
        "  line one\n"
        "  line two\n"
    )


def test_transcript_renderer_falls_back_to_complete_message_without_deltas() -> None:
    stdout = StringIO()
    renderer = TranscriptRenderer(stdout=stdout, stderr=StringIO())

    renderer.render(MessageEndEvent(message=AssistantMessage(content="Complete only")))

    assert renderer.finish() is True
    assert stdout.getvalue() == "Complete only\n"


def test_transcript_renderer_fails_on_non_recoverable_error() -> None:
    stderr = StringIO()
    renderer = TranscriptRenderer(stdout=StringIO(), stderr=stderr)

    renderer.render(ErrorEvent(message="provider failed", recoverable=False))

    assert renderer.finish() is False
    assert stderr.getvalue() == "Error: provider failed\n"


def test_renderers_fail_when_recoverable_compaction_error_aborts_request() -> None:
    error = ErrorEvent(
        message="compaction failed",
        recoverable=True,
        data={"kind": "auto_compaction", "request_aborted": True},
    )
    text = FinalTextRenderer(stdout=StringIO(), stderr=StringIO())
    json_renderer = JsonEventRenderer(stdout=StringIO())
    transcript = TranscriptRenderer(stdout=StringIO(), stderr=StringIO())

    for renderer in (text, json_renderer, transcript):
        renderer.render(error)
        assert renderer.finish() is False
