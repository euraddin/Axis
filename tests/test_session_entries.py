"""Tests for Axis append-only session entry and JSONL contracts."""

import json

import pytest
from pydantic import ValidationError

from axis_agent import (
    AssistantMessage,
    BranchSummaryEntry,
    CompactionEntry,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionInfoEntry,
    SessionJsonlError,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    entries_from_json_lines,
    entry_from_json_line,
    entry_to_json_line,
)


def test_session_entry_defaults_create_distinct_ids_and_timestamps() -> None:
    first = MessageEntry(message=UserMessage(content="first"))
    second = MessageEntry(message=UserMessage(content="second"))

    assert first.id != second.id
    assert len(first.id) == 32
    assert first.parent_id is None
    assert first.timestamp > 0


def test_all_v1_entry_shapes_are_stable() -> None:
    entries = [
        SessionInfoEntry(
            id="info",
            timestamp=10,
            created_at=5,
            cwd="/workspace",
            title="Axis session",
            system="You are Axis.",
        ),
        ModelChangeEntry(
            id="model",
            parent_id="info",
            timestamp=11,
            model="deepseek-v4-pro",
        ),
        MessageEntry(
            id="message",
            parent_id="model",
            timestamp=12,
            message=UserMessage(content="hello"),
        ),
        LeafEntry(
            id="leaf",
            parent_id="message",
            timestamp=13,
            entry_id="message",
        ),
    ]

    assert [entry.model_dump() for entry in entries] == [
        {
            "id": "info",
            "parent_id": None,
            "timestamp": 10.0,
            "type": "session_info",
            "created_at": 5.0,
            "cwd": "/workspace",
            "title": "Axis session",
            "system": "You are Axis.",
        },
        {
            "id": "model",
            "parent_id": "info",
            "timestamp": 11.0,
            "type": "model_change",
            "model": "deepseek-v4-pro",
        },
        {
            "id": "message",
            "parent_id": "model",
            "timestamp": 12.0,
            "type": "message",
            "message": {"role": "user", "content": "hello"},
        },
        {
            "id": "leaf",
            "parent_id": "message",
            "timestamp": 13.0,
            "type": "leaf",
            "entry_id": "message",
        },
    ]


@pytest.mark.parametrize(
    "message",
    [
        UserMessage(content="你好"),
        AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="call-1", name="read", arguments={"path": "a.py"})],
            provider_data={"reasoning_content": "inspect"},
        ),
        ToolResultMessage(
            tool_call_id="call-1",
            name="read",
            content="contents",
            data={"path": "a.py"},
            details={"bytes": 8},
        ),
    ],
)
def test_every_transcript_message_round_trips_through_jsonl(message: object) -> None:
    entry = MessageEntry(id="entry-1", timestamp=1, message=message)  # type: ignore[arg-type]

    line = entry_to_json_line(entry)
    parsed = entry_from_json_line(line)

    assert line.endswith("\n")
    assert "\n" not in line[:-1]
    assert parsed == entry
    assert isinstance(parsed, MessageEntry)


def test_json_line_preserves_discriminator_and_unicode() -> None:
    entry = MessageEntry(
        id="entry-1",
        timestamp=1,
        message=UserMessage(content="你好，Axis"),
    )

    payload = json.loads(entry_to_json_line(entry))

    assert payload["type"] == "message"
    assert payload["message"] == {"role": "user", "content": "你好，Axis"}


@pytest.mark.parametrize(
    "entry",
    [
        CompactionEntry(
            id="compact",
            summary="Earlier work summary.",
            replaces_entry_ids=["one", "two"],
        ),
        BranchSummaryEntry(
            id="branch-summary",
            summary="Abandoned branch summary.",
            branch_root_id="root",
        ),
    ],
)
def test_context_summary_entries_round_trip(entry: object) -> None:
    line = entry_to_json_line(entry)  # type: ignore[arg-type]

    assert entry_from_json_line(line) == entry


def test_legacy_session_info_without_system_remains_readable() -> None:
    entry = entry_from_json_line(
        '{"id":"info","parent_id":null,"timestamp":1,"type":"session_info",'
        '"created_at":1,"cwd":"/workspace","title":null}'
    )

    assert isinstance(entry, SessionInfoEntry)
    assert entry.system is None


def test_entries_decoder_skips_blank_lines_and_keeps_source_order() -> None:
    first = ModelChangeEntry(id="model", timestamp=1, model="deepseek-v4-pro")
    second = LeafEntry(id="leaf", parent_id="model", timestamp=2, entry_id="model")

    parsed = entries_from_json_lines(
        [entry_to_json_line(first), "\n", "   ", entry_to_json_line(second)]
    )

    assert parsed == [first, second]


def test_invalid_jsonl_reports_physical_line_number() -> None:
    with pytest.raises(SessionJsonlError, match="Invalid session entry on line 3"):
        entries_from_json_lines(["\n", entry_to_json_line(LeafEntry()), '{"type":"unknown"}'])


def test_entry_models_reject_unknown_or_invalid_fields() -> None:
    with pytest.raises(ValidationError):
        MessageEntry(
            id="entry",
            message=UserMessage(content="hello"),
            unexpected=True,  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        ModelChangeEntry(model="")

    with pytest.raises(ValidationError):
        LeafEntry(timestamp=float("nan"))
