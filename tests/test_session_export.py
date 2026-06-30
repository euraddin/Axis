"""Tests for Axis session tree exports."""

from pathlib import Path

import pytest

from axis_agent import (
    AssistantMessage,
    CompactionEntry,
    LeafEntry,
    MessageEntry,
    UserMessage,
)
from axis_coding.session_export import (
    SessionExportError,
    export_session_artifact,
    normalize_export_format,
    render_session_html,
)


def _branch_entries():
    root = MessageEntry(id="root", message=UserMessage(content="Start <session>"))
    left = MessageEntry(
        id="left",
        parent_id="root",
        message=AssistantMessage(content="Left branch"),
    )
    right = MessageEntry(
        id="right",
        parent_id="root",
        message=AssistantMessage(content="Right branch"),
    )
    compact = CompactionEntry(
        id="compact",
        parent_id="right",
        summary="The right branch was compacted.",
        replaces_entry_ids=["root", "right"],
    )
    leaf = LeafEntry(id="leaf", parent_id="compact", entry_id="compact")
    return [root, left, right, compact, leaf]


def test_html_export_preserves_tree_active_path_and_escapes_content() -> None:
    html = render_session_html(
        _branch_entries(),
        title="Test Export",
        source="/tmp/session.jsonl",
    )

    assert "<title>Test Export</title>" in html
    assert "Start &lt;session&gt;" in html
    assert 'id="entry-left"' in html
    assert 'id="entry-compact"' in html
    assert "active-path" in html
    assert "active-leaf" in html
    assert "Replaces entries: root, right" in html


def test_export_writes_html_and_exact_jsonl(tmp_path: Path) -> None:
    entries = _branch_entries()
    html_path = export_session_artifact(entries, tmp_path / "session.html")
    jsonl_path = export_session_artifact(
        entries,
        tmp_path / "session-copy.jsonl",
        format="jsonl",
    )

    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert jsonl_path.read_text(encoding="utf-8").count("\n") == len(entries)


def test_export_format_normalization_is_strict() -> None:
    assert normalize_export_format(".htm") == "html"
    assert normalize_export_format("JSONL") == "jsonl"
    with pytest.raises(SessionExportError, match="Unsupported export format"):
        normalize_export_format("markdown")
