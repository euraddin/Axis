from pathlib import Path

from axis_agent import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from axis_coding.voice import build_voice_context_snapshot


def test_voice_context_is_bounded_and_excludes_sensitive_tool_data() -> None:
    messages = [
        UserMessage(content="Previous conversation summary:\nWe are fixing ParserService."),
        UserMessage(content="Use src/parser.py and API_KEY=super-secret-value"),
        AssistantMessage(
            content="I will update parse_document.",
            tool_calls=[
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"path": "src/parser.py", "content": "PRIVATE FILE BODY"},
                ),
                ToolCall(
                    id="bash-1",
                    name="bash",
                    arguments={"command": "curl -H 'Authorization: Bearer secret'"},
                ),
            ],
            provider_data={"reasoning_content": "PRIVATE REASONING"},
        ),
        ToolResultMessage(
            tool_call_id="write-1",
            name="write",
            content="PRIVATE TOOL OUTPUT",
            ok=True,
        ),
        ToolResultMessage(
            tool_call_id="bash-1",
            name="bash",
            content="PRIVATE STDOUT",
            ok=False,
            error="PRIVATE STDERR",
        ),
    ]
    snapshot = build_voice_context_snapshot(
        messages=messages,
        editor_text="Please adjust parse_document here",
        cursor=14,
        cwd=Path("/repo"),
        session_title="Parser repair",
        skill_names=("python-review",),
        git_branch="voice-input",
    )
    combined = "\n".join(
        (snapshot.editor_context, snapshot.session_memory, snapshot.coding_context)
    )
    assert snapshot.character_count <= 8_000
    assert "ParserService" in combined
    assert "parse_document" in combined
    assert "src/parser.py" in combined
    assert "PRIVATE FILE BODY" not in combined
    assert "PRIVATE TOOL OUTPUT" not in combined
    assert "PRIVATE STDOUT" not in combined
    assert "PRIVATE STDERR" not in combined
    assert "PRIVATE REASONING" not in combined
    assert "super-secret-value" not in combined
    assert "[REDACTED]" in combined
    assert "bash: failed" in snapshot.tool_activity


def test_voice_context_keeps_recent_messages_and_hard_total_limit() -> None:
    messages = [UserMessage(content=f"message-{index} " + "x" * 2_000) for index in range(10)]
    snapshot = build_voice_context_snapshot(
        messages=messages,
        editor_text="y" * 5_000,
        cursor=2_500,
        cwd=Path("/repo"),
        git_branch="main",
    )
    assert snapshot.character_count <= 8_000
    assert "message-9" in snapshot.recent_dialogue
    assert "message-0" not in snapshot.recent_dialogue
    assert len(snapshot.editor_before) == 1_000
    assert len(snapshot.editor_after) == 1_000
