"""Tests for Axis session-tree validation and state replay."""

import asyncio
from pathlib import Path

import pytest

from axis_agent import (
    AssistantMessage,
    BranchSummaryEntry,
    CompactionEntry,
    JsonlSessionStorage,
    LeafEntry,
    MemoryProposalDecisionEntry,
    MemoryProposalEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionInfoEntry,
    SessionState,
    SessionTreeError,
    ThinkingLevelChangeEntry,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    infer_active_leaf,
    path_to_entry,
    validate_session_tree,
)


def _branched_entries() -> list[SessionInfoEntry | ModelChangeEntry | MessageEntry | LeafEntry]:
    return [
        SessionInfoEntry(id="info", timestamp=1, created_at=1, cwd="/workspace"),
        ModelChangeEntry(
            id="model",
            parent_id="info",
            timestamp=2,
            model="deepseek-v4-pro",
        ),
        MessageEntry(
            id="root",
            parent_id="model",
            timestamp=3,
            message=UserMessage(content="Choose"),
        ),
        MessageEntry(
            id="left",
            parent_id="root",
            timestamp=4,
            message=AssistantMessage(content="Left"),
        ),
        LeafEntry(
            id="left-pointer",
            parent_id="left",
            timestamp=5,
            entry_id="left",
        ),
        MessageEntry(
            id="right",
            parent_id="root",
            timestamp=6,
            message=AssistantMessage(content="Right"),
        ),
        LeafEntry(
            id="right-pointer",
            parent_id="right",
            timestamp=7,
            entry_id="right",
        ),
    ]


def test_path_to_entry_uses_logical_parents_not_physical_previous_rows() -> None:
    entries = _branched_entries()

    path = path_to_entry(entries, "right")

    assert [entry.id for entry in path] == ["info", "model", "root", "right"]
    assert "left" not in [entry.id for entry in path]


def test_session_state_uses_latest_leaf_pointer_and_supports_explicit_override() -> None:
    entries = _branched_entries()

    active = SessionState.from_entries(entries)
    left = SessionState.from_entries(entries, leaf_id="left")

    assert active.messages == (
        UserMessage(content="Choose"),
        AssistantMessage(content="Right"),
    )
    assert active.model == "deepseek-v4-pro"
    assert active.active_leaf_id == "right"
    assert active.context_entry_ids == ("root", "right")
    assert active.session_info == entries[0]
    assert left.messages == (
        UserMessage(content="Choose"),
        AssistantMessage(content="Left"),
    )


def test_session_state_can_select_explicit_empty_leaf() -> None:
    state = SessionState.from_entries(_branched_entries(), leaf_id=None)

    assert state.messages == ()
    assert state.model is None
    assert state.active_leaf_id is None
    assert state.session_info is not None
    assert state.entries == ()


def test_session_state_replays_full_and_partial_compaction() -> None:
    old_user = MessageEntry(id="old-user", message=UserMessage(content="Old request"))
    old_assistant = MessageEntry(
        id="old-assistant",
        parent_id="old-user",
        message=AssistantMessage(content="Old answer"),
    )
    recent = MessageEntry(
        id="recent",
        parent_id="old-assistant",
        message=UserMessage(content="Recent request"),
    )
    compact = CompactionEntry(
        id="compact",
        parent_id="recent",
        summary="Older work was summarized.",
        replaces_entry_ids=["old-user", "old-assistant"],
    )

    state = SessionState.from_entries([old_user, old_assistant, recent, compact])

    assert state.messages == (
        UserMessage(content="Previous conversation summary:\nOlder work was summarized."),
        UserMessage(content="Recent request"),
    )
    assert state.context_entry_ids == ("compact", "recent")
    assert state.compaction_entries == (compact,)


def test_session_state_tracks_pending_memory_proposals_outside_model_messages() -> None:
    root = MessageEntry(id="root", message=UserMessage(content="Implement memory"))
    proposal = MemoryProposalEntry(
        id="proposal",
        parent_id="root",
        task_type="implementation",
        target_file="progress.md",
        operation="append",
        reason="Milestone completed",
        proposed_content="- Memory implemented.",
        confidence=0.9,
        base_sha256="a" * 64,
    )
    pending = SessionState.from_entries([root, proposal])

    assert pending.messages == (UserMessage(content="Implement memory"),)
    assert pending.memory_proposals == (proposal,)
    assert pending.pending_memory_proposals == (proposal,)

    decision = MemoryProposalDecisionEntry(
        id="decision",
        parent_id="proposal",
        proposal_id="proposal",
        decision="discarded",
    )
    decided = SessionState.from_entries([root, proposal, decision])
    assert decided.messages == pending.messages
    assert decided.memory_proposal_decisions == (decision,)
    assert decided.pending_memory_proposals == ()


def test_branch_summary_replaces_earlier_path_context() -> None:
    root = MessageEntry(id="root", message=UserMessage(content="Root request"))
    answer = MessageEntry(
        id="answer",
        parent_id="root",
        message=AssistantMessage(content="Old answer"),
    )
    summary = BranchSummaryEntry(
        id="summary",
        parent_id="answer",
        branch_root_id="answer",
        summary="The abandoned branch explored another design.",
    )

    state = SessionState.from_entries([root, answer, summary], leaf_id="summary")

    assert state.messages == (
        UserMessage(
            content=(
                "The following is a summary of a branch that this conversation came back from:\n"
                "<summary>\nThe abandoned branch explored another design.\n</summary>"
            )
        ),
    )
    assert state.context_entry_ids == ("summary",)


def test_single_chain_without_leaf_pointer_is_inferred() -> None:
    entries = _branched_entries()[:4]

    assert infer_active_leaf(entries) == "left"
    assert SessionState.from_entries(entries).active_leaf_id == "left"


def test_branch_without_leaf_pointer_is_ambiguous() -> None:
    entries = [entry for entry in _branched_entries() if not isinstance(entry, LeafEntry)]

    with pytest.raises(SessionTreeError, match="multiple logical leaves"):
        SessionState.from_entries(entries)


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (
            [
                MessageEntry(id="same", timestamp=1, message=UserMessage(content="a")),
                MessageEntry(id="same", timestamp=2, message=UserMessage(content="b")),
            ],
            "Duplicate session entry id",
        ),
        (
            [
                MessageEntry(
                    id="child",
                    parent_id="missing",
                    timestamp=1,
                    message=UserMessage(content="a"),
                )
            ],
            "Missing parent session entry",
        ),
        (
            [
                MessageEntry(
                    id="a",
                    parent_id="b",
                    timestamp=1,
                    message=UserMessage(content="a"),
                ),
                MessageEntry(
                    id="b",
                    parent_id="a",
                    timestamp=2,
                    message=AssistantMessage(content="b"),
                ),
            ],
            "Cycle detected",
        ),
        (
            [
                MessageEntry(id="a", timestamp=1, message=UserMessage(content="a")),
                MessageEntry(id="b", timestamp=2, message=UserMessage(content="b")),
            ],
            "exactly one logical root",
        ),
        (
            [LeafEntry(id="pointer", timestamp=1, entry_id="missing")],
            "Missing active leaf target",
        ),
        (
            [
                LeafEntry(id="pointer", timestamp=1, entry_id="target"),
                MessageEntry(
                    id="target",
                    timestamp=2,
                    message=UserMessage(content="later"),
                ),
            ],
            "must appear before pointer",
        ),
        (
            [
                MessageEntry(
                    id="root",
                    timestamp=1,
                    message=UserMessage(content="root"),
                ),
                LeafEntry(id="first-pointer", timestamp=2, entry_id="root"),
                LeafEntry(
                    id="second-pointer",
                    timestamp=3,
                    entry_id="first-pointer",
                ),
            ],
            "cannot target another LeafEntry",
        ),
        (
            [
                MessageEntry(
                    id="root",
                    timestamp=1,
                    message=UserMessage(content="root"),
                ),
                LeafEntry(
                    id="pointer",
                    parent_id="root",
                    timestamp=2,
                    entry_id="root",
                ),
                MessageEntry(
                    id="child",
                    parent_id="pointer",
                    timestamp=3,
                    message=AssistantMessage(content="invalid"),
                ),
            ],
            "cannot use LeafEntry",
        ),
    ],
)
def test_tree_validation_rejects_corrupt_structure(
    entries: list[MessageEntry | LeafEntry],
    message: str,
) -> None:
    with pytest.raises(SessionTreeError, match=message):
        validate_session_tree(entries)


def test_tree_validation_rejects_parent_appended_after_child() -> None:
    entries = [
        MessageEntry(
            id="child",
            parent_id="parent",
            timestamp=1,
            message=AssistantMessage(content="child"),
        ),
        MessageEntry(
            id="parent",
            timestamp=2,
            message=UserMessage(content="parent"),
        ),
    ]

    with pytest.raises(SessionTreeError, match="must appear before child"):
        validate_session_tree(entries)


def test_process_restart_restores_exact_transcript_and_provider_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.jsonl"
    storage = JsonlSessionStorage(path)
    tool_call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    entries = [
        SessionInfoEntry(
            id="info",
            timestamp=1,
            created_at=1,
            cwd=str(tmp_path),
        ),
        ModelChangeEntry(
            id="model",
            parent_id="info",
            timestamp=2,
            model="deepseek-v4-pro",
        ),
        MessageEntry(
            id="user",
            parent_id="model",
            timestamp=3,
            message=UserMessage(content="Read README"),
        ),
        MessageEntry(
            id="tool-request",
            parent_id="user",
            timestamp=4,
            message=AssistantMessage(
                tool_calls=[tool_call],
                provider_data={"reasoning_content": "I should read the file."},
            ),
        ),
        MessageEntry(
            id="tool-result",
            parent_id="tool-request",
            timestamp=5,
            message=ToolResultMessage(
                tool_call_id="call-1",
                name="read",
                content="README contents",
                data={"path": "README.md"},
            ),
        ),
        MessageEntry(
            id="final",
            parent_id="tool-result",
            timestamp=6,
            message=AssistantMessage(content="Done"),
        ),
        LeafEntry(
            id="pointer",
            parent_id="final",
            timestamp=7,
            entry_id="final",
        ),
    ]

    async def persist() -> None:
        for entry in entries:
            await storage.append(entry)

    asyncio.run(persist())

    restarted_storage = JsonlSessionStorage(path)
    state = asyncio.run(SessionState.from_storage(restarted_storage))

    assert state.messages == tuple(
        entry.message for entry in entries if isinstance(entry, MessageEntry)
    )
    assert state.model == "deepseek-v4-pro"
    assert state.active_leaf_id == "final"
    assert state.context_entry_ids == (
        "user",
        "tool-request",
        "tool-result",
        "final",
    )
    restored_tool_request = state.messages[1]
    assert isinstance(restored_tool_request, AssistantMessage)
    assert restored_tool_request.provider_data == {"reasoning_content": "I should read the file."}


def test_session_state_replays_thinking_level_on_active_path() -> None:
    root = MessageEntry(id="root", message=UserMessage(content="Hello"))
    thinking = ThinkingLevelChangeEntry(
        id="thinking",
        parent_id="root",
        thinking_level="high",
    )

    state = SessionState.from_entries([root, thinking, LeafEntry(entry_id="thinking")])

    assert state.thinking_level == "high"


def test_branch_summary_replaces_messages_but_preserves_model_metadata() -> None:
    model = ModelChangeEntry(id="model", model="deepseek-v4-fast")
    thinking = ThinkingLevelChangeEntry(
        id="thinking",
        parent_id="model",
        thinking_level="high",
    )
    abandoned = MessageEntry(
        id="abandoned",
        parent_id="thinking",
        message=UserMessage(content="Old direction"),
    )
    summary = BranchSummaryEntry(
        id="summary",
        parent_id="abandoned",
        summary="Preserved branch context.",
    )

    state = SessionState.from_entries(
        [model, thinking, abandoned, summary, LeafEntry(entry_id="summary")]
    )

    assert state.model == "deepseek-v4-fast"
    assert state.thinking_level == "high"
    assert len(state.messages) == 1
    assert "Preserved branch context." in state.messages[0].content
