"""Tests for Tau-compatible semantic transcript reduction."""

from pathlib import Path

from axis_agent import (
    AgentEndEvent,
    AgentStartEvent,
    AgentToolResult,
    AssistantMessage,
    ContextCompactionEvent,
    ErrorEvent,
    MemoryContextEvent,
    MemoryProposalEvent,
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
    ToolResultMessage,
    TurnEndEvent,
    TurnStartEvent,
    UserMessage,
)
from axis_coding import Skill, format_skill_invocation
from axis_coding.tui import (
    ChatItem,
    TuiEventAdapter,
    TuiState,
    format_tool_call_block,
    format_tool_result_block,
)


def test_adapter_tracks_run_turn_and_queue_state() -> None:
    adapter = TuiEventAdapter()
    state = adapter.apply(AgentStartEvent())
    adapter.apply(TurnStartEvent(turn=2))
    adapter.apply(QueueUpdateEvent(steering=("adjust",), follow_up=("after",)))

    assert state.running is True
    assert state.current_turn == 2
    assert state.queued_message_count == 2

    adapter.apply(TurnEndEvent(turn=2))
    adapter.apply(AgentEndEvent())
    assert state.running is False
    assert state.current_turn is None


def test_adapter_reports_automatic_context_compaction() -> None:
    adapter = TuiEventAdapter()

    state = adapter.apply(
        ContextCompactionEvent(
            before_tokens=100_000,
            after_tokens=24_000,
            trigger_tokens=96_000,
            compacted_entries=12,
            retained_entries=4,
        )
    )

    assert state.items == [
        ChatItem(
            role="status",
            text="Auto-compacted context (100000 → 24000 tokens; kept 4 entries).",
        )
    ]


def test_adapter_reports_memory_warnings_and_generated_proposals() -> None:
    adapter = TuiEventAdapter()

    adapter.apply(
        MemoryContextEvent(
            task_type="debug",
            warnings=("pitfalls.md is missing",),
        )
    )
    adapter.apply(
        MemoryProposalEvent(
            status="generated",
            proposal_ids=("proposal-1",),
            message="Generated 1 memory proposal(s). Run /memory review.",
        )
    )

    assert [(item.role, item.text) for item in adapter.state.items] == [
        ("status", "Memory warning: pitfalls.md is missing"),
        (
            "status",
            "Auto Memory: Generated 1 memory proposal(s). Run /memory review.",
        ),
    ]


def test_adapter_commits_streamed_assistant_without_duplication() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(MessageStartEvent())
    adapter.apply(MessageDeltaEvent(delta="Hel"))
    adapter.apply(MessageDeltaEvent(delta="lo"))
    assert state.assistant_buffer == "Hello"

    adapter.apply(MessageEndEvent(message=AssistantMessage(content="Hello")))

    assert state.assistant_buffer == ""
    assert state.items == [ChatItem(role="assistant", text="Hello")]


def test_adapter_compacts_expanded_skill_user_message() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)
    skill = Skill(
        name="review",
        path=Path("/workspace/.axis/skills/review.md"),
        content="# Review\nFull instructions.",
    )

    adapter.apply(
        MessageEndEvent(message=UserMessage(content=format_skill_invocation(skill, "check auth")))
    )

    assert [(item.role, item.text) for item in state.items] == [
        ("skill", "Using skill: review"),
        ("user", "check auth"),
    ]


def test_adapter_groups_adjacent_thinking_deltas() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(ThinkingDeltaEvent(delta="hidden "))
    adapter.apply(ThinkingDeltaEvent(delta="reasoning"))

    assert state.items == [ChatItem(role="thinking", text="hidden reasoning")]
    assert state.show_thinking is False


def test_tool_start_flushes_partial_assistant_and_uses_human_invocation() -> None:
    state = TuiState(assistant_buffer="Before tool")
    adapter = TuiEventAdapter(state)

    adapter.apply(
        ToolExecutionStartEvent(
            tool_call=ToolCall(
                id="call-1",
                name="read",
                arguments={"path": "README.md", "offset": 1, "limit": 80},
            )
        )
    )

    assert [(item.role, item.text) for item in state.items] == [
        ("assistant", "Before tool"),
        ("tool", "→ read README.md:1-80"),
    ]
    assert state.active_tool_count == 1


def test_tool_approval_request_and_execution_start_share_one_item() -> None:
    call = ToolCall(id="call-1", name="bash", arguments={"command": "echo hello"})
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(ToolApprovalRequestEvent(tool_call=call))
    adapter.apply(ToolApprovalResolvedEvent(tool_call_id=call.id, decision="allow_once"))
    adapter.apply(ToolExecutionStartEvent(tool_call=call))

    assert [(item.role, item.text, item.tool_call_id) for item in state.items] == [
        ("tool", "$ echo hello", "call-1")
    ]


def test_skill_file_read_uses_skill_role_and_attaches_result() -> None:
    skill = Skill(
        name="review",
        path=Path("/workspace/.axis/skills/review.md"),
        content="# Review",
    )
    state = TuiState(skills=(skill,))
    adapter = TuiEventAdapter(state)
    adapter.apply(
        ToolExecutionStartEvent(
            tool_call=ToolCall(
                id="call-1",
                name="read",
                arguments={"path": str(skill.path)},
            )
        )
    )
    adapter.apply(
        ToolExecutionEndEvent(
            result=AgentToolResult(
                tool_call_id="call-1",
                name="read",
                ok=True,
                content="# Review\nInstructions",
            )
        )
    )

    assert [(item.role, item.text, item.tool_result_text) for item in state.items] == [
        ("skill", "Loading skill: review", "✓ read\n# Review\nInstructions")
    ]
    assert state.active_tool_count == 0


def test_tool_updates_and_orphan_results_remain_visible() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(ToolExecutionUpdateEvent(tool_call_id="call-1", message="reading"))
    adapter.apply(
        ToolExecutionEndEvent(
            result=AgentToolResult(
                tool_call_id="missing",
                name="bash",
                ok=False,
                content="failed",
            )
        )
    )

    assert [(item.role, item.text, item.tool_result_text) for item in state.items] == [
        ("tool", "… reading", None),
        ("tool", "✗ bash", "✗ bash\nfailed"),
    ]


def test_retry_and_queue_events_are_display_state() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(
        RetryEvent(
            attempt=2,
            max_attempts=3,
            delay_seconds=0,
            message="Retrying after HTTP 503",
        )
    )
    adapter.apply(QueueUpdateEvent(steering=("adjust",), follow_up=("after",)))

    assert state.items == [ChatItem(role="status", text="… Retrying after HTTP 503")]
    assert state.queued_message_count == 2


def test_tool_blocks_bound_content_and_include_edit_patch() -> None:
    content = "\n".join(f"line {index}" for index in range(1, 12))
    block = format_tool_result_block(name="read", ok=True, content=content)

    assert "line 8" in block
    assert "line 9" not in block
    assert "3 more lines" in block

    patch_block = format_tool_result_block(
        name="edit",
        ok=True,
        content="Successfully replaced 1 block.",
        data={"patch": "--- a.py\n+++ a.py\n@@\n-old\n+new"},
    )
    assert "Patch:\n--- a.py\n+++ a.py" in patch_block


def test_known_tool_call_blocks_match_tau_shape() -> None:
    assert (
        format_tool_call_block(
            ToolCall(id="1", name="edit", arguments={"path": "src/axis_coding/tui/app.py"})
        )
        == "→ edit src/axis_coding/tui/app.py"
    )
    assert (
        format_tool_call_block(
            ToolCall(id="2", name="bash", arguments={"command": "pytest", "timeout": 30})
        )
        == "$ pytest (timeout 30s)"
    )


def test_non_recoverable_error_flushes_text_and_stops() -> None:
    state = TuiState(running=True, assistant_buffer="partial")
    adapter = TuiEventAdapter(state)

    adapter.apply(ErrorEvent(message="provider failed", recoverable=False))

    assert state.running is False
    assert state.error == "provider failed"
    assert [(item.role, item.text) for item in state.items] == [
        ("assistant", "partial"),
        ("error", "Error: provider failed"),
    ]


def test_cancellation_is_status_not_terminal_error() -> None:
    state = TuiState(running=True)
    adapter = TuiEventAdapter(state)

    adapter.apply(ErrorEvent(message="Agent run cancelled", recoverable=True))

    assert state.running is True
    assert state.cancelled is True
    assert state.error is None
    assert state.items == [ChatItem(role="status", text="Agent run cancelled.")]


def test_load_messages_reconstructs_tool_pair_and_expanded_skill() -> None:
    skill = Skill(name="review", path=Path("/skills/review.md"), content="# Review")
    call = ToolCall(id="call-1", name="edit", arguments={"path": "README.md"})
    state = TuiState(skills=(skill,))

    state.load_messages(
        [
            UserMessage(content=format_skill_invocation(skill, "review README")),
            AssistantMessage(content="Inspecting", tool_calls=[call]),
            ToolResultMessage(
                tool_call_id="call-1",
                name="edit",
                content="Changed",
                data={"patch": "--- README.md\n+++ README.md"},
            ),
        ]
    )

    assert [(item.role, item.text) for item in state.items] == [
        ("skill", "Using skill: review"),
        ("user", "review README"),
        ("assistant", "Inspecting"),
        ("tool", "→ edit README.md"),
    ]
    assert state.items[-1].tool_result_text is not None
    assert "Patch:" in state.items[-1].tool_result_text
