"""Reconstruct current Axis session state from append-only facts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast, overload

from axis_agent.messages import AgentMessage, UserMessage
from axis_agent.session.entries import (
    BranchSummaryEntry,
    CompactionEntry,
    LeafEntry,
    MemoryProposalDecisionEntry,
    MemoryProposalEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionEntry,
    SessionInfoEntry,
    ThinkingLevelChangeEntry,
)
from axis_agent.session.storage import SessionStorage
from axis_agent.session.tree import infer_active_leaf, path_to_entry, validate_session_tree

_UNSET_LEAF_ID: Final[object] = object()


@dataclass(frozen=True, slots=True)
class SessionState:
    """Current model context derived from a validated session tree."""

    messages: tuple[AgentMessage, ...]
    model: str | None
    thinking_level: str | None
    active_leaf_id: str | None
    session_info: SessionInfoEntry | None
    compaction_entries: tuple[CompactionEntry, ...]
    memory_proposals: tuple[MemoryProposalEntry, ...]
    memory_proposal_decisions: tuple[MemoryProposalDecisionEntry, ...]
    pending_memory_proposals: tuple[MemoryProposalEntry, ...]
    context_entry_ids: tuple[str, ...]
    entries: tuple[SessionEntry, ...]

    @classmethod
    @overload
    def from_entries(cls, entries: Sequence[SessionEntry]) -> SessionState: ...

    @classmethod
    @overload
    def from_entries(
        cls,
        entries: Sequence[SessionEntry],
        *,
        leaf_id: str | None,
    ) -> SessionState: ...

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[SessionEntry],
        *,
        leaf_id: str | None | object = _UNSET_LEAF_ID,
    ) -> SessionState:
        """Validate entries and replay the selected root-to-leaf path."""
        validate_session_tree(entries)
        if leaf_id is _UNSET_LEAF_ID:
            pointer = _latest_leaf_pointer(entries)
            active_leaf_id = pointer.entry_id if pointer is not None else infer_active_leaf(entries)
        else:
            active_leaf_id = cast(str | None, leaf_id)

        full_replay_entries = path_to_entry(entries, active_leaf_id)
        model = next(
            (
                entry.model
                for entry in reversed(full_replay_entries)
                if isinstance(entry, ModelChangeEntry)
            ),
            None,
        )
        thinking_level = next(
            (
                entry.thinking_level
                for entry in reversed(full_replay_entries)
                if isinstance(entry, ThinkingLevelChangeEntry)
            ),
            None,
        )
        replay_entries = full_replay_entries
        memory_proposals = tuple(
            entry for entry in full_replay_entries if isinstance(entry, MemoryProposalEntry)
        )
        memory_proposal_decisions = tuple(
            entry for entry in full_replay_entries if isinstance(entry, MemoryProposalDecisionEntry)
        )
        decided_proposal_ids = {entry.proposal_id for entry in memory_proposal_decisions}
        pending_memory_proposals = tuple(
            entry for entry in memory_proposals if entry.id not in decided_proposal_ids
        )
        latest_branch_summary_index = _latest_branch_summary_index(replay_entries)
        if latest_branch_summary_index is not None:
            replay_entries = replay_entries[latest_branch_summary_index:]

        message_rows: list[tuple[str, AgentMessage]] = []
        compaction_entries: list[CompactionEntry] = []
        for entry in replay_entries:
            if isinstance(entry, MessageEntry):
                message_rows.append((entry.id, entry.message))
            elif isinstance(entry, CompactionEntry):
                compaction_entries.append(entry)
                message_rows = _apply_compaction(message_rows, entry)
            elif isinstance(entry, BranchSummaryEntry):
                message_rows.append(
                    (
                        entry.id,
                        UserMessage(content=_format_branch_summary(entry.summary)),
                    )
                )

        session_info = next(
            (entry for entry in reversed(entries) if isinstance(entry, SessionInfoEntry)),
            None,
        )
        return cls(
            messages=tuple(message for _entry_id, message in message_rows),
            model=model,
            thinking_level=thinking_level,
            active_leaf_id=active_leaf_id,
            session_info=session_info,
            compaction_entries=tuple(compaction_entries),
            memory_proposals=memory_proposals,
            memory_proposal_decisions=memory_proposal_decisions,
            pending_memory_proposals=pending_memory_proposals,
            context_entry_ids=tuple(entry_id for entry_id, _message in message_rows),
            entries=tuple(replay_entries),
        )

    @classmethod
    @overload
    async def from_storage(cls, storage: SessionStorage) -> SessionState: ...

    @classmethod
    @overload
    async def from_storage(
        cls,
        storage: SessionStorage,
        *,
        leaf_id: str | None,
    ) -> SessionState: ...

    @classmethod
    async def from_storage(
        cls,
        storage: SessionStorage,
        *,
        leaf_id: str | None | object = _UNSET_LEAF_ID,
    ) -> SessionState:
        """Read all persisted facts and replay them into current state."""
        entries = await storage.read_all()
        if leaf_id is _UNSET_LEAF_ID:
            return cls.from_entries(entries)
        return cls.from_entries(entries, leaf_id=cast(str | None, leaf_id))


def _latest_leaf_pointer(entries: Sequence[SessionEntry]) -> LeafEntry | None:
    return next(
        (entry for entry in reversed(entries) if isinstance(entry, LeafEntry)),
        None,
    )


def _latest_branch_summary_index(entries: Sequence[SessionEntry]) -> int | None:
    return next(
        (
            index
            for index in range(len(entries) - 1, -1, -1)
            if isinstance(entries[index], BranchSummaryEntry)
        ),
        None,
    )


def _apply_compaction(
    rows: list[tuple[str, AgentMessage]],
    entry: CompactionEntry,
) -> list[tuple[str, AgentMessage]]:
    replaced = set(entry.replaces_entry_ids)
    retained: list[tuple[str, AgentMessage]] = []
    inserted = False
    for entry_id, message in rows:
        if entry_id not in replaced:
            retained.append((entry_id, message))
            continue
        if not inserted:
            retained.append(
                (
                    entry.id,
                    UserMessage(content=f"Previous conversation summary:\n{entry.summary}"),
                )
            )
            inserted = True
    if not inserted:
        retained.append(
            (
                entry.id,
                UserMessage(content=f"Previous conversation summary:\n{entry.summary}"),
            )
        )
    return retained


def _format_branch_summary(summary: str) -> str:
    return (
        "The following is a summary of a branch that this conversation came back from:\n"
        f"<summary>\n{summary}\n</summary>"
    )
