"""Reconstruct current Axis session state from append-only facts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast, overload

from axis_agent.messages import AgentMessage
from axis_agent.session.entries import (
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionEntry,
    SessionInfoEntry,
)
from axis_agent.session.storage import SessionStorage
from axis_agent.session.tree import infer_active_leaf, path_to_entry, validate_session_tree

_UNSET_LEAF_ID: Final[object] = object()


@dataclass(frozen=True, slots=True)
class SessionState:
    """Current model context derived from a validated session tree."""

    messages: tuple[AgentMessage, ...]
    model: str | None
    active_leaf_id: str | None
    session_info: SessionInfoEntry | None
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

        replay_entries = path_to_entry(entries, active_leaf_id)
        messages: list[AgentMessage] = []
        context_entry_ids: list[str] = []
        model: str | None = None
        for entry in replay_entries:
            if isinstance(entry, MessageEntry):
                messages.append(entry.message)
                context_entry_ids.append(entry.id)
            elif isinstance(entry, ModelChangeEntry):
                model = entry.model

        session_info = next(
            (entry for entry in reversed(entries) if isinstance(entry, SessionInfoEntry)),
            None,
        )
        return cls(
            messages=tuple(messages),
            model=model,
            active_leaf_id=active_leaf_id,
            session_info=session_info,
            context_entry_ids=tuple(context_entry_ids),
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
