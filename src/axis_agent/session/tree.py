"""Validation and traversal for append-only Axis session trees."""

from collections.abc import Sequence

from axis_agent.session.entries import LeafEntry, SessionEntry


class SessionTreeError(ValueError):
    """Session entries do not form a valid traversable tree."""


def entries_by_id(entries: Sequence[SessionEntry]) -> dict[str, SessionEntry]:
    """Index entries by id while rejecting duplicates."""
    result: dict[str, SessionEntry] = {}
    for entry in entries:
        if entry.id in result:
            raise SessionTreeError(f"Duplicate session entry id: {entry.id}")
        result[entry.id] = entry
    return result


def validate_session_tree(entries: Sequence[SessionEntry]) -> None:
    """Validate references, chronology, cycles and one logical root."""
    by_id = entries_by_id(entries)
    positions = {entry.id: index for index, entry in enumerate(entries)}

    for entry in entries:
        if entry.parent_id is not None and entry.parent_id not in by_id:
            raise SessionTreeError(f"Missing parent session entry {entry.parent_id} for {entry.id}")
        if isinstance(entry, LeafEntry) and entry.entry_id is not None:
            target = by_id.get(entry.entry_id)
            if target is None:
                raise SessionTreeError(f"Missing active leaf target: {entry.entry_id}")
            if isinstance(target, LeafEntry):
                raise SessionTreeError("A LeafEntry cannot target another LeafEntry")

    _reject_cycles(entries, by_id)

    for entry in entries:
        if entry.parent_id is not None:
            parent = by_id[entry.parent_id]
            if not isinstance(entry, LeafEntry) and isinstance(parent, LeafEntry):
                raise SessionTreeError(
                    f"Logical entry {entry.id} cannot use LeafEntry {parent.id} as parent"
                )
            if positions[parent.id] >= positions[entry.id]:
                raise SessionTreeError(f"Parent {parent.id} must appear before child {entry.id}")
        if (
            isinstance(entry, LeafEntry)
            and entry.entry_id is not None
            and positions[entry.entry_id] >= positions[entry.id]
        ):
            raise SessionTreeError(
                f"Active leaf target {entry.entry_id} must appear before pointer {entry.id}"
            )

    logical_entries = [entry for entry in entries if not isinstance(entry, LeafEntry)]
    roots = [entry for entry in logical_entries if entry.parent_id is None]
    if logical_entries and len(roots) != 1:
        raise SessionTreeError(
            f"Session tree must have exactly one logical root; found {len(roots)}"
        )


def path_to_entry(
    entries: Sequence[SessionEntry],
    leaf_id: str | None,
) -> list[SessionEntry]:
    """Return the validated logical root-to-leaf path."""
    validate_session_tree(entries)
    if leaf_id is None:
        return []

    by_id = entries_by_id(entries)
    target = by_id.get(leaf_id)
    if target is None:
        raise SessionTreeError(f"Missing session entry: {leaf_id}")
    if isinstance(target, LeafEntry):
        raise SessionTreeError(f"Cannot replay from LeafEntry pointer: {leaf_id}")

    path: list[SessionEntry] = []
    current: SessionEntry | None = target
    while current is not None:
        path.append(current)
        current = by_id[current.parent_id] if current.parent_id is not None else None
    path.reverse()
    return path


def infer_active_leaf(entries: Sequence[SessionEntry]) -> str | None:
    """Infer the only logical tip when no explicit LeafEntry exists."""
    validate_session_tree(entries)
    logical_entries = [entry for entry in entries if not isinstance(entry, LeafEntry)]
    if not logical_entries:
        return None

    parent_ids = {entry.parent_id for entry in logical_entries if entry.parent_id is not None}
    tips = [entry.id for entry in logical_entries if entry.id not in parent_ids]
    if len(tips) != 1:
        raise SessionTreeError(
            "Session tree has multiple logical leaves but no active LeafEntry pointer"
        )
    return tips[0]


def _reject_cycles(
    entries: Sequence[SessionEntry],
    by_id: dict[str, SessionEntry],
) -> None:
    for entry in entries:
        path_ids: set[str] = set()
        current: SessionEntry | None = entry
        while current is not None:
            if current.id in path_ids:
                raise SessionTreeError(f"Cycle detected at session entry: {current.id}")
            path_ids.add(current.id)
            current = by_id.get(current.parent_id) if current.parent_id is not None else None
