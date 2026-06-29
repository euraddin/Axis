"""Pure JSON Lines codec for typed Axis session entries."""

from collections.abc import Iterable

from pydantic import TypeAdapter, ValidationError

from axis_agent.session.entries import SessionEntry

_SESSION_ENTRY_ADAPTER: TypeAdapter[SessionEntry] = TypeAdapter(SessionEntry)


class SessionJsonlError(ValueError):
    """A JSONL row could not be decoded as a valid session entry."""


def entry_to_json_line(entry: SessionEntry) -> str:
    """Serialize one typed entry as exactly one newline-terminated JSON row."""
    return _SESSION_ENTRY_ADAPTER.dump_json(entry).decode() + "\n"


def entry_from_json_line(
    line: str,
    *,
    line_number: int | None = None,
) -> SessionEntry:
    """Decode one JSON row and preserve its concrete entry type."""
    try:
        return _SESSION_ENTRY_ADAPTER.validate_json(line)
    except ValidationError as exc:
        location = f" on line {line_number}" if line_number is not None else ""
        raise SessionJsonlError(f"Invalid session entry{location}: {exc}") from exc


def entries_from_json_lines(lines: Iterable[str]) -> list[SessionEntry]:
    """Decode non-empty rows in source order."""
    entries: list[SessionEntry] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        entries.append(entry_from_json_line(line, line_number=line_number))
    return entries
