"""Append-only storage boundary for Axis session entries."""

import asyncio
import os
from pathlib import Path
from typing import Protocol

from axis_agent.session.entries import SessionEntry
from axis_agent.session.jsonl import entries_from_json_lines, entry_to_json_line


class SessionStorage(Protocol):
    """Minimal asynchronous persistence interface for session facts."""

    async def append(self, entry: SessionEntry) -> None:
        """Durably append one entry."""
        ...

    async def read_all(self) -> list[SessionEntry]:
        """Return all entries in physical storage order."""
        ...


class JsonlSessionStorage:
    """Local append-only JSONL storage with per-instance serialization."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()

    async def append(self, entry: SessionEntry) -> None:
        """Append and fsync one complete JSON row."""
        encoded = entry_to_json_line(entry).encode("utf-8")
        async with self._lock:
            await asyncio.to_thread(_append_bytes, self.path, encoded)

    async def read_all(self) -> list[SessionEntry]:
        """Read a consistent snapshot; a missing file is an empty session."""
        async with self._lock:
            lines = await asyncio.to_thread(_read_lines, self.path)
        return entries_from_json_lines(lines)


def _append_bytes(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as file:
        file.write(encoded)
        file.flush()
        os.fsync(file.fileno())


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()
