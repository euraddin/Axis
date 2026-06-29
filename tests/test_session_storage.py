"""Tests for append-only Axis JSONL session storage."""

import asyncio
from pathlib import Path

import pytest

from axis_agent import (
    JsonlSessionStorage,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionJsonlError,
    SessionStorage,
    UserMessage,
    entry_to_json_line,
)


def test_jsonl_storage_appends_without_rewriting_existing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "sessions" / "session.jsonl"
    storage = JsonlSessionStorage(path)
    first = MessageEntry(
        id="first",
        timestamp=1,
        message=UserMessage(content="hello"),
    )
    second = ModelChangeEntry(
        id="second",
        parent_id="first",
        timestamp=2,
        model="deepseek-v4-pro",
    )

    asyncio.run(storage.append(first))
    original_bytes = path.read_bytes()
    asyncio.run(storage.append(second))

    assert original_bytes == entry_to_json_line(first).encode()
    assert path.read_bytes() == original_bytes + entry_to_json_line(second).encode()
    assert asyncio.run(storage.read_all()) == [first, second]


def test_jsonl_storage_missing_file_is_empty_without_creating_it(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    storage = JsonlSessionStorage(path)

    assert asyncio.run(storage.read_all()) == []
    assert path.exists() is False


def test_new_storage_instance_restores_existing_entries(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    original = JsonlSessionStorage(path)
    entry = MessageEntry(
        id="message",
        timestamp=1,
        message=UserMessage(content="persist me"),
    )
    asyncio.run(original.append(entry))

    restarted = JsonlSessionStorage(path)

    assert asyncio.run(restarted.read_all()) == [entry]


def test_concurrent_appends_produce_complete_non_interleaved_rows(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    storage = JsonlSessionStorage(path)
    entries = [
        LeafEntry(
            id=f"leaf-{index}",
            timestamp=float(index),
            entry_id=f"message-{index}",
        )
        for index in range(20)
    ]

    async def append_all() -> None:
        await asyncio.gather(*(storage.append(entry) for entry in entries))

    asyncio.run(append_all())

    restored = asyncio.run(storage.read_all())
    assert len(path.read_text(encoding="utf-8").splitlines()) == 20
    assert {entry.id for entry in restored} == {entry.id for entry in entries}


def test_storage_surfaces_corrupt_row_with_physical_line_number(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    valid = ModelChangeEntry(id="model", timestamp=1, model="deepseek-v4-pro")
    path.write_text(f"{entry_to_json_line(valid)}not-json\n", encoding="utf-8")

    with pytest.raises(SessionJsonlError, match="Invalid session entry on line 2"):
        asyncio.run(JsonlSessionStorage(path).read_all())


def test_jsonl_storage_satisfies_session_storage_protocol(tmp_path: Path) -> None:
    storage: SessionStorage = JsonlSessionStorage(tmp_path / "session.jsonl")

    assert isinstance(storage, JsonlSessionStorage)
