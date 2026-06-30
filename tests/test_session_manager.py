"""Tests for project-scoped Axis session indexes."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from axis_coding import AxisPaths
from axis_coding.session_manager import SessionManager


def _manager(tmp_path: Path) -> SessionManager:
    return SessionManager(AxisPaths(home=tmp_path / ".axis", agents_home=tmp_path / ".agents"))


def test_manager_creates_project_index_and_lists_sessions(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    cwd = tmp_path / "project"
    cwd.mkdir()

    record = manager.create_session(
        cwd=cwd,
        model="deepseek-v4",
        provider_name="deepseek",
        title="First",
    )

    assert record.path.name == f"{record.id}.jsonl"
    assert record.path.parent == manager.paths.project_session_dir(cwd)
    assert record.provider_name == "deepseek"
    assert manager.project_index_path(cwd).exists()
    assert not manager.index_path.exists()
    assert manager.get_session(record.id) == record
    assert manager.list_sessions(cwd) == [record]


def test_manager_filters_projects_and_sorts_by_latest_touch(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    first_cwd.mkdir()
    second_cwd.mkdir()
    older = manager.create_session(cwd=first_cwd, model="old", session_id="older")
    newer = manager.create_session(cwd=first_cwd, model="new", session_id="newer")
    other = manager.create_session(cwd=second_cwd, model="other")

    touched = manager.touch_session(older.id, model="updated", title="Renamed")

    assert touched is not None
    assert touched.model == "updated"
    assert touched.title == "Renamed"
    assert [record.id for record in manager.list_sessions(first_cwd)] == ["older", "newer"]
    assert manager.list_sessions(second_cwd) == [other]
    assert manager.latest_session_for_cwd(first_cwd) == touched
    assert {record.id for record in manager.list_sessions()} == {
        older.id,
        newer.id,
        other.id,
    }


def test_manager_gets_or_creates_one_default_session(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    cwd = tmp_path / "project"
    cwd.mkdir()

    first = manager.get_or_create_default_session(cwd=cwd, model="deepseek-v4")
    second = manager.get_or_create_default_session(cwd=cwd, model="other")

    assert first == second
    assert first.id.startswith("default-")
    assert first.path.name == "default.jsonl"


def test_manager_accepts_forward_compatible_metadata(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    cwd = tmp_path / "project"
    cwd.mkdir()
    index = manager.project_index_path(cwd)
    index.parent.mkdir(parents=True)
    index.write_text(
        json.dumps(
            {
                "id": "session-1",
                "path": str(index.parent / "session-1.jsonl"),
                "cwd": str(cwd.resolve()),
                "model": "deepseek-v4",
                "title": None,
                "created_at": 1,
                "updated_at": 2,
                "future_field": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    [record] = manager.list_sessions(cwd)

    assert record.id == "session-1"
    assert record.updated_at == 2


def test_manager_hard_fails_corrupt_index_rows(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    cwd = tmp_path / "project"
    cwd.mkdir()
    index = manager.project_index_path(cwd)
    index.parent.mkdir(parents=True)
    index.write_text('{"id":"broken"}\n', encoding="utf-8")

    with pytest.raises(ValidationError):
        manager.list_sessions(cwd)
