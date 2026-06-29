"""Tests for Axis filesystem paths and AGENTS.md discovery."""

from pathlib import Path

import pytest

from axis_coding.context import (
    discover_project_context,
    discover_project_context_with_diagnostics,
)
from axis_coding.paths import AxisPaths
from axis_coding.resources import (
    AxisResourcePaths,
    find_project_root,
    resource_paths_with_cwd,
)


def test_axis_paths_describe_user_project_and_session_locations(tmp_path: Path) -> None:
    paths = AxisPaths(home=tmp_path / ".axis", agents_home=tmp_path / ".agents")
    project = tmp_path / "My Project"
    project.mkdir()

    session_path = paths.session_path(project, "session-1")

    assert paths.sessions_dir == tmp_path / ".axis" / "sessions"
    assert paths.user_skills_dir == tmp_path / ".axis" / "skills"
    assert paths.user_prompts_dir == tmp_path / ".axis" / "prompts"
    assert paths.user_agents_skills_dir == tmp_path / ".agents" / "skills"
    assert paths.project_axis_dir(project) == project / ".axis"
    assert paths.project_agents_dir(project) == project / ".agents"
    assert session_path.name == "session-1.jsonl"
    assert session_path.parent.parent == paths.sessions_dir
    assert "my-project" in session_path.parent.name
    assert session_path.exists() is False
    assert paths.home.exists() is False


def test_session_directory_hash_distinguishes_equal_project_names(tmp_path: Path) -> None:
    paths = AxisPaths(home=tmp_path / "axis-home", agents_home=tmp_path / "agents-home")
    first = tmp_path / "one" / "project"
    second = tmp_path / "two" / "project"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    assert paths.project_session_dir(first) != paths.project_session_dir(second)


def test_new_session_paths_are_unique_without_touching_filesystem(tmp_path: Path) -> None:
    paths = AxisPaths(home=tmp_path / "axis-home")

    first = paths.new_session_path(tmp_path)
    second = paths.new_session_path(tmp_path)

    assert first != second
    assert first.parent == second.parent
    assert first.exists() is False
    assert second.exists() is False


def test_session_id_cannot_escape_session_directory(tmp_path: Path) -> None:
    paths = AxisPaths(home=tmp_path / ".axis")

    with pytest.raises(ValueError, match="Invalid session id"):
        paths.session_path(tmp_path, "../escape")


def test_resource_directories_are_ordered_user_then_project(tmp_path: Path) -> None:
    home = tmp_path / "user" / ".axis"
    agents_home = tmp_path / "user" / ".agents"
    project = tmp_path / "project"
    nested = project / "src" / "package"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    paths = AxisPaths(home=home, agents_home=agents_home)
    resources = AxisResourcePaths(paths=paths, cwd=nested)

    assert resources.project_root == project
    assert resources.skills_dirs == (
        home / "skills",
        agents_home / "skills",
        project / ".axis" / "skills",
        project / ".agents" / "skills",
    )
    assert resources.prompts_dirs == (
        home / "prompts",
        agents_home / "prompts",
        project / ".axis" / "prompts",
        project / ".agents" / "prompts",
    )


def test_resource_paths_are_rebound_to_the_session_working_directory(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    session_cwd = tmp_path / "session"
    original.mkdir()
    session_cwd.mkdir()
    paths = AxisPaths(home=tmp_path / "axis-home")
    resources = AxisResourcePaths(paths=paths, cwd=original)

    rebound = resource_paths_with_cwd(resources, session_cwd)

    assert rebound.paths is paths
    assert rebound.cwd == session_cwd
    assert resources.cwd == original


def test_agents_context_is_discovered_in_increasing_precedence(tmp_path: Path) -> None:
    home = tmp_path / "user" / ".axis"
    agents_home = tmp_path / "user" / ".agents"
    project = tmp_path / "project"
    nested = project / "src" / "package"
    nested.mkdir(parents=True)
    home.mkdir(parents=True)
    agents_home.mkdir(parents=True)
    (project / ".axis").mkdir()
    (project / ".agents").mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")

    files = [
        (home / "AGENTS.md", "user axis"),
        (agents_home / "AGENTS.md", "user agents"),
        (project / "AGENTS.md", "project root"),
        (project / "src" / "AGENTS.md", "src"),
        (nested / "AGENTS.md", "nested"),
        (project / ".axis" / "AGENTS.md", "project axis"),
        (project / ".agents" / "AGENTS.md", "project agents"),
    ]
    for path, content in files:
        path.write_text(content, encoding="utf-8")

    context = discover_project_context(
        AxisResourcePaths(
            paths=AxisPaths(home=home, agents_home=agents_home),
            cwd=nested,
        )
    )

    assert [(item.path, item.content) for item in context] == files


def test_project_discovery_stops_at_nearest_marker(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    project = outer / "project"
    cwd = project / "src"
    cwd.mkdir(parents=True)
    (outer / "AGENTS.md").write_text("must not load", encoding="utf-8")
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    (project / "AGENTS.md").write_text("load", encoding="utf-8")

    resources = AxisResourcePaths(
        paths=AxisPaths(
            home=tmp_path / "missing-axis-home",
            agents_home=tmp_path / "missing-agents-home",
        ),
        cwd=cwd,
    )

    assert find_project_root(cwd) == project
    assert [item.path for item in discover_project_context(resources)] == [project / "AGENTS.md"]


def test_without_project_marker_only_cwd_agents_file_is_loaded(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    cwd = parent / "cwd"
    cwd.mkdir(parents=True)
    (parent / "AGENTS.md").write_text("outside", encoding="utf-8")
    (cwd / "AGENTS.md").write_text("inside", encoding="utf-8")

    resources = AxisResourcePaths(
        paths=AxisPaths(
            home=tmp_path / "missing-axis-home",
            agents_home=tmp_path / "missing-agents-home",
        ),
        cwd=cwd,
    )

    assert find_project_root(cwd) == cwd
    assert [item.content for item in discover_project_context(resources)] == ["inside"]


def test_unreadable_agents_file_becomes_non_fatal_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    agents_file = tmp_path / "AGENTS.md"
    agents_file.write_text("rules", encoding="utf-8")
    original_read_text = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path == agents_file.resolve():
            raise OSError("permission denied")
        return original_read_text(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", read_text)
    resources = AxisResourcePaths(
        paths=AxisPaths(
            home=tmp_path / "missing-axis-home",
            agents_home=tmp_path / "missing-agents-home",
        ),
        cwd=tmp_path,
    )

    context, diagnostics = discover_project_context_with_diagnostics(resources)

    assert context == ()
    assert len(diagnostics) == 1
    assert diagnostics[0].path == agents_file.resolve()
    assert "permission denied" in diagnostics[0].format()


def test_non_utf8_agents_file_becomes_non_fatal_diagnostic(tmp_path: Path) -> None:
    agents_file = tmp_path / "AGENTS.md"
    agents_file.write_bytes(b"\xff")
    resources = AxisResourcePaths(
        paths=AxisPaths(
            home=tmp_path / "missing-axis-home",
            agents_home=tmp_path / "missing-agents-home",
        ),
        cwd=tmp_path,
    )

    context, diagnostics = discover_project_context_with_diagnostics(resources)

    assert context == ()
    assert len(diagnostics) == 1
    assert diagnostics[0].path == agents_file.resolve()
    assert "decode" in diagnostics[0].message
