"""Tests for Axis Markdown skill discovery and expansion."""

from pathlib import Path

import pytest

from axis_coding import (
    AxisPaths,
    AxisResourcePaths,
    ResourceError,
    Skill,
    expand_skill_command,
    format_skill_invocation,
    load_skills,
    load_skills_with_diagnostics,
    parse_skill_invocation,
)


def resources(tmp_path: Path, *, cwd: Path | None = None) -> AxisResourcePaths:
    return AxisResourcePaths(
        paths=AxisPaths(
            home=tmp_path / "user" / ".axis",
            agents_home=tmp_path / "user" / ".agents",
        ),
        cwd=cwd,
    )


def test_missing_skill_directories_load_as_empty(tmp_path: Path) -> None:
    assert load_skills(resources(tmp_path)) == ()
    assert load_skills_with_diagnostics(resources(tmp_path)) == ((), ())


def test_loads_directory_and_file_skills_with_descriptions(tmp_path: Path) -> None:
    skills_dir = tmp_path / "user" / ".axis" / "skills"
    (skills_dir / "python-testing").mkdir(parents=True)
    (skills_dir / "python-testing" / "SKILL.md").write_text(
        "---\ndescription: Test Python code\n---\n# Python Testing\nUse pytest.",
        encoding="utf-8",
    )
    (skills_dir / "git-review.md").write_text("# Git Review\nReview the diff.", encoding="utf-8")
    (skills_dir / "AGENTS.md").write_text("not a skill", encoding="utf-8")

    skills = load_skills(resources(tmp_path))

    assert [skill.name for skill in skills] == ["git-review", "python-testing"]
    assert skills[0].description == "Git Review"
    assert skills[1].description == "Test Python code"
    assert skills[1].content == "# Python Testing\nUse pytest."
    assert skills[1].path == skills_dir / "python-testing" / "SKILL.md"


def test_higher_precedence_skill_overrides_case_insensitively(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    user_skill = tmp_path / "user" / ".axis" / "skills" / "Review.md"
    project_skill = project / ".agents" / "skills" / "review.md"
    user_skill.parent.mkdir(parents=True)
    project_skill.parent.mkdir(parents=True)
    user_skill.write_text("# User Review", encoding="utf-8")
    project_skill.write_text("# Project Review", encoding="utf-8")

    skills, diagnostics = load_skills_with_diagnostics(resources(tmp_path, cwd=project))

    assert len(skills) == 1
    assert skills[0].path == project_skill
    assert skills[0].description == "Project Review"
    assert len(diagnostics) == 1
    assert diagnostics[0].name == "review"
    assert str(user_skill) in diagnostics[0].message


def test_duplicate_skill_in_one_directory_is_deterministic_and_diagnostic(
    tmp_path: Path,
) -> None:
    skills_dir = tmp_path / "user" / ".axis" / "skills"
    (skills_dir / "dup").mkdir(parents=True)
    directory_skill = skills_dir / "dup" / "SKILL.md"
    directory_skill.write_text("# Directory Skill", encoding="utf-8")
    (skills_dir / "dup.md").write_text("# File Skill", encoding="utf-8")

    skills, diagnostics = load_skills_with_diagnostics(resources(tmp_path))

    assert [skill.path for skill in skills] == [directory_skill]
    assert len(diagnostics) == 1
    assert "duplicate skill name" in diagnostics[0].message
    with pytest.raises(ResourceError, match="duplicate skill name"):
        load_skills(resources(tmp_path))


def test_invalid_and_non_utf8_skills_are_reported_and_skipped(tmp_path: Path) -> None:
    skills_dir = tmp_path / "user" / ".axis" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "code review.md").write_text("invalid name", encoding="utf-8")
    (skills_dir / "broken.md").write_bytes(b"\xff")

    skills, diagnostics = load_skills_with_diagnostics(resources(tmp_path))

    assert skills == ()
    assert {diagnostic.name for diagnostic in diagnostics} == {"code review", "broken"}
    assert {diagnostic.severity for diagnostic in diagnostics} == {"error"}


def test_skill_command_expands_case_insensitively_with_reference_base(
    tmp_path: Path,
) -> None:
    skill = Skill(
        name="testing",
        path=tmp_path / "skills" / "testing" / "SKILL.md",
        content="# Testing\nRun pytest.",
        description="Test code",
    )

    expanded = expand_skill_command("/skill:TESTING add parser tests", [skill])

    assert expanded == (
        f'<skill name="testing" location="{skill.path}">\n'
        f"References are relative to {skill.path.parent}.\n\n"
        "# Testing\nRun pytest.\n"
        "</skill>\n\n"
        "add parser tests"
    )
    assert format_skill_invocation(skill).endswith("Run pytest.\n</skill>")


def test_skill_command_accepts_non_space_whitespace(tmp_path: Path) -> None:
    skill = Skill(
        name="testing",
        path=tmp_path / "SKILL.md",
        content="Run tests.",
    )

    expanded = expand_skill_command("/skill:testing\nrun only parser tests", [skill])

    assert expanded is not None
    assert expanded.endswith("</skill>\n\nrun only parser tests")


def test_expanded_skill_invocation_can_be_recovered_for_transcript_display(
    tmp_path: Path,
) -> None:
    skill = Skill(name="review", path=tmp_path / "review.md", content="# Review\nInspect code.")
    expanded = format_skill_invocation(skill, "check auth")

    invocation = parse_skill_invocation(expanded)

    assert invocation is not None
    assert invocation.name == "review"
    assert invocation.location == str(skill.path)
    assert invocation.additional_instructions == "check auth"
    assert parse_skill_invocation("ordinary prompt") is None


def test_skill_command_distinguishes_normal_missing_and_unknown_prompts() -> None:
    assert expand_skill_command("hello", []) is None
    with pytest.raises(ResourceError, match="must include a skill name"):
        expand_skill_command("/skill:", [])
    with pytest.raises(ResourceError, match="Unknown skill: missing"):
        expand_skill_command("/skill:missing", [])
