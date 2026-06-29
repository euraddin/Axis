"""Tests for Axis Markdown prompt-template discovery and rendering."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from axis_coding import (
    AxisPaths,
    AxisResourcePaths,
    PromptTemplate,
    ResourceError,
    expand_prompt_template_command,
    load_prompt_templates,
    load_prompt_templates_with_diagnostics,
    render_prompt_template,
)


def resources(tmp_path: Path, *, cwd: Path | None = None) -> AxisResourcePaths:
    return AxisResourcePaths(
        paths=AxisPaths(
            home=tmp_path / "user" / ".axis",
            agents_home=tmp_path / "user" / ".agents",
        ),
        cwd=cwd,
    )


def test_missing_prompt_directories_load_as_empty(tmp_path: Path) -> None:
    assert load_prompt_templates(resources(tmp_path)) == ()
    assert load_prompt_templates_with_diagnostics(resources(tmp_path)) == ((), ())


def test_loads_prompt_frontmatter_and_ignores_non_markdown(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "user" / ".axis" / "prompts"
    prompts_dir.mkdir(parents=True)
    path = prompts_dir / "review.md"
    path.write_text(
        "---\ndescription: Review code\n---\nReview {{ arguments }}.",
        encoding="utf-8",
    )
    (prompts_dir / "ignore.txt").write_text("ignored", encoding="utf-8")

    templates = load_prompt_templates(resources(tmp_path))

    assert templates == (
        PromptTemplate(
            name="review",
            path=path,
            content="Review {{ arguments }}.",
            description="Review code",
        ),
    )


def test_higher_precedence_template_overrides_case_insensitively(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    user_template = tmp_path / "user" / ".agents" / "prompts" / "Review.md"
    project_template = project / ".axis" / "prompts" / "review.md"
    user_template.parent.mkdir(parents=True)
    project_template.parent.mkdir(parents=True)
    user_template.write_text("User review", encoding="utf-8")
    project_template.write_text("Project review", encoding="utf-8")

    templates, diagnostics = load_prompt_templates_with_diagnostics(
        resources(tmp_path, cwd=project)
    )

    assert len(templates) == 1
    assert templates[0].path == project_template
    assert len(diagnostics) == 1
    assert str(user_template) in diagnostics[0].message


def test_duplicate_case_variant_in_one_directory_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompts_dir = tmp_path / "user" / ".axis" / "prompts"
    prompts_dir.mkdir(parents=True)
    first = tmp_path / "case-variants" / "first" / "Review.md"
    second = tmp_path / "case-variants" / "second" / "review.md"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    original_iterdir = Path.iterdir

    def iterdir(path: Path) -> Iterator[Path]:
        if path == prompts_dir.resolve():
            return iter((first, second))
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)

    templates, diagnostics = load_prompt_templates_with_diagnostics(resources(tmp_path))

    assert len(templates) == 1
    assert templates[0].name == "Review"
    assert len(diagnostics) == 1
    assert "duplicate prompt template name" in diagnostics[0].message
    with pytest.raises(ResourceError, match="duplicate prompt template name"):
        load_prompt_templates(resources(tmp_path))


def test_invalid_and_non_utf8_templates_are_reported_and_skipped(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "user" / ".axis" / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "code review.md").write_text("invalid", encoding="utf-8")
    (prompts_dir / "broken.md").write_bytes(b"\xff")

    templates, diagnostics = load_prompt_templates_with_diagnostics(resources(tmp_path))

    assert templates == ()
    assert {diagnostic.name for diagnostic in diagnostics} == {"code review", "broken"}
    assert {diagnostic.severity for diagnostic in diagnostics} == {"error"}


def test_render_prompt_template_is_strict_by_default() -> None:
    template = PromptTemplate(
        name="review",
        path=Path("review.md"),
        content="Review {{ target }} for {{ focus }}.",
    )

    assert render_prompt_template(template, {"target": "auth", "focus": "security"}) == (
        "Review auth for security."
    )
    with pytest.raises(ResourceError, match="Missing prompt template variable: focus"):
        render_prompt_template(template, {"target": "auth"})


@pytest.mark.parametrize(
    ("content", "command", "expected"),
    [
        ("Review {{ arguments }}.", "/review src/app.py", "Review src/app.py."),
        ("Review {{ args }}.", "/REVIEW auth.py", "Review auth.py."),
        ("Review this code.", "/review src/app.py", "Review this code.\n\nsrc/app.py"),
        (
            "Focus: {{ focus }}\nReview {{ arguments }}.",
            "/review 168",
            "Focus: \nReview 168.",
        ),
    ],
)
def test_prompt_template_command_expansion(
    content: str,
    command: str,
    expected: str,
) -> None:
    template = PromptTemplate(name="review", path=Path("review.md"), content=content)

    assert expand_prompt_template_command(command, [template]) == expected


def test_prompt_template_command_leaves_unrelated_commands_alone() -> None:
    template = PromptTemplate(name="review", path=Path("review.md"), content="Review this code.")

    assert expand_prompt_template_command("hello", [template]) is None
    assert expand_prompt_template_command("//review", [template]) is None
    assert expand_prompt_template_command("/skill:review", [template]) is None
    assert expand_prompt_template_command("/missing", [template]) is None
