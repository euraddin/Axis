"""Pure tests for prompt autocomplete and replacement spans."""

from pathlib import Path

from rich.console import Console

from axis_coding.prompt_templates import PromptTemplate
from axis_coding.skills import Skill
from axis_coding.tui import (
    CompletionCommand,
    CompletionOption,
    build_completion_state,
    render_completion_suggestions,
)

COMMANDS = (
    CompletionCommand("new", "Start a new session", search_terms=("clear", "reset")),
    CompletionCommand("resume", "Resume a session"),
    CompletionCommand("session", "Show the active session", aliases=("status",)),
    CompletionCommand("skill", "Load a skill"),
)


def test_slash_completion_groups_commands_and_templates() -> None:
    state = build_completion_state(
        "/",
        commands=COMMANDS,
        prompt_templates=(
            PromptTemplate(
                name="review",
                path=Path("review.md"),
                content="Review this.",
                description="Review code",
            ),
        ),
    )

    assert [item.display for item in state.items] == [
        "/new",
        "/resume",
        "/session",
        "/skill:",
        "/review",
    ]
    assert [state.items[0].category, state.items[-1].category] == [
        "Commands",
        "Custom prompts",
    ]


def test_command_search_terms_use_canonical_replacement() -> None:
    state = build_completion_state("/cl", commands=COMMANDS)

    assert [item.display for item in state.items] == ["/new"]
    assert state.selected is not None
    assert state.selected.apply("/cl") == "/new"


def test_direct_command_match_precedes_search_term_match() -> None:
    state = build_completion_state("/res", commands=COMMANDS)

    assert [item.display for item in state.items[:2]] == ["/resume", "/new"]
    assert state.items[0].replacement == "/resume"


def test_skill_completion_preserves_request_suffix() -> None:
    state = build_completion_state(
        "/skill:r fix tests",
        commands=COMMANDS,
        skills=(
            Skill(
                name="review",
                path=Path("review/SKILL.md"),
                content="Review code.",
                description="Review code",
            ),
        ),
    )

    assert [item.display for item in state.items] == ["/skill:review"]
    assert state.selected is not None
    assert state.selected.apply("/skill:r fix tests") == "/skill:review fix tests"


def test_complete_skill_and_template_hide_completion_after_space() -> None:
    skill = Skill("review", Path("review.md"), "Review code")
    template = PromptTemplate("explain", Path("explain.md"), "Explain code")

    assert not build_completion_state(
        "/skill:review details",
        commands=COMMANDS,
        skills=(skill,),
    ).items
    assert not build_completion_state(
        "/explain details",
        commands=COMMANDS,
        prompt_templates=(template,),
    ).items


def test_argument_completion_replaces_only_first_argument() -> None:
    state = build_completion_state(
        "/theme axis continue",
        commands=(CompletionCommand("theme", "Change theme"),),
        argument_options={
            "theme": (
                CompletionOption("axis-dark", "Dark theme"),
                CompletionOption("high-contrast", "High contrast"),
            )
        },
    )

    assert [item.display for item in state.items] == ["axis-dark"]
    assert state.selected is not None
    assert state.selected.apply("/theme axis continue") == "/theme axis-dark continue"


def test_completion_selection_wraps() -> None:
    state = build_completion_state("/s", commands=COMMANDS)

    assert len(state.items) > 1
    assert state.select_previous().selected_index == len(state.items) - 1
    assert state.select_next().selected_index == 1


def test_file_reference_completion_is_recursive_and_ignores_hidden_dirs(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / ".axis").mkdir()
    (tmp_path / ".axis" / "secret.py").write_text("secret\n", encoding="utf-8")

    state = build_completion_state("read @app", cwd=tmp_path)

    assert [item.display for item in state.items] == ["@src/app.py"]
    assert state.selected is not None
    assert state.selected.apply("read @app") == "read @src/app.py"


def test_file_reference_completion_stays_off_for_slash_commands(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Axis\n", encoding="utf-8")

    assert not build_completion_state(
        "/skill:review @READ",
        commands=COMMANDS,
        cwd=tmp_path,
    ).items


def test_shell_path_completion_preserves_bang_prefixes(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Axis\n", encoding="utf-8")

    single = build_completion_state("!cat READ", cwd=tmp_path)
    double = build_completion_state("!!cat READ", cwd=tmp_path)

    assert [item.display for item in single.items] == ["README.md"]
    assert single.selected is not None
    assert single.selected.apply("!cat READ") == "!cat README.md"
    assert double.selected is not None
    assert double.selected.apply("!!cat READ") == "!!cat README.md"


def test_shell_completion_adds_directory_slash_and_lists_children(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    directory = build_completion_state("!cat sr", cwd=tmp_path)
    child = build_completion_state("!cat src/", cwd=tmp_path)

    assert [item.display for item in directory.items] == ["src/"]
    assert [item.display for item in child.items] == ["src/main.py"]


def test_shell_path_completion_can_be_disabled_until_execution_exists(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Axis\n", encoding="utf-8")

    state = build_completion_state(
        "!cat READ",
        cwd=tmp_path,
        shell_paths_enabled=False,
    )

    assert state.items == ()


def test_completion_rendering_marks_selection_and_keeps_categories() -> None:
    state = build_completion_state(
        "/",
        commands=COMMANDS,
        prompt_templates=(PromptTemplate("review", Path("review.md"), "Review"),),
    )
    console = Console(width=80, record=True)
    console.print(render_completion_suggestions(state))
    rendered = console.export_text()

    assert "Commands" in rendered
    assert "Custom prompts" in rendered
    assert "› /new" in rendered
