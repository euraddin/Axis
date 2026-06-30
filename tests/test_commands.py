"""Tests for the Axis slash-command registry."""

from pathlib import Path

import pytest

from axis_coding import (
    AxisPaths,
    CodingReloadSummary,
    CommandRegistry,
    CommandResult,
    ReloadCategorySummary,
    SessionManager,
    SlashCommand,
    create_default_command_registry,
    format_reload_summary,
)
from axis_coding.commands import BUILTIN_COMMAND_THEME_NAMES
from axis_coding.tui.config import BUILTIN_TUI_THEME_NAMES


class FakeCommandSession:
    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd
        self.model = "deepseek-v4"
        self.provider_name = "deepseek"
        self.available_models = ("deepseek-v4", "deepseek-v4-pro")
        self.thinking_level = "high"
        self.available_thinking_levels = ("high", "xhigh")
        self.thinking_unavailable_reason = None
        self.tools = (object(), object(), object(), object())
        self.skills = (object(),)
        self.prompt_templates = (object(), object())
        self.context_files = (object(),)
        self.resource_diagnostics = ()
        self.messages = (object(), object(), object())
        self.session_id = None
        self.session_title = None
        self.session_manager = None

    def reload_provider_settings(self) -> None:
        return None


def _handled(_context: object) -> CommandResult:
    return CommandResult(handled=True)


def test_registry_ignores_prompts_templates_and_skill_expansion(tmp_path: Path) -> None:
    registry = create_default_command_registry()
    session = FakeCommandSession(tmp_path)

    assert registry.execute(session, "hello").handled is False  # type: ignore[arg-type]
    assert registry.execute(session, "/skill:review this").handled is False  # type: ignore[arg-type]
    assert registry.execute(session, "/").handled is False  # type: ignore[arg-type]


def test_default_registry_only_advertises_implemented_commands() -> None:
    commands = create_default_command_registry().list_commands()

    assert [command.name for command in commands] == [
        "compact",
        "export",
        "hotkeys",
        "login",
        "logout",
        "model",
        "name",
        "new",
        "quit",
        "reload",
        "resume",
        "scoped-models",
        "session",
        "skill",
        "theme",
        "thinking",
        "tree",
    ]


def test_registry_rejects_duplicate_names_and_aliases() -> None:
    registry = CommandRegistry()
    registry.register(SlashCommand("one", "One", "/one", _handled, aliases=("first",)))

    with pytest.raises(ValueError, match="Duplicate slash command"):
        registry.register(SlashCommand("one", "Again", "/one", _handled))
    with pytest.raises(ValueError, match="Duplicate slash command alias"):
        registry.register(SlashCommand("two", "Two", "/two", _handled, aliases=("first",)))


def test_unknown_tau_commands_are_not_silently_sent_to_the_model(tmp_path: Path) -> None:
    registry = create_default_command_registry()
    session = FakeCommandSession(tmp_path)

    for text in ("/help", "/clear", "/status"):
        result = registry.execute(session, text)  # type: ignore[arg-type]
        assert result.handled is True
        assert result.message == f"Unknown command: {text}"


def test_session_command_reports_only_current_axis_capabilities(tmp_path: Path) -> None:
    result = create_default_command_registry().execute(  # type: ignore[arg-type]
        FakeCommandSession(tmp_path),
        "/session",
    )

    assert result.message is not None
    assert "Model: deepseek:deepseek-v4" in result.message
    assert "Thinking mode: high" in result.message
    assert f"CWD: {tmp_path}" in result.message
    assert "Tools: 4" in result.message
    assert "Skills: 1" in result.message
    assert "Prompt templates: 2" in result.message
    assert "Context files: 1" in result.message
    assert "Messages: 3" in result.message
    assert "Resource diagnostics: 0" in result.message


def test_hotkeys_and_skill_commands_return_guidance(tmp_path: Path) -> None:
    registry = create_default_command_registry()
    session = FakeCommandSession(tmp_path)

    hotkeys = registry.execute(session, "/hotkeys")  # type: ignore[arg-type]
    skill = registry.execute(session, "/skill")  # type: ignore[arg-type]

    assert hotkeys.message is not None
    assert "Ctrl+K: open slash-command completions" in hotkeys.message
    assert "Alt+Enter: queue follow-up" in hotkeys.message
    assert skill.message == (
        "Use /skill:<name> [request] to expand a loaded skill into your prompt."
    )


def test_quit_reload_and_theme_return_declarative_actions(tmp_path: Path) -> None:
    registry = create_default_command_registry()
    session = FakeCommandSession(tmp_path)

    assert registry.execute(session, "/quit").exit_requested is True  # type: ignore[arg-type]
    assert registry.execute(session, "/reload").reload_requested is True  # type: ignore[arg-type]
    assert registry.execute(session, "/reload now").message == "Usage: /reload"  # type: ignore[arg-type]
    assert registry.execute(session, "/theme").theme_picker_requested is True  # type: ignore[arg-type]
    assert registry.execute(session, "/theme axis-light").theme == "axis-light"  # type: ignore[arg-type]

    unknown = registry.execute(session, "/theme solarized")  # type: ignore[arg-type]
    assert unknown.message is not None
    assert "Unknown theme: solarized" in unknown.message
    assert BUILTIN_COMMAND_THEME_NAMES == BUILTIN_TUI_THEME_NAMES


def test_reload_summary_reports_counts_deltas_and_system_change() -> None:
    summary = CodingReloadSummary(
        skills=ReloadCategorySummary(0, 2, True),
        prompt_templates=ReloadCategorySummary(1, 1, False),
        context_files=ReloadCategorySummary(1, 0, True),
        diagnostics=ReloadCategorySummary(0, 0, False),
        system_prompt_rebuilt=True,
    )

    rendered = format_reload_summary(summary)

    assert "Skills: 2 total (changed, +2)" in rendered
    assert "Prompt templates: 1 total (unchanged)" in rendered
    assert "Project context files: 0 total (changed, -1)" in rendered
    assert "Next-turn system prompt: rebuilt" in rendered


def test_session_lifecycle_commands_return_async_application_actions(tmp_path: Path) -> None:
    registry = create_default_command_registry()
    session = FakeCommandSession(tmp_path)

    assert registry.execute(session, "/new").new_session_requested is True  # type: ignore[arg-type]
    assert registry.execute(session, "/compact").compact_instructions == ""  # type: ignore[arg-type]
    assert (
        registry.execute(  # type: ignore[arg-type]
            session, "/compact Keep decisions"
        ).compact_instructions
        == "Keep decisions"
    )
    exported = registry.execute(  # type: ignore[arg-type]
        session,
        "/export --format jsonl exports/session.jsonl",
    )
    assert exported.export_requested is True
    assert exported.export_format == "jsonl"
    assert exported.export_destination == Path("exports/session.jsonl")
    assert registry.execute(session, "/tree").tree_picker_requested is True  # type: ignore[arg-type]


def test_resume_and_name_validate_indexed_session_metadata(tmp_path: Path) -> None:
    manager = SessionManager(AxisPaths(home=tmp_path / ".axis", agents_home=tmp_path / ".agents"))
    manager.create_session(cwd=tmp_path, model="fake", session_id="known", title="Current")
    session = FakeCommandSession(tmp_path)
    session.session_manager = manager
    session.session_id = "known"
    session.session_title = "Current"
    registry = create_default_command_registry()

    assert registry.execute(session, "/resume").resume_picker_requested is True  # type: ignore[arg-type]
    assert registry.execute(session, "/resume known").resume_session_id == "known"  # type: ignore[arg-type]
    assert registry.execute(session, "/resume missing").message == "Unknown session: missing"  # type: ignore[arg-type]
    assert registry.execute(session, "/name Updated").rename_to == "Updated"  # type: ignore[arg-type]
    assert "Current session name: Current" in (  # type: ignore[arg-type]
        registry.execute(session, "/name").message or ""
    )


def test_model_thinking_and_scoped_commands_return_tui_actions(tmp_path: Path) -> None:
    registry = create_default_command_registry()
    session = FakeCommandSession(tmp_path)

    assert registry.execute(session, "/model").model_picker_requested is True  # type: ignore[arg-type]
    assert registry.execute(session, "/model deepseek-v4-pro").model_name == (  # type: ignore[arg-type]
        "deepseek-v4-pro"
    )
    unknown = registry.execute(session, "/model missing")  # type: ignore[arg-type]
    assert "Unknown model for provider deepseek" in (unknown.message or "")
    assert registry.execute(  # type: ignore[arg-type]
        session, "/scoped-models"
    ).scoped_models_picker_requested
    assert registry.execute(session, "/thinking xhigh").thinking_level == "xhigh"  # type: ignore[arg-type]
    status = registry.execute(session, "/thinking")  # type: ignore[arg-type]
    assert status.message == "Thinking mode: high\nAvailable modes: high, xhigh"


def test_login_and_logout_commands_validate_builtin_provider(tmp_path: Path) -> None:
    registry = create_default_command_registry()
    session = FakeCommandSession(tmp_path)

    assert registry.execute(session, "/login").login_picker_requested  # type: ignore[arg-type]
    assert registry.execute(session, "/login deepseek").login_provider == "deepseek"  # type: ignore[arg-type]
    assert registry.execute(session, "/logout").logout_picker_requested  # type: ignore[arg-type]
    assert registry.execute(session, "/logout deepseek").logout_provider == "deepseek"  # type: ignore[arg-type]
    assert "Unknown login provider" in (  # type: ignore[arg-type]
        registry.execute(session, "/login missing").message or ""
    )
