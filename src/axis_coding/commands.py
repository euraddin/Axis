"""Slash-command registry for Axis coding sessions."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from axis_agent import AgentTool
from axis_coding.context import ProjectContextFile
from axis_coding.prompt_templates import PromptTemplate
from axis_coding.provider_catalog import BUILTIN_PROVIDER_CATALOG, builtin_provider_entry
from axis_coding.reload import CodingReloadSummary, ReloadCategorySummary
from axis_coding.resources import ResourceDiagnostic
from axis_coding.session_manager import SessionManager
from axis_coding.skills import Skill
from axis_coding.thinking import ThinkingLevel, normalize_thinking_level

BUILTIN_COMMAND_THEME_NAMES = ("axis-dark", "axis-light", "high-contrast", "omni")


class CommandSession(Protocol):
    """Read-only session surface available to synchronous handlers."""

    @property
    def cwd(self) -> Path: ...

    @property
    def model(self) -> str: ...

    @property
    def provider_name(self) -> str: ...

    @property
    def available_models(self) -> Sequence[str]: ...

    @property
    def available_thinking_levels(self) -> Sequence[ThinkingLevel]: ...

    @property
    def thinking_level(self) -> ThinkingLevel: ...

    @property
    def thinking_unavailable_reason(self) -> str | None: ...

    def reload_provider_settings(self) -> None: ...

    @property
    def tools(self) -> Sequence[AgentTool]: ...

    @property
    def skills(self) -> Sequence[Skill]: ...

    @property
    def prompt_templates(self) -> Sequence[PromptTemplate]: ...

    @property
    def context_files(self) -> Sequence[ProjectContextFile]: ...

    @property
    def resource_diagnostics(self) -> Sequence[ResourceDiagnostic]: ...

    @property
    def messages(self) -> Sequence[object]: ...

    @property
    def session_id(self) -> str | None: ...

    @property
    def session_title(self) -> str | None: ...

    @property
    def session_manager(self) -> SessionManager | None: ...


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Declarative action requested by one handled slash command."""

    handled: bool
    exit_requested: bool = False
    reload_requested: bool = False
    new_session_requested: bool = False
    compact_instructions: str | None = None
    export_requested: bool = False
    export_destination: Path | None = None
    export_format: str | None = None
    resume_session_id: str | None = None
    resume_picker_requested: bool = False
    tree_picker_requested: bool = False
    rename_to: str | None = None
    model_name: str | None = None
    model_picker_requested: bool = False
    scoped_models_picker_requested: bool = False
    thinking_level: str | None = None
    login_picker_requested: bool = False
    login_provider: str | None = None
    logout_picker_requested: bool = False
    logout_provider: str | None = None
    theme_picker_requested: bool = False
    theme: str | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class CommandContext:
    """Normalized command invocation passed to a handler."""

    session: CommandSession
    registry: CommandRegistry
    text: str
    name: str
    args: str


type CommandHandler = Callable[[CommandContext], CommandResult]


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """Registered behavior plus the metadata shown in autocomplete."""

    name: str
    description: str
    usage: str
    handler: CommandHandler
    aliases: tuple[str, ...] = ()
    search_terms: tuple[str, ...] = ()


class CommandRegistry:
    """Register, resolve and execute slash commands deterministically."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._aliases: dict[str, str] = {}

    def register(self, command: SlashCommand) -> None:
        name = _normalize_name(command.name)
        if name in self._commands or name in self._aliases:
            raise ValueError(f"Duplicate slash command: /{name}")
        aliases = tuple(_normalize_name(alias) for alias in command.aliases)
        if len(set(aliases)) != len(aliases):
            raise ValueError(f"Duplicate alias within slash command: /{name}")
        for alias in aliases:
            if alias in self._commands or alias in self._aliases or alias == name:
                raise ValueError(f"Duplicate slash command alias: /{alias}")
        self._commands[name] = command
        self._aliases.update({alias: name for alias in aliases})

    def get(self, name: str) -> SlashCommand | None:
        normalized = _normalize_name(name)
        canonical = self._aliases.get(normalized, normalized)
        return self._commands.get(canonical)

    def list_commands(self) -> tuple[SlashCommand, ...]:
        return tuple(self._commands[name] for name in sorted(self._commands))

    def execute(self, session: CommandSession, text: str) -> CommandResult:
        stripped = text.strip()
        if not stripped.startswith("/") or stripped.startswith("/skill:"):
            return CommandResult(handled=False)
        name, args = _parse_command(stripped)
        if not name:
            return CommandResult(handled=False)
        command = self.get(name)
        if command is None:
            return CommandResult(handled=True, message=f"Unknown command: /{name}")
        return command.handler(
            CommandContext(
                session=session,
                registry=self,
                text=stripped,
                name=name,
                args=args,
            )
        )


def create_default_command_registry() -> CommandRegistry:
    """Create the commands whose behavior is implemented through TUI-7."""
    registry = CommandRegistry()
    registry.register(SlashCommand("quit", "Exit the current session.", "/quit", _quit_command))
    registry.register(
        SlashCommand(
            "new",
            "Start a new session.",
            "/new",
            _new_command,
            search_terms=("clear", "reset"),
        )
    )
    registry.register(
        SlashCommand(
            "compact",
            "Summarize and compact active context.",
            "/compact [instructions]",
            _compact_command,
        )
    )
    registry.register(
        SlashCommand(
            "export",
            "Export the current session.",
            "/export [--format html|jsonl] [destination]",
            _export_command,
        )
    )
    registry.register(
        SlashCommand(
            "session",
            "Show session info and stats.",
            "/session",
            _session_command,
            search_terms=("info", "status"),
        )
    )
    registry.register(
        SlashCommand(
            "skill",
            "Expand a loaded skill into your prompt.",
            "/skill:<name> [request]",
            _skill_command,
            search_terms=("skills",),
        )
    )
    registry.register(
        SlashCommand(
            "hotkeys",
            "Show common keyboard shortcuts.",
            "/hotkeys",
            _hotkeys_command,
            search_terms=("keys", "shortcuts", "bindings"),
        )
    )
    registry.register(
        SlashCommand(
            "reload",
            "Reload local resources and project context.",
            "/reload",
            _reload_command,
        )
    )
    registry.register(
        SlashCommand(
            "resume",
            "Resume a previous session.",
            "/resume [session-id]",
            _resume_command,
            search_terms=("history", "previous"),
        )
    )
    registry.register(
        SlashCommand(
            "tree",
            "Branch from a previous session entry.",
            "/tree",
            _tree_command,
            search_terms=("branch", "history", "fork"),
        )
    )
    registry.register(
        SlashCommand(
            "name",
            "Rename the current session.",
            "/name <new name>",
            _name_command,
            search_terms=("rename", "title"),
        )
    )
    registry.register(
        SlashCommand(
            "model",
            "Choose the active model.",
            "/model [model]",
            _model_command,
        )
    )
    registry.register(
        SlashCommand(
            "scoped-models",
            "Choose models available to quick-cycle with Ctrl+P.",
            "/scoped-models",
            _scoped_models_command,
            search_terms=("scope", "quick", "cycle", "ctrl+p"),
        )
    )
    registry.register(
        SlashCommand(
            "thinking",
            "Show or set model reasoning effort.",
            "/thinking [level]",
            _thinking_command,
            search_terms=("reasoning", "effort"),
        )
    )
    registry.register(
        SlashCommand(
            "login",
            "Save an API key for a built-in provider.",
            "/login [provider]",
            _login_command,
            search_terms=("auth", "api key", "credential"),
        )
    )
    registry.register(
        SlashCommand(
            "logout",
            "Remove an API key saved by Axis.",
            "/logout [provider]",
            _logout_command,
            search_terms=("auth", "credential"),
        )
    )
    registry.register(
        SlashCommand(
            "theme",
            "Show or set the TUI theme.",
            "/theme [name]",
            _theme_command,
            search_terms=("light", "dark", "contrast"),
        )
    )
    return registry


def format_reload_summary(summary: CodingReloadSummary) -> str:
    """Render a stable multi-section reload report."""
    return "\n".join(
        [
            "Reloaded local coding resources and project context.",
            "Resources:",
            f"- Skills: {_format_reload_category(summary.skills)}",
            f"- Prompt templates: {_format_reload_category(summary.prompt_templates)}",
            "Context:",
            f"- Project context files: {_format_reload_category(summary.context_files)}",
            "- Next-turn system prompt: "
            + ("rebuilt" if summary.system_prompt_rebuilt else "unchanged"),
            "Diagnostics:",
            f"- Resource diagnostics: {_format_reload_category(summary.diagnostics)}",
            "Provider config:",
            "- Not refreshed by /reload; provider settings are unchanged.",
        ]
    )


def _quit_command(context: CommandContext) -> CommandResult:
    del context
    return CommandResult(handled=True, exit_requested=True, message="Exiting session.")


def _new_command(context: CommandContext) -> CommandResult:
    if context.args:
        return CommandResult(handled=True, message="Usage: /new")
    return CommandResult(handled=True, new_session_requested=True)


def _compact_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, compact_instructions=context.args)


def _export_command(context: CommandContext) -> CommandResult:
    try:
        export_format, destination = _parse_export_args(context.args)
    except ValueError as exc:
        return CommandResult(handled=True, message=str(exc))
    return CommandResult(
        handled=True,
        export_requested=True,
        export_destination=destination,
        export_format=export_format,
    )


def _session_command(context: CommandContext) -> CommandResult:
    session = context.session
    lines = [
        f"Model: {session.provider_name}:{session.model}",
        f"Thinking mode: {session.thinking_level}",
        f"CWD: {session.cwd}",
        f"Tools: {len(session.tools)}",
        f"Skills: {len(session.skills)}",
        f"Prompt templates: {len(session.prompt_templates)}",
        f"Context files: {len(session.context_files)}",
        f"Messages: {len(session.messages)}",
        f"Resource diagnostics: {len(session.resource_diagnostics)}",
    ]
    if session.session_id is not None:
        lines.append(f"Session: {session.session_id}")
    if session.session_title:
        lines.append(f"Session name: {session.session_title}")
    return CommandResult(handled=True, message="\n".join(lines))


def _hotkeys_command(context: CommandContext) -> CommandResult:
    del context
    return CommandResult(
        handled=True,
        message="\n".join(
            [
                "Common keyboard shortcuts:",
                "- Enter: submit or steer while running",
                "- Shift+Enter: insert newline",
                "- Alt+Enter: queue follow-up while running",
                "- Up: edit latest queued follow-up from an empty prompt",
                "- Esc: cancel active run",
                "- Ctrl+K: open slash-command completions",
                "- Ctrl+T: toggle thinking tokens",
                "- Shift+Tab: cycle thinking mode",
                "- Ctrl+P: cycle scoped models",
                "- Ctrl+O: collapse or expand tool output",
                "- Ctrl+C: clear prompt input when no text is selected",
                "- Ctrl+D: quit",
            ]
        ),
    )


def _reload_command(context: CommandContext) -> CommandResult:
    if context.args:
        return CommandResult(handled=True, message="Usage: /reload")
    return CommandResult(handled=True, reload_requested=True)


def _resume_command(context: CommandContext) -> CommandResult:
    if not context.args:
        return CommandResult(handled=True, resume_picker_requested=True)
    manager = context.session.session_manager
    if manager is None:
        return CommandResult(handled=True, message="Session manager is not available.")
    session_id = context.args.strip()
    if manager.get_session(session_id) is None:
        return CommandResult(handled=True, message=f"Unknown session: {session_id}")
    return CommandResult(handled=True, resume_session_id=session_id)


def _tree_command(context: CommandContext) -> CommandResult:
    if context.args:
        return CommandResult(handled=True, message="Usage: /tree")
    return CommandResult(handled=True, tree_picker_requested=True)


def _name_command(context: CommandContext) -> CommandResult:
    if context.session.session_manager is None or context.session.session_id is None:
        return CommandResult(handled=True, message="Session manager is not available.")
    if not context.args:
        current = context.session.session_title or "Untitled session"
        return CommandResult(
            handled=True,
            message=f"Current session name: {current}\nUsage: /name <new name>",
        )
    return CommandResult(handled=True, rename_to=context.args)


def _skill_command(context: CommandContext) -> CommandResult:
    del context
    return CommandResult(
        handled=True,
        message="Use /skill:<name> [request] to expand a loaded skill into your prompt.",
    )


def _model_command(context: CommandContext) -> CommandResult:
    refresh_error = _refresh_provider_settings(context.session)
    if refresh_error is not None:
        return refresh_error
    if not context.args:
        return CommandResult(handled=True, model_picker_requested=True)
    model = context.args.strip()
    available = tuple(context.session.available_models)
    if available and model not in available:
        return CommandResult(
            handled=True,
            message=(
                f"Unknown model for provider {context.session.provider_name}: {model}\n"
                f"Available models: {', '.join(available)}"
            ),
        )
    return CommandResult(handled=True, model_name=model)


def _scoped_models_command(context: CommandContext) -> CommandResult:
    refresh_error = _refresh_provider_settings(context.session)
    if refresh_error is not None:
        return refresh_error
    if context.args:
        return CommandResult(handled=True, message="Usage: /scoped-models")
    return CommandResult(handled=True, scoped_models_picker_requested=True)


def _thinking_command(context: CommandContext) -> CommandResult:
    available = tuple(context.session.available_thinking_levels)
    if not context.args:
        if available:
            return CommandResult(
                handled=True,
                message=(
                    f"Thinking mode: {context.session.thinking_level}\n"
                    f"Available modes: {', '.join(available)}"
                ),
            )
        reason = context.session.thinking_unavailable_reason
        suffix = f"\nThinking unavailable: {reason}" if reason else ""
        return CommandResult(handled=True, message=f"Thinking mode: unavailable{suffix}")
    if not available:
        reason = context.session.thinking_unavailable_reason
        suffix = f": {reason}" if reason else ""
        return CommandResult(
            handled=True,
            message=(
                "Thinking controls are unavailable for "
                f"{context.session.provider_name}:{context.session.model}{suffix}"
            ),
        )
    try:
        level = normalize_thinking_level(context.args)
    except ValueError as exc:
        return CommandResult(handled=True, message=str(exc))
    if level not in available:
        return CommandResult(
            handled=True,
            message=(
                f"Thinking mode {level} is not available for "
                f"{context.session.provider_name}:{context.session.model}\n"
                f"Available modes: {', '.join(available)}"
            ),
        )
    return CommandResult(handled=True, thinking_level=level)


def _login_command(context: CommandContext) -> CommandResult:
    if not context.args:
        return CommandResult(handled=True, login_picker_requested=True)
    provider = builtin_provider_entry(context.args)
    if provider is None:
        available = ", ".join(entry.name for entry in BUILTIN_PROVIDER_CATALOG)
        return CommandResult(
            handled=True,
            message=(f"Unknown login provider: {context.args}\nAvailable providers: {available}"),
        )
    return CommandResult(handled=True, login_provider=provider.name)


def _logout_command(context: CommandContext) -> CommandResult:
    if not context.args:
        return CommandResult(handled=True, logout_picker_requested=True)
    provider = builtin_provider_entry(context.args)
    if provider is None:
        available = ", ".join(entry.name for entry in BUILTIN_PROVIDER_CATALOG)
        return CommandResult(
            handled=True,
            message=(f"Unknown logout provider: {context.args}\nAvailable providers: {available}"),
        )
    return CommandResult(handled=True, logout_provider=provider.name)


def _refresh_provider_settings(session: CommandSession) -> CommandResult | None:
    try:
        session.reload_provider_settings()
    except ValueError as exc:
        return CommandResult(
            handled=True,
            message=f"Could not refresh provider settings: {exc}",
        )
    return None


def _theme_command(context: CommandContext) -> CommandResult:
    if not context.args:
        return CommandResult(handled=True, theme_picker_requested=True)
    name = context.args.strip()
    if name not in BUILTIN_COMMAND_THEME_NAMES:
        available = ", ".join(BUILTIN_COMMAND_THEME_NAMES)
        return CommandResult(
            handled=True,
            message=f"Unknown theme: {name}\nAvailable themes: {available}",
        )
    return CommandResult(handled=True, theme=name)


def _format_reload_category(summary: ReloadCategorySummary) -> str:
    status = "changed" if summary.changed else "unchanged"
    delta = summary.delta
    suffix = f", {delta:+d}" if delta else ""
    return f"{summary.after} total ({status}{suffix})"


def _parse_command(text: str) -> tuple[str, str]:
    command, separator, args = text[1:].partition(" ")
    if not command:
        return "", ""
    return _normalize_name(command), args.strip() if separator else ""


def _normalize_name(name: str) -> str:
    normalized = name.strip().casefold()
    if not normalized or any(character.isspace() for character in normalized):
        raise ValueError("Slash command names must be non-empty and contain no whitespace")
    return normalized


def _parse_export_args(args: str) -> tuple[str | None, Path | None]:
    parts = args.split()
    export_format: str | None = None
    destination: Path | None = None
    index = 0
    while index < len(parts):
        part = parts[index]
        if part == "--format":
            index += 1
            if index >= len(parts):
                raise ValueError("Usage: /export [--format html|jsonl] [destination]")
            export_format = parts[index].casefold()
            if export_format not in {"html", "jsonl"}:
                raise ValueError(f"Unsupported export format: {parts[index]}")
        elif part.startswith("--"):
            raise ValueError(f"Unknown export option: {part}")
        elif destination is None:
            destination = Path(part)
        else:
            raise ValueError("Usage: /export [--format html|jsonl] [destination]")
        index += 1
    return export_format, destination
