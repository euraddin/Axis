"""Textual application for one Axis coding session."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from inspect import isawaitable
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, cast

from rich.console import Group
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import Click, Key, Resize
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
    TextArea,
)
from textual.worker import Worker

from axis_agent import (
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    ThinkingDeltaEvent,
    ToolApprovalDecision,
    ToolCall,
    TurnStartEvent,
)
from axis_agent.tools import AgentTool, ToolCancellationToken
from axis_coding.commands import (
    CommandRegistry,
    CommandResult,
    CommandSession,
    create_default_command_registry,
    format_reload_summary,
)
from axis_coding.context_window import ContextUsageEstimate, RequestContextBreakdown
from axis_coding.credentials import FileCredentialStore, credentials_path
from axis_coding.paths import AxisPaths
from axis_coding.permissions import (
    SessionToolApprovalController,
    ToolApprovalPreview,
    build_tool_approval_preview,
)
from axis_coding.provider_catalog import (
    BUILTIN_PROVIDER_CATALOG,
    ProviderCatalogEntry,
    builtin_provider_entry,
)
from axis_coding.provider_config import (
    load_provider_settings,
    provider_config_from_catalog_entry,
    provider_settings_path,
    save_provider_settings,
    upsert_provider,
)
from axis_coding.session import (
    ModelChoice,
    SessionTreeBranchResult,
    SessionTreeChoice,
    StreamingBehavior,
    TerminalCommandResult,
    parse_terminal_command,
)
from axis_coding.session_manager import CodingSessionRecord
from axis_coding.thinking import THINKING_LEVEL_DESCRIPTIONS
from axis_coding.tui.adapter import TuiEventAdapter
from axis_coding.tui.autocomplete import (
    CompletionCommand,
    CompletionOption,
    CompletionState,
    build_completion_state,
)
from axis_coding.tui.config import (
    BUILTIN_TUI_THEME_NAMES,
    TuiKeybindings,
    TuiSettings,
    TuiTheme,
    TuiThemeName,
    load_tui_settings,
    save_tui_settings,
)
from axis_coding.tui.state import TuiState, format_terminal_command_result_block
from axis_coding.tui.widgets import (
    CompactSessionInfo,
    SessionSidebar,
    TranscriptView,
    render_completion_suggestions,
    render_request_context_usage,
)
from axis_coding.voice import (
    AudioInputDevice,
    VoiceContextSnapshot,
    VoiceInputController,
    VoiceInputEvent,
    build_voice_context_snapshot,
    create_voice_input_controller,
    list_audio_input_devices,
    load_voice_config,
    resolve_voice_api_key,
    save_voice_config,
    test_voice_configuration,
)
from axis_coding.voice.config import VOLCENGINE_ASR_CREDENTIAL_NAME

ACTIVITY_TICK_SECONDS = 0.15
ACTIVITY_INDICATOR_HEIGHT = 3
SIDEBAR_MIN_WIDTH = 96
SIDEBAR_MIN_HEIGHT = 24


class CompletionActionTarget(Protocol):
    """App actions used by the prompt editor."""

    def action_accept_completion(self) -> None: ...

    def action_completion_next(self) -> None: ...

    def action_completion_previous(self) -> None: ...

    def action_edit_queued_follow_up(self) -> bool: ...

    def action_open_command_palette(self) -> None: ...

    def action_open_session_picker(self) -> None: ...

    def action_cycle_thinking(self) -> None: ...

    def action_cycle_model(self) -> None: ...

    def action_cancel_run(self) -> None: ...

    def action_toggle_voice(self) -> None: ...

    def action_toggle_tool_results(self) -> None: ...

    def action_toggle_thinking(self) -> None: ...

    def action_exit_app(self) -> None: ...

    async def action_submit_prompt(self) -> None: ...

    async def action_submit_follow_up(self) -> None: ...


class PromptInput(TextArea):
    """Multiline prompt editor with completion-aware key routing."""

    BINDINGS: ClassVar[list[BindingType]] = []

    def __init__(
        self,
        *,
        tui_keybindings: TuiKeybindings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tui_keybindings = tui_keybindings or TuiKeybindings()
        self.shell_mode_style = ""
        self._base_bindings = self._bindings.copy()
        self._footer_mode: Literal["normal", "completion", "running"] = "normal"
        self._voice_active = False
        self._apply_prompt_bindings()

    def set_footer_mode(self, mode: Literal["normal", "completion", "running"]) -> None:
        if mode == self._footer_mode:
            return
        self._footer_mode = mode
        self._apply_prompt_bindings()
        self.refresh_bindings()

    def _apply_prompt_bindings(self) -> None:
        self._bindings = BindingsMap.merge(
            [
                self._base_bindings,
                BindingsMap(
                    _prompt_bindings(
                        self.tui_keybindings,
                        mode=self._footer_mode,
                        voice_active=self._voice_active,
                    )
                ),
            ]
        )

    def set_voice_active(self, active: bool) -> None:
        if active == self._voice_active:
            return
        self._voice_active = active
        self._apply_prompt_bindings()
        self.refresh_bindings()

    @property
    def value(self) -> str:
        """Compatibility alias for the former single-line Input widget."""
        return self.text

    @value.setter
    def value(self, text: str) -> None:
        self.text = text

    @property
    def cursor_position(self) -> int:
        """Return the cursor as a flat string offset."""
        row, column = self.cursor_location
        lines = self.text.split("\n")
        return sum(len(line) + 1 for line in lines[:row]) + column

    @cursor_position.setter
    def cursor_position(self, offset: int) -> None:
        bounded = min(max(offset, 0), len(self.text))
        before = self.text[:bounded]
        self.move_cursor((before.count("\n"), len(before.rsplit("\n", 1)[-1])))

    async def on_key(self, event: Key) -> None:
        """Reserve submit and completion keys while leaving text editing native."""
        target = cast(CompletionActionTarget, self.app)
        keybindings = self.tui_keybindings
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            await target.action_submit_prompt()
        elif event.key == keybindings.voice_record:
            event.stop()
            event.prevent_default()
            target.action_toggle_voice()
        elif event.key == keybindings.queue_follow_up:
            event.stop()
            event.prevent_default()
            await target.action_submit_follow_up()
        elif event.key == "shift+enter":
            event.stop()
            event.prevent_default()
            self.insert("\n")
        elif event.key == keybindings.accept_completion:
            event.stop()
            event.prevent_default()
            target.action_accept_completion()
        elif event.key == keybindings.command_palette:
            event.stop()
            event.prevent_default()
            target.action_open_command_palette()
        elif event.key == keybindings.session_picker:
            event.stop()
            event.prevent_default()
            target.action_open_session_picker()
        elif event.key == keybindings.thinking_cycle:
            event.stop()
            event.prevent_default()
            target.action_cycle_thinking()
        elif event.key == keybindings.model_cycle:
            event.stop()
            event.prevent_default()
            target.action_cycle_model()
        elif event.key == keybindings.cancel:
            event.stop()
            event.prevent_default()
            target.action_cancel_run()
        elif event.key == keybindings.completion_next and self._has_completions():
            event.stop()
            event.prevent_default()
            target.action_completion_next()
        elif event.key == keybindings.completion_previous and self._has_completions():
            event.stop()
            event.prevent_default()
            target.action_completion_previous()
        elif (
            event.key == keybindings.completion_previous
            and not self.text
            and target.action_edit_queued_follow_up()
        ):
            event.stop()
            event.prevent_default()
        elif event.key == keybindings.copy_message and not self.selected_text:
            event.stop()
            event.prevent_default()
            self.text = ""
            self.move_cursor((0, 0))
        elif event.key == keybindings.quit:
            event.stop()
            event.prevent_default()
            target.action_exit_app()

    def action_accept_completion(self) -> None:
        cast(CompletionActionTarget, self.app).action_accept_completion()

    def action_completion_next(self) -> None:
        target = cast(CompletionActionTarget, self.app)
        if self._has_completions():
            target.action_completion_next()
        else:
            self.action_cursor_down()

    def action_completion_previous(self) -> None:
        target = cast(CompletionActionTarget, self.app)
        if self._has_completions():
            target.action_completion_previous()
        elif not self.text and target.action_edit_queued_follow_up():
            return
        else:
            self.action_cursor_up()

    def action_cancel(self) -> None:
        cast(CompletionActionTarget, self.app).action_cancel_run()

    def action_toggle_voice(self) -> None:
        cast(CompletionActionTarget, self.app).action_toggle_voice()

    def action_open_command_palette(self) -> None:
        cast(CompletionActionTarget, self.app).action_open_command_palette()

    def action_open_session_picker(self) -> None:
        cast(CompletionActionTarget, self.app).action_open_session_picker()

    def action_cycle_thinking(self) -> None:
        cast(CompletionActionTarget, self.app).action_cycle_thinking()

    def action_cycle_model(self) -> None:
        cast(CompletionActionTarget, self.app).action_cycle_model()

    def action_toggle_tool_results(self) -> None:
        cast(CompletionActionTarget, self.app).action_toggle_tool_results()

    def action_toggle_thinking(self) -> None:
        cast(CompletionActionTarget, self.app).action_toggle_thinking()

    def action_clear_prompt(self) -> None:
        if not self.selected_text:
            self.text = ""
            self.move_cursor((0, 0))

    async def action_submit_prompt(self) -> None:
        await cast(CompletionActionTarget, self.app).action_submit_prompt()

    async def action_submit_follow_up(self) -> None:
        await cast(CompletionActionTarget, self.app).action_submit_follow_up()

    def action_insert_newline(self) -> None:
        self.insert("\n")

    def action_quit(self) -> None:
        cast(CompletionActionTarget, self.app).action_exit_app()

    def get_line(self, line_index: int) -> Text:
        """Highlight the leading shell-mode prefix on the first line."""
        line = super().get_line(line_index)
        if line_index != 0 or not self.shell_mode_style:
            return line
        span = _terminal_command_prefix_span(self.text)
        if span is not None:
            line.stylize(self.shell_mode_style, *span)
        return line

    def _has_completions(self) -> bool:
        return bool(getattr(self.app, "completion_state", CompletionState()).items)


class TuiSession(Protocol):
    """Narrow session interface required by the basic Textual frontend."""

    @property
    def cwd(self) -> Path:
        """Return the session working directory."""
        ...

    @property
    def model(self) -> str:
        """Return the active model name."""
        ...

    @property
    def provider_name(self) -> str: ...

    @property
    def available_models(self) -> tuple[str, ...]: ...

    @property
    def available_providers(self) -> tuple[str, ...]: ...

    @property
    def available_model_choices(self) -> tuple[ModelChoice, ...]: ...

    @property
    def scoped_model_choices(self) -> tuple[ModelChoice, ...]: ...

    @property
    def thinking_level(self) -> str: ...

    @property
    def available_thinking_levels(self) -> tuple[str, ...]: ...

    @property
    def context_token_estimate(self) -> int: ...

    @property
    def context_usage(self) -> ContextUsageEstimate: ...

    @property
    def context_window_tokens(self) -> int: ...

    @property
    def auto_compact_token_threshold(self) -> int | None: ...

    @property
    def tools(self) -> Sequence[object]: ...

    @property
    def skills(self) -> Sequence[object]: ...

    @property
    def prompt_templates(self) -> Sequence[object]: ...

    @property
    def context_files(self) -> Sequence[object]: ...

    @property
    def is_running(self) -> bool:
        """Return whether an agent run is active."""
        ...

    def prompt(
        self,
        content: str,
        *,
        streaming_behavior: StreamingBehavior | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Start one prompt event stream."""
        ...

    async def run_terminal_command(
        self,
        command: str,
        *,
        add_to_context: bool,
    ) -> TerminalCommandResult:
        """Execute one input-bar shell command."""
        ...

    def queue_update_event(self) -> AgentEvent:
        """Return the current queued-message snapshot."""
        ...

    def pop_latest_follow_up_message(self) -> str | None:
        """Remove the newest follow-up for editing."""
        ...

    def cancel(self) -> None:
        """Request cancellation of the active run."""
        ...


class CommandOutputScroll(VerticalScroll):
    """Arrow-scrollable viewport used by command output modals."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "scroll_up", "Scroll up", show=False, priority=True),
        Binding("down", "scroll_down", "Scroll down", show=False, priority=True),
    ]

    def action_scroll_up(self) -> None:
        self.scroll_y = max(0, self.scroll_y - 1)

    def action_scroll_down(self) -> None:
        self.scroll_y = min(self.max_scroll_y, self.scroll_y + 1)


class CommandOutputScreen(ModalScreen[None]):
    """Scrollable command output that does not pollute the transcript."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close"),
        Binding("enter", "close", "Close"),
    ]

    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self.title_text = title
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="command-output"):
            yield Static(self.title_text, id="command-output-title")
            with CommandOutputScroll(id="command-output-scroll"):
                yield Static(self.message, id="command-output-body", markup=False)
            yield Static("Enter or Escape closes", id="command-output-help")

    def on_mount(self) -> None:
        self.query_one("#command-output-scroll", VerticalScroll).focus()

    def action_close(self) -> None:
        self.dismiss(None)


class ThemePickerScreen(ModalScreen[TuiThemeName | None]):
    """Small keyboard-driven picker for built-in Axis themes."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select_cursor", "Select", show=False),
    ]

    def __init__(self, current: TuiThemeName) -> None:
        super().__init__()
        self.current = current
        self.theme_names = BUILTIN_TUI_THEME_NAMES

    def compose(self) -> ComposeResult:
        with Vertical(id="theme-picker"):
            yield Static("Choose theme", id="theme-picker-title")
            yield ListView(
                *(
                    ListItem(
                        Label(f"{'✓' if name == self.current else ' '} {name}"),
                    )
                    for name in self.theme_names
                ),
                id="theme-picker-list",
            )
            yield Static("↑/↓ select · Enter apply · Escape cancel", id="theme-picker-help")

    def on_mount(self) -> None:
        themes = self.query_one("#theme-picker-list", ListView)
        themes.index = self.theme_names.index(self.current)
        themes.focus()

    def action_cursor_up(self) -> None:
        self.query_one("#theme-picker-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        self.query_one("#theme-picker-list", ListView).action_cursor_down()

    def action_select_cursor(self) -> None:
        index = self.query_one("#theme-picker-list", ListView).index
        if index is not None:
            self.dismiss(self.theme_names[index])

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Accept mouse and native ListView Enter selection."""
        event.stop()
        self.dismiss(self.theme_names[event.index])

    def action_cancel(self) -> None:
        self.dismiss(None)


class ToolApprovalScreen(ModalScreen[ToolApprovalDecision]):
    """Block a protected tool call until the user makes an explicit decision."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "allow_once", "Allow once"),
        Binding("y", "allow_once", "Allow once", show=False),
        Binding("a", "allow_session", "Allow session"),
        Binding("d", "deny", "Deny"),
        Binding("escape", "deny", "Deny"),
    ]

    def __init__(self, preview: ToolApprovalPreview) -> None:
        super().__init__()
        self.preview = preview

    def compose(self) -> ComposeResult:
        with Vertical(id="tool-approval"):
            yield Static(self.preview.title, id="tool-approval-title")
            yield Static(
                self.preview.render_plain(),
                id="tool-approval-details",
                markup=False,
            )
            with Horizontal(id="tool-approval-actions"):
                yield Button("Allow once", id="tool-approval-once", variant="success")
                yield Button("Allow this tool", id="tool-approval-session", variant="primary")
                yield Button("Deny", id="tool-approval-deny", variant="error")
            yield Static(
                "Enter/Y once · A allow this tool for session · D/Escape deny",
                id="tool-approval-help",
            )

    def on_mount(self) -> None:
        self.query_one("#tool-approval-once", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        decisions: dict[str, ToolApprovalDecision] = {
            "tool-approval-once": "allow_once",
            "tool-approval-session": "allow_session",
            "tool-approval-deny": "deny",
        }
        if event.button.id is not None:
            self.dismiss(decisions[event.button.id])

    def action_allow_once(self) -> None:
        self.dismiss("allow_once")

    def action_allow_session(self) -> None:
        self.dismiss("allow_session")

    def action_deny(self) -> None:
        self.dismiss("deny")


class ModelPickerScreen(ModalScreen[ModelChoice | None]):
    """Search all usable provider/model pairs or the scoped Ctrl+P subset."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
        Binding("enter", "accept", "Select", show=False, priority=True),
        Binding("tab", "toggle_mode", "Mode", show=False, priority=True),
    ]

    def __init__(
        self,
        choices: Sequence[ModelChoice],
        *,
        scoped_choices: Sequence[ModelChoice],
        current: ModelChoice,
        on_toggle_scoped: Callable[[ModelChoice], Sequence[ModelChoice]] | None = None,
        picker_kind: Literal["model", "scoped"] = "model",
    ) -> None:
        super().__init__()
        self.choices = tuple(dict.fromkeys(choices))
        self.scoped_choices = tuple(dict.fromkeys(scoped_choices))
        self.current = current
        self.on_toggle_scoped = on_toggle_scoped
        self.picker_kind = picker_kind
        self.mode: Literal["all", "scoped"] = "all"
        self.visible_choices = self.choices

    def compose(self) -> ComposeResult:
        with Vertical(id="model-picker"):
            yield Static(
                "Model" if self.picker_kind == "model" else "Scoped models",
                id="model-picker-title",
            )
            yield Static(id="model-picker-tabs")
            yield Input(placeholder="Search models", id="model-picker-search")
            yield ListView(id="model-picker-list")
            yield Static(id="model-picker-help")

    async def on_mount(self) -> None:
        self.query_one("#model-picker-search", Input).focus()
        await self._rebuild()

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "model-picker-search":
            await self._rebuild()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "model-picker-search":
            event.stop()
            self.action_accept()

    def action_cursor_up(self) -> None:
        self.query_one("#model-picker-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        self.query_one("#model-picker-list", ListView).action_cursor_down()

    def action_accept(self) -> None:
        model_list = self.query_one("#model-picker-list", ListView)
        if model_list.index is None or model_list.index >= len(self.visible_choices):
            return
        choice = self.visible_choices[model_list.index]
        if self.picker_kind == "scoped":
            if self.on_toggle_scoped is not None:
                self.scoped_choices = tuple(self.on_toggle_scoped(choice))
                self.run_worker(self._rebuild())
            return
        self.dismiss(choice)

    def action_toggle_mode(self) -> None:
        if self.picker_kind == "model":
            self.mode = "scoped" if self.mode == "all" else "all"
            self.run_worker(self._rebuild())

    def action_cancel(self) -> None:
        self.dismiss(None)

    async def _rebuild(self) -> None:
        search = self.query_one("#model-picker-search", Input).value.strip().casefold()
        base = self.scoped_choices if self.mode == "scoped" else self.choices
        self.visible_choices = tuple(
            choice
            for choice in base
            if search in f"{choice.provider_name}:{choice.model}".casefold()
        )
        model_list = self.query_one("#model-picker-list", ListView)
        await model_list.clear()
        await model_list.mount(
            *(
                ListItem(
                    Label(
                        _model_choice_label(
                            choice,
                            current=self.current,
                            scoped=choice in self.scoped_choices,
                        )
                    )
                )
                for choice in self.visible_choices
            )
        )
        try:
            model_list.index = self.visible_choices.index(self.current)
        except ValueError:
            model_list.index = 0 if self.visible_choices else None
        if self.picker_kind == "scoped":
            tabs = "Enter toggles scoped membership; active model is unchanged"
            help_text = f"{len(self.scoped_choices)} scoped · Escape closes"
        else:
            tabs = (
                "Tabs: ● All models  ○ Scoped models"
                if self.mode == "all"
                else "Tabs: ○ All models  ● Scoped models"
            )
            help_text = "Enter selects · Tab switches tabs · Escape closes"
        self.query_one("#model-picker-tabs", Static).update(tabs)
        self.query_one("#model-picker-help", Static).update(help_text)


class LoginProviderPickerScreen(ModalScreen[str | None]):
    """Choose a built-in provider for login or logout."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select_cursor", "Select", show=False),
    ]

    def __init__(
        self,
        providers: Sequence[ProviderCatalogEntry],
        *,
        title: str = "Login",
    ) -> None:
        super().__init__()
        self.providers = tuple(providers)
        self.title_text = title

    def compose(self) -> ComposeResult:
        with Vertical(id="login-provider-picker"):
            yield Static(self.title_text, id="login-provider-title")
            yield ListView(
                *(
                    ListItem(Label(f"{provider.display_name} · {provider.name}"))
                    for provider in self.providers
                ),
                id="login-provider-list",
            )
            yield Static("Enter selects · Escape closes", id="login-provider-help")

    def on_mount(self) -> None:
        provider_list = self.query_one("#login-provider-list", ListView)
        provider_list.index = 0 if self.providers else None
        provider_list.focus()

    def action_cursor_up(self) -> None:
        self.query_one("#login-provider-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        self.query_one("#login-provider-list", ListView).action_cursor_down()

    def action_select_cursor(self) -> None:
        index = self.query_one("#login-provider-list", ListView).index
        if index is not None:
            self.dismiss(self.providers[index].name)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        self.dismiss(self.providers[event.index].name)

    def action_cancel(self) -> None:
        self.dismiss(None)


class LoginScreen(ModalScreen[str | None]):
    """Collect one provider API key without echoing it."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, provider: ProviderCatalogEntry) -> None:
        super().__init__()
        self.provider = provider

    def compose(self) -> ComposeResult:
        with Vertical(id="login-screen"):
            yield Static(f"Login: {self.provider.display_name}", id="login-title")
            yield Static("Paste this provider's API key.", id="login-help")
            yield Input(placeholder="Paste API key", password=True, id="login-api-key")
            yield Static("Enter saves · Escape closes", id="login-footer")

    def on_mount(self) -> None:
        self.query_one("#login-api-key", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "login-api-key":
            event.stop()
            self.dismiss(event.value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


@dataclass(frozen=True, slots=True)
class VoiceSetupResult:
    """Validated values returned by the voice setup modal."""

    api_key: str | None
    input_device: int | str | None


type VoiceSetupTester = Callable[[str, int | str | None, Callable[[str], None]], Awaitable[str]]
type VoiceControllerFactory = Callable[[Callable[[VoiceInputEvent], None]], VoiceInputController]


class VoiceSetupScreen(ModalScreen[VoiceSetupResult | None]):
    """Collect a secret and microphone choice, then require a real ASR test."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(
        self,
        devices: Sequence[AudioInputDevice],
        *,
        has_existing_key: bool,
        current_device: int | str | None,
        tester: VoiceSetupTester,
    ) -> None:
        super().__init__()
        self.devices = tuple(devices)
        self.has_existing_key = has_existing_key
        self.current_device = current_device
        self.tester = tester
        self.device_values: tuple[int | str | None, ...] = (
            None,
            *(device.index for device in self.devices),
        )
        self._test_succeeded = False

    def compose(self) -> ComposeResult:
        with Vertical(id="voice-setup"):
            yield Static("Voice input setup", id="voice-setup-title")
            key_help = (
                "Leave blank to keep the configured Volcengine ASR key."
                if self.has_existing_key
                else "Paste a Volcengine Seed ASR 2.0 API key."
            )
            yield Static(key_help, id="voice-setup-help")
            yield Input(placeholder="Volcengine ASR API key", password=True, id="voice-api-key")
            yield Static("Microphone", classes="voice-setup-label")
            yield ListView(
                ListItem(Label("System default")),
                *(
                    ListItem(Label(f"{device.name}{' · default' if device.default else ''}"))
                    for device in self.devices
                ),
                id="voice-device-list",
            )
            yield Static(
                "Test records 3 seconds and sends it to Volcengine; nothing is saved.",
                id="voice-test-status",
                markup=False,
            )
            with Horizontal(id="voice-setup-actions"):
                yield Button("Test 3 seconds", id="voice-test", variant="primary")
                yield Button("Save", id="voice-save", variant="success", disabled=True)
                yield Button("Cancel", id="voice-cancel")

    def on_mount(self) -> None:
        device_list = self.query_one("#voice-device-list", ListView)
        try:
            device_list.index = self.device_values.index(self.current_device)
        except ValueError:
            device_list.index = 0
        self.query_one("#voice-api-key", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "voice-api-key":
            self._invalidate_test()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id == "voice-device-list":
            self._invalidate_test()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "voice-test":
            self.run_worker(
                self._run_test(),
                name="axis-voice-setup-test",
                group="voice-setup",
                exclusive=True,
                exit_on_error=False,
            )
        elif event.button.id == "voice-save" and self._test_succeeded:
            self.dismiss(
                VoiceSetupResult(
                    api_key=self.query_one("#voice-api-key", Input).value.strip() or None,
                    input_device=self._selected_device(),
                )
            )
        elif event.button.id == "voice-cancel":
            self.dismiss(None)

    async def _run_test(self) -> None:
        key = self.query_one("#voice-api-key", Input).value.strip()
        if not key and not self.has_existing_key:
            self.query_one("#voice-test-status", Static).update("Enter an ASR API key first.")
            return
        test_button = self.query_one("#voice-test", Button)
        save_button = self.query_one("#voice-save", Button)
        test_button.disabled = True
        save_button.disabled = True
        status = self.query_one("#voice-test-status", Static)
        status.update("Connecting, then recording for 3 seconds…")
        try:
            transcript = await self.tester(key, self._selected_device(), status.update)
        except Exception as exc:
            self._test_succeeded = False
            status.update(f"Test failed: {exc}")
        else:
            self._test_succeeded = True
            status.update(f"Test succeeded: {transcript}")
            save_button.disabled = False
        finally:
            test_button.disabled = False

    def _selected_device(self) -> int | str | None:
        index = self.query_one("#voice-device-list", ListView).index
        return self.device_values[index] if index is not None else None

    def _invalidate_test(self) -> None:
        self._test_succeeded = False
        with suppress(NoMatches):
            self.query_one("#voice-save", Button).disabled = True

    def action_cancel(self) -> None:
        self.dismiss(None)


class SessionPickerScreen(ModalScreen[str | None]):
    """Select one indexed session for the current working directory."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select_cursor", "Resume", show=False),
    ]

    def __init__(self, records: tuple[CodingSessionRecord, ...]) -> None:
        super().__init__()
        self.records = records

    def compose(self) -> ComposeResult:
        with Vertical(id="session-picker"):
            yield Static("Resume session", id="session-picker-title")
            yield ListView(
                *(
                    ListItem(Label(f"{record.title or 'Untitled'} · {record.model} · {record.id}"))
                    for record in self.records
                ),
                id="session-picker-list",
            )
            yield Static("↑/↓ select · Enter resume · Escape cancel", id="session-picker-help")

    def on_mount(self) -> None:
        sessions = self.query_one("#session-picker-list", ListView)
        sessions.index = 0 if self.records else None
        sessions.focus()

    def action_cursor_up(self) -> None:
        self.query_one("#session-picker-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        self.query_one("#session-picker-list", ListView).action_cursor_down()

    def action_select_cursor(self) -> None:
        index = self.query_one("#session-picker-list", ListView).index
        if index is not None:
            self.dismiss(self.records[index].id)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        self.dismiss(self.records[event.index].id)

    def action_cancel(self) -> None:
        self.dismiss(None)


@dataclass(frozen=True, slots=True)
class TreePickerSelection:
    entry_id: str
    summarize: bool = False
    custom_instructions: str | None = None


class TreePickerScreen(ModalScreen[TreePickerSelection | None]):
    """Select a historical branch point, optionally summarizing abandoned work."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select_cursor", "Branch", show=False),
        Binding("s", "summarize", "Summarize branch", show=False),
        Binding("c", "custom_summary", "Custom summary", show=False),
        Binding("ctrl+t", "toggle_tool_calls", "Tool calls", show=False, priority=True),
    ]

    def __init__(self, choices: tuple[SessionTreeChoice, ...]) -> None:
        super().__init__()
        self.choices = choices
        self.show_tool_calls = True
        self._visible_choices = choices

    def compose(self) -> ComposeResult:
        with Vertical(id="tree-picker"):
            yield Static("Branch session", id="tree-picker-title")
            yield ListView(id="tree-picker-list")
            yield Static(id="tree-picker-help")

    async def on_mount(self) -> None:
        await self._rebuild()

    async def _rebuild(self) -> None:
        selected_id: str | None = None
        tree = self.query_one("#tree-picker-list", ListView)
        if tree.index is not None and tree.index < len(self._visible_choices):
            selected_id = self._visible_choices[tree.index].entry_id
        self._visible_choices = tuple(
            choice for choice in self.choices if self.show_tool_calls or not choice.is_tool_call
        )
        await tree.clear()
        await tree.mount(
            *(
                ListItem(Label(f"{'*' if choice.active else ' '} {choice.label}"))
                for choice in self._visible_choices
            )
        )
        selected_index = next(
            (
                index
                for index, choice in enumerate(self._visible_choices)
                if choice.entry_id == selected_id
            ),
            None,
        )
        active_index = next(
            (index for index, choice in enumerate(self._visible_choices) if choice.active),
            0,
        )
        tree.index = selected_index if selected_index is not None else active_index
        tree.focus()
        visibility = "shown" if self.show_tool_calls else "hidden"
        self.query_one("#tree-picker-help", Static).update(
            f"Enter branch · S summarize · C custom summary · Ctrl+T tool calls {visibility}"
        )

    def action_cursor_up(self) -> None:
        self.query_one("#tree-picker-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        self.query_one("#tree-picker-list", ListView).action_cursor_down()

    def action_select_cursor(self) -> None:
        self._dismiss_selected(summarize=False)

    def action_summarize(self) -> None:
        self._dismiss_selected(summarize=True)

    def action_custom_summary(self) -> None:
        index = self.query_one("#tree-picker-list", ListView).index
        if index is None or index >= len(self._visible_choices):
            return
        entry_id = self._visible_choices[index].entry_id
        self.app.push_screen(
            BranchSummaryInstructionsScreen(),
            callback=lambda instructions: self._finish_custom_summary(entry_id, instructions),
        )

    def _finish_custom_summary(self, entry_id: str, instructions: str | None) -> None:
        if instructions is not None:
            self.dismiss(
                TreePickerSelection(
                    entry_id,
                    summarize=True,
                    custom_instructions=instructions,
                )
            )

    async def action_toggle_tool_calls(self) -> None:
        self.show_tool_calls = not self.show_tool_calls
        await self._rebuild()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        self.dismiss(TreePickerSelection(self._visible_choices[event.index].entry_id))

    def _dismiss_selected(self, *, summarize: bool) -> None:
        index = self.query_one("#tree-picker-list", ListView).index
        if index is not None and index < len(self._visible_choices):
            self.dismiss(
                TreePickerSelection(
                    self._visible_choices[index].entry_id,
                    summarize=summarize,
                )
            )

    def action_cancel(self) -> None:
        self.dismiss(None)


class BranchSummaryInstructionsScreen(ModalScreen[str | None]):
    """Collect an optional focus for a branch summary."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="branch-summary-instructions"):
            yield Static(
                "Custom summarization instructions",
                id="branch-summary-instructions-title",
            )
            yield TextArea(id="branch-summary-instructions-input")
            yield Static(
                "Ctrl+Enter submit · Escape return to tree",
                id="branch-summary-instructions-help",
            )

    def on_mount(self) -> None:
        self.query_one("#branch-summary-instructions-input", TextArea).focus()

    def on_key(self, event: Key) -> None:
        if event.key == "ctrl+enter":
            event.stop()
            event.prevent_default()
            self.action_submit()
        elif event.key == "escape":
            event.stop()
            event.prevent_default()
            self.action_cancel()

    def action_submit(self) -> None:
        instructions = self.query_one("#branch-summary-instructions-input", TextArea).text.strip()
        self.dismiss(instructions or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class AxisTuiApp(App[None]):
    """Minimal interactive frontend over a CodingSession-like object."""

    TITLE = "Axis"
    SUB_TITLE = "Personal coding agent"
    BINDINGS: ClassVar[list[BindingType]] = []
    CSS = """
    Screen {
        background: $axis-screen-background;
        color: $axis-screen-text;
    }

    Header, Footer {
        background: $axis-chrome-background;
        color: $axis-chrome-text;
    }

    Toast {
        background: $axis-chrome-background;
        color: $axis-chrome-text;
    }

    Toast .toast--title {
        color: $axis-accent;
    }

    #workspace {
        height: 1fr;
    }

    #sidebar {
        width: 32;
        min-width: 28;
        height: 1fr;
        padding: 1 1 0 0;
        background: $axis-sidebar-background;
        border-right: tall $axis-border;
    }

    AxisTuiApp.-hide-sidebar #sidebar {
        display: none;
    }

    #main-pane {
        width: 1fr;
        padding: 1 1 0 1;
    }

    AxisTuiApp.-hide-sidebar #main-pane {
        padding-left: 1;
    }

    #transcript {
        height: 1fr;
        padding: 0 0 0 2;
        border: none;
        background: $axis-transcript-background;
        scrollbar-size-vertical: 0;
        scrollbar-size-horizontal: 0;
    }

    #queued-messages {
        height: auto;
        max-height: 8;
        margin: 0 1 1 1;
        padding: 0 1;
        background: $axis-screen-background;
        color: $axis-muted-text;
    }

    #prompt-row {
        height: auto;
        margin: 0 1 1 1;
    }

    #prompt-prefix {
        width: 2;
        height: 3;
        padding: 0;
        margin: 0;
        content-align: center middle;
        color: $axis-accent;
        text-style: bold;
    }

    #prompt {
        width: 1fr;
        height: auto;
        max-height: 8;
        margin: 0;
        padding: 0 1;
        border: tall transparent;
        background: $axis-prompt-background;
        color: $axis-prompt-text;
    }

    #prompt:focus {
        border: tall $axis-accent;
    }

    #prompt.-shell-mode {
        border: tall $axis-accent;
    }

    #autocomplete {
        height: auto;
        max-height: 18;
        margin: 0 1 1 1;
        padding: 0 1;
        background: $axis-autocomplete-background;
        color: $axis-screen-text;
    }

    #compact-session-info {
        height: auto;
        max-height: 3;
        margin: 0 1 1 1;
        padding: 0 1;
        color: $axis-muted-text;
    }

    #request-context-usage {
        height: auto;
        max-height: 4;
        margin: 0 1 1 1;
        padding: 0 1;
        color: $axis-muted-text;
    }

    #voice-status {
        height: auto;
        max-height: 4;
        margin: 0 1 1 1;
        padding: 0 1;
        color: $axis-accent;
    }

    CommandOutputScreen, ThemePickerScreen, SessionPickerScreen, TreePickerScreen,
    BranchSummaryInstructionsScreen, ModelPickerScreen, LoginProviderPickerScreen,
    LoginScreen, ToolApprovalScreen, VoiceSetupScreen {
        align: center middle;
        background: $axis-screen-overlay-background;
    }

    #command-output, #theme-picker, #session-picker, #tree-picker,
    #branch-summary-instructions, #model-picker, #login-provider-picker, #login-screen,
    #tool-approval, #voice-setup {
        width: 76;
        max-width: 90%;
        height: auto;
        max-height: 70%;
        padding: 1 2;
        border: tall $axis-border;
        background: $axis-chrome-background;
        color: $axis-chrome-text;
    }

    #command-output-title, #theme-picker-title, #session-picker-title, #tree-picker-title,
    #branch-summary-instructions-title, #model-picker-title, #login-provider-title,
    #login-title, #tool-approval-title, #voice-setup-title {
        height: 1;
        margin-bottom: 1;
        text-style: bold;
    }

    #command-output-scroll {
        height: auto;
        max-height: 18;
        background: $axis-transcript-background;
        border: tall $axis-border;
    }

    #command-output-body {
        padding: 1;
        color: $axis-screen-text;
    }

    #command-output-help, #theme-picker-help, #session-picker-help, #tree-picker-help,
    #branch-summary-instructions-help, #model-picker-help, #login-provider-help,
    #login-footer, #tool-approval-help, #voice-setup-help {
        height: 1;
        margin-top: 1;
        color: $axis-muted-text;
    }

    #tool-approval-details {
        height: auto;
        max-height: 18;
        margin-bottom: 1;
        padding: 1;
        background: $axis-transcript-background;
        border: tall $axis-border;
        color: $axis-screen-text;
        overflow-y: auto;
    }

    #tool-approval-actions {
        height: 3;
        align-horizontal: center;
    }

    #tool-approval-actions Button {
        margin: 0 1;
    }

    #theme-picker-list, #session-picker-list, #tree-picker-list, #model-picker-list,
    #login-provider-list, #voice-device-list {
        height: auto;
        max-height: 10;
        background: $axis-transcript-background;
    }

    #branch-summary-instructions-input {
        height: 8;
        background: $axis-transcript-background;
        border: tall $axis-border;
    }

    #model-picker-tabs, #login-help {
        height: auto;
        color: $axis-muted-text;
        margin-bottom: 1;
    }

    #model-picker-search, #login-api-key, #voice-api-key {
        margin-bottom: 1;
        border: tall $axis-border;
        background: $axis-transcript-background;
        color: $axis-screen-text;
    }

    #voice-device-list {
        height: 8;
        margin-bottom: 1;
    }

    #voice-test-status {
        height: auto;
        min-height: 2;
        color: $axis-muted-text;
        margin-bottom: 1;
    }

    #voice-setup-actions {
        height: 3;
        align-horizontal: center;
    }

    #voice-setup-actions Button {
        margin: 0 1;
    }

    ListView > ListItem.--highlight {
        background: $axis-highlight-background;
        color: $axis-highlight-text;
    }
    """

    def __init__(
        self,
        session: TuiSession,
        *,
        tui_settings: TuiSettings | None = None,
        startup_message: str | None = None,
        initial_prompt: str | None = None,
        paths: AxisPaths | None = None,
        voice_controller_factory: VoiceControllerFactory | None = None,
    ) -> None:
        self.tui_settings = tui_settings or TuiSettings()
        self.startup_message = startup_message
        self.initial_prompt = initial_prompt
        self.paths = paths or AxisPaths()
        self._voice_controller_factory = voice_controller_factory or (
            lambda listener: create_voice_input_controller(
                paths=self.paths,
                listener=listener,
            )
        )
        super().__init__()
        self.theme = _textual_base_theme(self.tui_settings.theme)
        self._bindings = BindingsMap(_app_bindings(self.tui_settings.keybindings))
        self.session = session
        registry = getattr(session, "command_registry", None)
        self.command_registry = (
            registry if isinstance(registry, CommandRegistry) else create_default_command_registry()
        )
        self.state = TuiState(skills=tuple(getattr(session, "skills", ())))
        self.state.load_messages(getattr(session, "messages", ()))
        self.adapter = TuiEventAdapter(self.state)
        self.completion_state = CompletionState()
        self._cancel_requested = False
        self._activity_frame = 0
        self._activity_timer: Timer | None = None
        self._terminal_active = False
        self._compaction_worker: Worker[None] | None = None
        self._prompt_worker: Worker[None] | None = None
        self._request_context_usage: ContextUsageEstimate | RequestContextBreakdown | None = None
        self._request_context_turn: int | None = None
        self._voice_controller: VoiceInputController | None = None
        self._voice_anchor: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._voice_editor_text = ""
        self._voice_editor_cursor = 0
        self._voice_worker: Worker[None] | None = None
        self._tool_approval_controller = SessionToolApprovalController(self._show_tool_approval)
        set_approval_handler = getattr(session, "set_tool_approval_handler", None)
        if callable(set_approval_handler):
            set_approval_handler(self._tool_approval_controller)

    def get_theme_variable_defaults(self) -> dict[str, str]:
        """Add Axis theme variables used by the app stylesheet."""
        return {
            **super().get_theme_variable_defaults(),
            **_theme_css_variables(self.tui_settings.resolved_theme),
        }

    def compose(self) -> ComposeResult:
        """Create the compact single-session layout."""
        yield Header()
        with Horizontal(id="workspace"):
            yield SessionSidebar(id="sidebar")
            with Vertical(id="main-pane"):
                yield TranscriptView(id="transcript", min_width=1)
                yield Static(id="queued-messages")
                with Horizontal(id="prompt-row"):
                    yield Static("A", id="prompt-prefix")
                    yield PromptInput(
                        placeholder="Ask Axis…  Enter submits · Shift+Enter adds a line",
                        id="prompt",
                        tui_keybindings=self.tui_settings.keybindings,
                    )
                yield CompactSessionInfo(id="compact-session-info")
                yield Static(id="request-context-usage")
                yield Static(id="voice-status")
                yield Static(id="autocomplete")
        yield Footer()

    def on_mount(self) -> None:
        """Render initial state and focus the prompt."""
        self._render_state()
        self._update_responsive_layout(self.size.width, self.size.height)
        prompt = self.query_one("#prompt", PromptInput)
        prompt.shell_mode_style = self.tui_settings.resolved_theme.accent
        prompt.focus()
        self._rebuild_completions(prompt)
        if self.startup_message:
            self._notify(self.startup_message, severity="warning")
        if self.initial_prompt and self.initial_prompt.strip():
            content = self.initial_prompt.strip()
            self._prompt_worker = self.run_worker(
                self._run_prompt(content),
                name="axis-initial-prompt",
                group="agent",
                exclusive=True,
                exit_on_error=False,
            )

    def on_resize(self, event: Resize) -> None:
        self._update_responsive_layout(event.size.width, event.size.height)

    def on_click(self, event: Click) -> None:
        if event.button == 1:
            with suppress(NoMatches):
                self.query_one("#prompt", PromptInput).focus()

    async def on_text_selected(self) -> None:
        """Optionally copy the current native Textual selection."""
        if not self.tui_settings.auto_copy_selection:
            return
        selected = self.screen.get_selected_text()
        if selected:
            self.copy_to_clipboard(selected)

    def on_unmount(self) -> None:
        """Do not leave an active provider/tool run behind the UI."""
        if self._activity_timer is not None:
            self._activity_timer.stop()
            self._activity_timer = None
        if self.session.is_running:
            self.session.cancel()
        if self._voice_controller is not None and self._voice_controller.active:
            asyncio.create_task(self._voice_controller.cancel())

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Recompute pure suggestions whenever prompt content changes."""
        if event.text_area.id == "prompt":
            prompt = cast(PromptInput, event.text_area)
            self._sync_prompt_shell_mode(prompt)
            self._rebuild_completions(prompt)

    async def action_submit_prompt(self) -> None:
        """Submit normally, using steering while an agent run is active."""
        await self._submit_prompt_from_editor(streaming_behavior="steer")

    async def action_submit_follow_up(self) -> None:
        """Submit as a follow-up while an agent run is active."""
        await self._submit_prompt_from_editor(streaming_behavior="follow_up")

    async def _submit_prompt_from_editor(
        self,
        *,
        streaming_behavior: Literal["steer", "follow_up"],
    ) -> None:
        prompt = self.query_one("#prompt", PromptInput)
        self._rebuild_completions(prompt)
        selected = self.completion_state.selected
        if selected is not None and selected.apply(prompt.text) != prompt.text:
            self.action_accept_completion()
            return
        content = prompt.text.strip()
        if not content:
            prompt.text = ""
            return
        if self._is_compaction_active():
            self.state.add_item("status", "Compaction is still running; wait to submit.")
            self._render_state()
            return

        terminal = parse_terminal_command(content)
        if terminal is not None:
            if self.session.is_running or self.state.running:
                self.state.add_item("status", "Wait for the active operation before running shell.")
                self._render_state()
                return
            prompt.text = ""
            self.completion_state = CompletionState()
            self._refresh_completions()
            prompt.disabled = True
            self.run_worker(
                self._run_terminal_command(
                    terminal.command,
                    add_to_context=terminal.add_to_context,
                ),
                name="axis-terminal-command",
                group="terminal",
                exclusive=True,
                exit_on_error=False,
            )
            return

        command = self._handle_command(content)
        if command.handled:
            if self._is_compaction_active() and (
                command.new_session_requested
                or command.resume_session_id is not None
                or command.resume_picker_requested
                or command.compact_instructions is not None
            ):
                self.state.add_item("status", "Compaction is still running; wait to submit.")
                self._render_state()
                return
            if command.compact_instructions is not None:
                self._sync_queue_state()
                if self.session.is_running or self.state.running or self.state.queued_message_count:
                    self.state.add_item(
                        "status",
                        "Wait for the current agent turn and queued messages before compacting.",
                    )
                    self._render_state()
                    return
            prompt.text = ""
            self.completion_state = CompletionState()
            self._refresh_completions()
            await self._apply_command_result(content, command)
            return

        prompt.text = ""
        self.completion_state = CompletionState()
        self._refresh_completions()
        self._cancel_requested = False
        if self.session.is_running or self.state.running:
            await self._queue_prompt(content, streaming_behavior=streaming_behavior)
            return
        self._prompt_worker = self.run_worker(
            self._run_prompt(content),
            name="axis-agent-run",
            group="agent",
            exclusive=True,
            exit_on_error=False,
        )

    def _handle_command(self, text: str) -> CommandResult:
        handle = getattr(self.session, "handle_command", None)
        if callable(handle):
            return cast(CommandResult, handle(text))
        return self.command_registry.execute(cast(CommandSession, self.session), text)

    async def _apply_command_result(self, text: str, command: CommandResult) -> None:
        message = command.message
        if command.reload_requested:
            reload_resources = getattr(self.session, "reload", None)
            if not callable(reload_resources):
                message = "Could not reload: resource reload is unavailable"
            else:
                try:
                    summary = await reload_resources()
                except Exception as exc:  # UI boundary: surface reload failure
                    message = f"Could not reload: {exc}"
                else:
                    message = format_reload_summary(summary)
                    self.state.set_skills(tuple(getattr(self.session, "skills", ())))
        if command.new_session_requested:
            await self._new_session()
        if command.compact_instructions is not None:
            self._compaction_worker = self.run_worker(
                self._run_compaction(command.compact_instructions),
                name="axis-compaction",
                group="compaction",
                exclusive=True,
                exit_on_error=False,
            )
        if command.export_requested:
            export_session = getattr(self.session, "export", None)
            if callable(export_session):
                try:
                    path = await export_session(
                        command.export_destination,
                        format=command.export_format,
                    )
                except Exception as exc:
                    self._notify(f"Could not export session: {exc}", severity="error")
                else:
                    self._notify(f"Exported session to {path}")
            else:
                self._notify("Session export is unavailable.", severity="error")
        if command.resume_session_id is not None:
            await self._resume_session(command.resume_session_id)
        if command.resume_picker_requested:
            self.action_open_session_picker()
        if command.tree_picker_requested:
            await self._open_tree_picker()
        if command.rename_to is not None:
            rename = getattr(self.session, "rename", None)
            if callable(rename):
                try:
                    renamed = await rename(command.rename_to)
                except Exception as exc:
                    message = f"Could not rename session: {exc}"
                else:
                    self._notify(str(renamed))
            else:
                message = "Session manager is not available."
        if command.model_name is not None:
            await self._set_model(ModelChoice(self._provider_name(), command.model_name))
        if command.model_picker_requested:
            self._open_model_picker()
        if command.scoped_models_picker_requested:
            self._open_scoped_models_picker()
        if command.thinking_level is not None:
            await self._set_thinking_level(command.thinking_level)
        if command.login_picker_requested:
            self._open_login_picker()
        if command.login_provider is not None:
            self._open_login(command.login_provider)
        if command.logout_picker_requested:
            self._open_logout_picker()
        if command.logout_provider is not None:
            self._logout(command.logout_provider)
        if command.theme is not None:
            self._set_tui_theme(cast(TuiThemeName, command.theme))
        if command.theme_picker_requested:
            self._open_theme_picker()
        if command.voice_status_requested:
            self._show_command_message("/voice", self._voice_status_message())
        if command.voice_setup_requested:
            self._open_voice_setup()
        if message:
            if _command_name(text) == "reload":
                self.state.add_item("status", f"/reload\n{message}")
            elif not command.exit_requested:
                self._show_command_message(text, message)
        self._render_state()
        if command.exit_requested:
            self.action_exit_app()

    async def _new_session(self) -> None:
        await self._stop_active_prompt()
        new_session = getattr(self.session, "new_session", None)
        if not callable(new_session):
            self._notify("Session manager is not available.", severity="error")
            return
        try:
            message = await new_session()
        except Exception as exc:
            self._notify(f"Could not start session: {exc}", severity="error")
            return
        self._reload_visible_session()
        self._notify(str(message))

    async def _resume_session(self, session_id: str) -> None:
        await self._stop_active_prompt()
        resume = getattr(self.session, "resume", None)
        if not callable(resume):
            self._notify("Session manager is not available.", severity="error")
            return
        try:
            message = await resume(session_id)
        except Exception as exc:
            self._notify(f"Could not resume session: {exc}", severity="error")
            return
        self._reload_visible_session()
        self._notify(str(message))

    async def _run_compaction(self, instructions: str) -> None:
        compact = getattr(self.session, "compact", None)
        if not callable(compact):
            self._notify("Session compaction is unavailable.", severity="error")
            self._compaction_worker = None
            return
        self.state.clear()
        self.state.running = True
        self.state.add_item("status", "Compacting session…")
        self._render_state()
        try:
            message = await compact(instructions or None)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._notify(f"Could not compact session: {exc}", severity="error")
        else:
            self._reload_visible_session()
            self._notify(str(message))
        finally:
            self.state.running = False
            self._compaction_worker = None
            self._render_state()

    async def _stop_active_prompt(self) -> None:
        worker = self._prompt_worker
        if not (self.session.is_running or (worker is not None and not worker.is_finished)):
            return
        self.session.cancel()
        if worker is not None and not worker.is_finished:
            worker.cancel()
            with suppress(BaseException):
                await worker.wait()
        self._prompt_worker = None
        self.state.running = False

    def _reload_visible_session(self) -> None:
        self._request_context_usage = None
        self._request_context_turn = None
        self.state.clear()
        self.state.set_skills(tuple(getattr(self.session, "skills", ())))
        self.state.load_messages(tuple(getattr(self.session, "messages", ())))
        self._render_state()

    async def _queue_prompt(
        self,
        content: str,
        *,
        streaming_behavior: StreamingBehavior,
    ) -> None:
        try:
            async for event in self.session.prompt(
                content,
                streaming_behavior=streaming_behavior,
            ):
                self.adapter.apply(event)
        except Exception as exc:  # UI boundary: queue expansion and session errors
            self.adapter.apply(ErrorEvent(message=str(exc), recoverable=True))
        self._render_state()

    async def _run_prompt(self, content: str) -> None:
        try:
            async for event in self.session.prompt(content):
                self.adapter.apply(event)
                if isinstance(event, TurnStartEvent):
                    self._capture_request_context_usage(event.turn)
                await self._apply_streaming_transcript_event(event)
                if self.state.cancelled:
                    self._cancel_requested = False
                self._render_state(redraw_transcript=False)
        except Exception as exc:  # UI boundary: surface unexpected session failures
            self.adapter.apply(ErrorEvent(message=str(exc), recoverable=False))
            self.adapter.apply(AgentEndEvent())
            self._render_state()
        finally:
            self._prompt_worker = None
            try:
                prompt = self.query_one("#prompt", PromptInput)
            except NoMatches:
                pass
            else:
                prompt.focus()
                self._render_state()

    async def _show_tool_approval(
        self,
        tool: AgentTool,
        tool_call: ToolCall,
        signal: ToolCancellationToken | None = None,
    ) -> ToolApprovalDecision:
        del tool
        if signal is not None and signal.is_cancelled():
            return "deny"
        preview = build_tool_approval_preview(tool_call, cwd=self.session.cwd)
        decision = await self.push_screen_wait(ToolApprovalScreen(preview))
        return decision if decision in {"allow_once", "allow_session", "deny"} else "deny"

    async def _run_terminal_command(self, command: str, *, add_to_context: bool) -> None:
        self._terminal_active = True
        self.state.running = True
        item = self.state.add_item(
            "tool",
            f"$ {command.strip()}",
            always_show_tool_result=True,
        )
        self.query_one("#transcript", TranscriptView).follow_output()
        self._render_state()
        try:
            result = await self.session.run_terminal_command(
                command,
                add_to_context=add_to_context,
            )
        except Exception as exc:  # UI boundary: show local command failures in transcript
            item.tool_result_text = format_terminal_command_result_block(
                ok=False,
                added_to_context=add_to_context,
                output=str(exc),
            )
        else:
            item.text = f"$ {result.command}"
            item.tool_result_text = format_terminal_command_result_block(
                ok=result.ok,
                added_to_context=result.added_to_context,
                output=result.output,
            )
        finally:
            self._terminal_active = False
            self.state.running = False
            self._cancel_requested = False
            prompt = self.query_one("#prompt", PromptInput)
            prompt.disabled = False
            prompt.focus()
            self._render_state()

    def action_accept_completion(self) -> None:
        """Replace the selected completion span and put the cursor after it."""
        prompt = self.query_one("#prompt", PromptInput)
        selected = self.completion_state.selected
        if selected is None:
            return
        prompt.text = selected.apply(prompt.text)
        prompt.cursor_position = selected.start + len(selected.replacement)
        self._rebuild_completions(prompt)

    def action_completion_next(self) -> None:
        """Select the next completion, wrapping around."""
        self.completion_state = self.completion_state.select_next()
        self._refresh_completions()

    def action_completion_previous(self) -> None:
        """Select the previous completion, wrapping around."""
        self.completion_state = self.completion_state.select_previous()
        self._refresh_completions()

    def action_edit_queued_follow_up(self) -> bool:
        """Move the newest follow-up back into an empty prompt for editing."""
        pop_latest = getattr(self.session, "pop_latest_follow_up_message", None)
        if not callable(pop_latest):
            return False
        message = pop_latest()
        if message is None:
            return False
        prompt = self.query_one("#prompt", PromptInput)
        prompt.text = message
        prompt.cursor_position = len(message)
        self._sync_queue_state()
        self._render_state()
        return True

    def action_open_command_palette(self) -> None:
        """Open slash completion from the configured command-palette key."""
        prompt = self.query_one("#prompt", PromptInput)
        prompt.focus()
        prompt.text = "/"
        prompt.cursor_position = 1
        self._rebuild_completions(prompt)

    def action_open_session_picker(self) -> None:
        """Open project-scoped indexed sessions from Ctrl+R or /resume."""
        manager = getattr(self.session, "session_manager", None)
        if manager is None:
            self._notify("Session manager is not available.", severity="warning")
            return
        records = tuple(manager.list_sessions(self.session.cwd))
        if not records:
            self._notify("No sessions found.", severity="warning")
            return
        self.push_screen(
            SessionPickerScreen(records),
            callback=self._handle_session_picker_result,
        )

    def action_cycle_thinking(self) -> None:
        """Cycle the active provider's supported reasoning effort."""
        if self.state.running:
            self._notify("Axis is already working. Press Escape to cancel.")
            return
        self.run_worker(
            self._cycle_thinking_level(),
            name="axis-cycle-thinking",
            group="model-selection",
            exclusive=True,
            exit_on_error=False,
        )

    def action_cycle_model(self) -> None:
        """Cycle through models selected by /scoped-models."""
        if self.state.running:
            self._notify("Axis is already working. Press Escape to cancel.")
            return
        self.run_worker(
            self._cycle_scoped_model(),
            name="axis-cycle-model",
            group="model-selection",
            exclusive=True,
            exit_on_error=False,
        )

    def _handle_session_picker_result(self, session_id: str | None) -> None:
        if session_id is not None:
            self.run_worker(
                self._resume_session(session_id),
                name="axis-resume-session",
                group="session-navigation",
                exclusive=True,
                exit_on_error=False,
            )

    def _provider_name(self) -> str:
        value = getattr(self.session, "provider_name", "deepseek")
        return value if isinstance(value, str) and value else "deepseek"

    def _available_model_choices(self) -> tuple[ModelChoice, ...]:
        choices = getattr(self.session, "available_model_choices", None)
        if choices is not None:
            return tuple(cast(Sequence[ModelChoice], choices))
        return tuple(
            ModelChoice(self._provider_name(), model)
            for model in tuple(getattr(self.session, "available_models", (self.session.model,)))
        )

    def _open_model_picker(self) -> None:
        choices = self._available_model_choices()
        if not choices:
            self._notify(
                "No configured providers are usable. Run /login to set up a provider.",
                severity="warning",
            )
            return
        self.push_screen(
            ModelPickerScreen(
                choices,
                scoped_choices=tuple(getattr(self.session, "scoped_model_choices", ())),
                current=ModelChoice(self._provider_name(), self.session.model),
            ),
            callback=self._handle_model_picker_result,
        )

    def _open_scoped_models_picker(self) -> None:
        choices = self._available_model_choices()
        if not choices:
            self._notify(
                "No configured providers are usable. Run /login to set up a provider.",
                severity="warning",
            )
            return
        self.push_screen(
            ModelPickerScreen(
                choices,
                scoped_choices=tuple(getattr(self.session, "scoped_model_choices", ())),
                current=ModelChoice(self._provider_name(), self.session.model),
                on_toggle_scoped=self._toggle_scoped_model,
                picker_kind="scoped",
            )
        )

    def _toggle_scoped_model(self, choice: ModelChoice) -> tuple[ModelChoice, ...]:
        toggle = getattr(self.session, "toggle_scoped_model", None)
        if not callable(toggle):
            self._notify("Scoped model controls are not available.", severity="warning")
            return tuple(getattr(self.session, "scoped_model_choices", ()))
        try:
            return tuple(cast(Sequence[ModelChoice], toggle(choice)))
        except Exception as exc:
            self._notify(f"Could not update scoped models: {exc}", severity="error")
            return tuple(getattr(self.session, "scoped_model_choices", ()))

    def _handle_model_picker_result(self, choice: ModelChoice | None) -> None:
        if choice is not None:
            self.run_worker(
                self._set_model(choice),
                name="axis-switch-model",
                group="model-selection",
                exclusive=True,
                exit_on_error=False,
            )

    async def _set_model(self, choice: ModelChoice) -> None:
        setter = getattr(self.session, "set_model_choice", None)
        if not callable(setter):
            self._notify("Model controls are not available.", severity="warning")
            return
        try:
            result = setter(choice)
            message = await result if isawaitable(result) else result
        except Exception as exc:
            self._notify(f"Could not switch model: {exc}", severity="error")
            return
        self._notify(str(message))
        self._render_state()

    async def _set_thinking_level(self, level: str) -> None:
        setter = getattr(self.session, "set_thinking_level", None)
        if not callable(setter):
            self._notify("Thinking controls are not available.", severity="warning")
            return
        try:
            result = setter(level)
            message = await result if isawaitable(result) else result
        except Exception as exc:
            self._notify(f"Could not change thinking mode: {exc}", severity="error")
            return
        self._notify(str(message))
        self._render_state()

    async def _cycle_thinking_level(self) -> None:
        cycler = getattr(self.session, "cycle_thinking_level", None)
        if not callable(cycler):
            self._notify("Thinking controls are not available.", severity="warning")
            return
        try:
            result = cycler()
            message = await result if isawaitable(result) else result
        except Exception as exc:
            self._notify(f"Could not change thinking mode: {exc}", severity="error")
            return
        self._notify(str(message))
        self._render_state()

    async def _cycle_scoped_model(self) -> None:
        cycler = getattr(self.session, "cycle_scoped_model", None)
        if not callable(cycler):
            self._notify("Scoped model controls are not available.", severity="warning")
            return
        try:
            result = cycler()
            choice = await result if isawaitable(result) else result
        except Exception as exc:
            self._notify(f"Could not switch scoped model: {exc}", severity="error")
            return
        if isinstance(choice, ModelChoice):
            self._notify(f"Current model: {choice.provider_name}:{choice.model}")
        self._render_state()

    def _credential_store(self) -> FileCredentialStore:
        resources = getattr(self.session, "resource_paths", None)
        paths = getattr(resources, "paths", None)
        return FileCredentialStore(credentials_path(paths))

    def _open_voice_setup(self) -> None:
        if self.session.is_running or self.state.running:
            self._notify(
                "Wait for the active Agent operation before voice setup.", severity="warning"
            )
            return
        try:
            config = load_voice_config(self.paths)
            devices = list_audio_input_devices()
            has_key = resolve_voice_api_key(self.paths) is not None
        except Exception as exc:
            self._notify(f"Could not open voice setup: {exc}", severity="error")
            return
        self.push_screen(
            VoiceSetupScreen(
                devices,
                has_existing_key=has_key,
                current_device=config.input_device,
                tester=self._test_voice_setup,
            ),
            callback=self._handle_voice_setup_result,
        )

    async def _test_voice_setup(
        self,
        api_key: str,
        input_device: int | str | None,
        update: Callable[[str], None],
    ) -> str:
        key = api_key or resolve_voice_api_key(self.paths)
        if not key:
            raise RuntimeError("Enter a Volcengine ASR API key")
        config = replace(load_voice_config(self.paths), input_device=input_device)

        def listener(event: VoiceInputEvent) -> None:
            if event.type == "partial" and event.text:
                update(f"Listening: {event.text}")

        return await test_voice_configuration(config, key, listener=listener)

    def _handle_voice_setup_result(self, result: VoiceSetupResult | None) -> None:
        if result is None:
            return
        try:
            if result.api_key is not None:
                FileCredentialStore(credentials_path(self.paths)).set(
                    VOLCENGINE_ASR_CREDENTIAL_NAME,
                    result.api_key,
                )
            config = replace(
                load_voice_config(self.paths),
                input_device=result.input_device,
            )
            save_voice_config(config, self.paths)
        except Exception as exc:
            self._notify(f"Could not save voice setup: {exc}", severity="error")
            return
        self._notify("Voice input configured. Press F2 to start recording.")

    def _voice_status_message(self) -> str:
        try:
            config = load_voice_config(self.paths)
            configured = resolve_voice_api_key(self.paths) is not None
            devices = list_audio_input_devices()
        except Exception as exc:
            return f"Voice input status could not be loaded: {exc}"
        selected = "system default" if config.input_device is None else str(config.input_device)
        selected_device = next(
            (device.name for device in devices if device.index == config.input_device),
            selected,
        )
        return "\n".join(
            (
                "Context-aware voice input",
                f"- ASR: Volcengine Seed ASR 2.0 ({config.language})",
                f"- Credentials: {'configured' if configured else 'missing'}",
                f"- Microphone: {selected_device}",
                f"- Maximum recording: {config.max_recording_seconds:g} seconds",
                f"- Shortcut: {self.tui_settings.keybindings.voice_record}",
                "- Setup: /voice setup",
                "- Audio and raw transcripts are not persisted.",
            )
        )

    def _open_login_picker(self) -> None:
        self.push_screen(
            LoginProviderPickerScreen(BUILTIN_PROVIDER_CATALOG),
            callback=lambda provider: self._open_login(provider) if provider else None,
        )

    def _open_login(self, provider_name: str) -> None:
        provider = builtin_provider_entry(provider_name)
        if provider is None:
            self._notify(f"Unknown provider: {provider_name}", severity="error")
            return
        self.push_screen(
            LoginScreen(provider),
            callback=lambda key: self._handle_login_result(provider, key),
        )

    def _handle_login_result(
        self,
        provider: ProviderCatalogEntry,
        api_key: str | None,
    ) -> None:
        if api_key is None:
            return
        try:
            store = self._credential_store()
            store.set(provider.credential_name, api_key)
            resources = getattr(self.session, "resource_paths", None)
            paths = getattr(resources, "paths", None)
            settings_path = provider_settings_path(paths)
            fallback_settings = getattr(self.session, "provider_settings", None)
            settings = (
                load_provider_settings(paths)
                if settings_path.exists() or fallback_settings is None
                else fallback_settings
            )
            if all(item.name != provider.name for item in settings.providers):
                settings = upsert_provider(
                    settings,
                    provider_config_from_catalog_entry(provider.name),
                )
            save_provider_settings(settings, paths)
            reload_settings = getattr(self.session, "reload_provider_settings", None)
            if callable(reload_settings):
                reload_settings()
        except (OSError, ValueError, RuntimeError) as exc:
            self._notify(f"Could not save provider login: {exc}", severity="error")
            return
        self._notify(f"Saved API key for {provider.display_name}.")

    def _open_logout_picker(self) -> None:
        try:
            names = set(self._credential_store().names())
        except ValueError as exc:
            self._notify(f"Could not read stored credentials: {exc}", severity="error")
            return
        providers = tuple(
            provider for provider in BUILTIN_PROVIDER_CATALOG if provider.credential_name in names
        )
        if not providers:
            self._notify(
                "No stored credentials to remove. Environment variables are unchanged.",
                severity="warning",
            )
            return
        self.push_screen(
            LoginProviderPickerScreen(providers, title="Logout"),
            callback=lambda provider: self._logout(provider) if provider else None,
        )

    def _logout(self, provider_name: str) -> None:
        provider = builtin_provider_entry(provider_name)
        if provider is None:
            self._notify(f"Unknown provider: {provider_name}", severity="error")
            return
        try:
            removed = self._credential_store().delete(provider.credential_name)
        except ValueError as exc:
            self._notify(f"Could not log out: {exc}", severity="error")
            return
        if not removed:
            self._notify(
                "No stored credentials to remove. Environment variables are unchanged.",
                severity="warning",
            )
            return
        self._notify(
            f"Removed stored API key for {provider.display_name}. "
            "Environment variables and providers.json are unchanged."
        )

    async def _open_tree_picker(self) -> None:
        choices_method = getattr(self.session, "tree_choices", None)
        if not callable(choices_method):
            self._notify("Session tree is unavailable.", severity="warning")
            return
        try:
            choices = tuple(await choices_method())
        except Exception as exc:
            self._notify(f"Could not load session tree: {exc}", severity="error")
            return
        if not choices:
            self._notify("No branchable session entries.", severity="warning")
            return
        self.push_screen(
            TreePickerScreen(choices),
            callback=self._handle_tree_picker_result,
        )

    def _handle_tree_picker_result(self, selection: TreePickerSelection | None) -> None:
        if selection is not None:
            self.run_worker(
                self._branch_session(selection),
                name="axis-branch-session",
                group="session-navigation",
                exclusive=True,
                exit_on_error=False,
            )

    async def _branch_session(self, selection: TreePickerSelection) -> None:
        branch = getattr(self.session, "branch_to_entry", None)
        if not callable(branch):
            self._notify("Session branching is unavailable.", severity="error")
            return
        prompt = self.query_one("#prompt", PromptInput)
        if selection.summarize:
            prompt.disabled = True
            self.state.clear()
            self.state.running = True
            self.state.add_item("status", "Summarizing branch…")
            self._render_state()
        try:
            result = cast(
                SessionTreeBranchResult,
                await branch(
                    selection.entry_id,
                    summarize=selection.summarize,
                    custom_instructions=selection.custom_instructions,
                ),
            )
        except Exception as exc:
            self._notify(f"Could not branch session: {exc}", severity="error")
            self._reload_visible_session()
            return
        finally:
            self.state.running = False
            prompt.disabled = False
            prompt.focus()
        self._reload_visible_session()
        if result.input_prefill is not None:
            prompt = self.query_one("#prompt", PromptInput)
            prompt.text = result.input_prefill
            prompt.cursor_position = len(result.input_prefill)
        self._notify(result.message)

    def action_toggle_voice(self) -> None:
        """Start recording or stop the active recording for finalization."""
        controller = self._voice_controller
        if controller is not None and controller.active:
            if controller.state == "recording":
                self._voice_worker = self.run_worker(
                    self._stop_voice(controller),
                    name="axis-voice-stop",
                    group="voice",
                    exclusive=False,
                    exit_on_error=False,
                )
            return
        if isinstance(self.screen, ModalScreen):
            self._notify("Close the active dialog before starting voice input.", severity="warning")
            return
        prompt = self.query_one("#prompt", PromptInput)
        if prompt.disabled:
            self._notify(
                "Wait for the active editor operation before recording.", severity="warning"
            )
            return
        try:
            controller = self._voice_controller_factory(self._queue_voice_event)
        except Exception as exc:
            self._notify(str(exc), severity="error")
            return
        self._voice_controller = controller
        self._voice_anchor = (prompt.selection.start, prompt.selection.end)
        self._voice_editor_text = prompt.text
        self._voice_editor_cursor = prompt.cursor_position
        prompt.disabled = True
        prompt.set_voice_active(True)
        self._set_app_voice_binding(True)
        self._set_voice_status("Connecting to Volcengine ASR…")
        self._voice_worker = self.run_worker(
            self._start_voice(controller),
            name="axis-voice-start",
            group="voice",
            exclusive=False,
            exit_on_error=False,
        )

    async def _start_voice(self, controller: VoiceInputController) -> None:
        await controller.start(self._build_voice_context)

    async def _stop_voice(self, controller: VoiceInputController) -> None:
        await controller.stop()

    async def _cancel_voice(self) -> None:
        controller = self._voice_controller
        if controller is None:
            return
        await controller.cancel()

    async def _dispose_voice(self, controller: VoiceInputController) -> None:
        await controller.aclose()
        if self._voice_controller is controller:
            self._voice_controller = None
            self._voice_worker = None

    def _build_voice_context(self) -> VoiceContextSnapshot:
        messages = cast(Sequence[AgentMessage], tuple(getattr(self.session, "messages", ())))
        skills = tuple(getattr(self.session, "skills", ()))
        return build_voice_context_snapshot(
            messages=messages,
            editor_text=self._voice_editor_text,
            cursor=self._voice_editor_cursor,
            cwd=self.session.cwd,
            session_title=cast(str | None, getattr(self.session, "session_title", None)),
            skill_names=tuple(
                str(name) for skill in skills if (name := getattr(skill, "name", None))
            ),
        )

    def _handle_voice_event(self, event: VoiceInputEvent) -> None:
        if event.type == "connecting":
            self._set_voice_status("Connecting to Volcengine ASR…")
        elif event.type == "recording":
            self._set_voice_status("● Recording · F2 finishes · Escape discards")
        elif event.type == "partial":
            if event.message:
                self._notify(event.message, severity="warning")
            self._set_voice_status(f"● Recording · {event.text}" if event.text else event.message)
        elif event.type == "finalizing":
            self._set_voice_status("Finalizing speech recognition…")
        elif event.type == "polishing":
            self._set_voice_status("Polishing with DeepSeekV4pro…")
        elif event.type == "completed":
            prompt = self.query_one("#prompt", PromptInput)
            start, end = self._voice_anchor or (prompt.cursor_location, prompt.cursor_location)
            prompt.replace(event.text, start, end)
            self._unlock_voice_prompt()
            if event.breakdown is not None:
                self._request_context_usage = event.breakdown
                self._request_context_turn = None
            if event.used_fallback:
                self._notify(
                    "DeepSeek polishing failed; inserted the raw ASR transcript.",
                    severity="warning",
                )
            self._set_voice_status("")
            self._render_state()
            self._schedule_voice_disposal()
        elif event.type == "error":
            self._unlock_voice_prompt()
            self._set_voice_status("")
            self._notify(f"Voice input failed: {event.message}", severity="error")
            self._schedule_voice_disposal()
        elif event.type == "cancelled":
            self._unlock_voice_prompt()
            self._set_voice_status("")
            self._notify("Voice input cancelled.")
            self._schedule_voice_disposal()

    def _queue_voice_event(self, event: VoiceInputEvent) -> None:
        self.call_later(self._handle_voice_event, event)

    def _unlock_voice_prompt(self) -> None:
        with suppress(NoMatches):
            prompt = self.query_one("#prompt", PromptInput)
            prompt.disabled = False
            prompt.set_voice_active(False)
            prompt.focus()
        self._set_app_voice_binding(False)
        self._voice_anchor = None

    def _set_voice_status(self, text: str) -> None:
        with suppress(NoMatches):
            status = self.query_one("#voice-status", Static)
            status.display = bool(text)
            status.update(text)

    def _set_app_voice_binding(self, active: bool) -> None:
        self._bindings = BindingsMap(
            _app_bindings(self.tui_settings.keybindings, voice_active=active)
        )
        self.refresh_bindings()

    def _schedule_voice_disposal(self) -> None:
        controller = self._voice_controller
        if controller is not None:
            self.run_worker(
                self._dispose_voice(controller),
                name="axis-voice-dispose",
                group="voice-dispose",
                exclusive=True,
                exit_on_error=False,
            )

    def action_cancel_run(self) -> None:
        """Request cooperative cancellation without killing the UI worker."""
        if self._voice_controller is not None and self._voice_controller.active:
            self.run_worker(
                self._cancel_voice(),
                name="axis-voice-cancel",
                group="voice",
                exclusive=True,
                exit_on_error=False,
            )
            return
        if self._is_compaction_active():
            worker = self._compaction_worker
            if worker is not None:
                worker.cancel()
            self._compaction_worker = None
            self.state.running = False
            self._reload_visible_session()
            self._notify("Cancelled compaction.")
            return
        if not (self.session.is_running or self.state.running):
            return
        if not self._cancel_requested:
            self.state.add_item("status", "Cancellation requested…")
        self._cancel_requested = True
        self.session.cancel()
        self._render_state()

    def action_exit_app(self) -> None:
        """Cancel active work and exit the application."""
        if self._voice_controller is not None and self._voice_controller.active:
            asyncio.create_task(self._voice_controller.cancel())
        if self.session.is_running or self.state.running:
            self.session.cancel()
        self.exit()

    def action_toggle_tool_results(self) -> None:
        """Expand or collapse all recorded tool results."""
        self.state.toggle_tool_results()
        self._render_state()

    def action_toggle_thinking(self) -> None:
        """Show or hide recorded reasoning blocks."""
        if isinstance(self.screen, TreePickerScreen):
            self.run_worker(
                self.screen.action_toggle_tool_calls(),
                name="axis-tree-toggle-tools",
                group="picker",
                exclusive=True,
                exit_on_error=False,
            )
            return
        self.state.toggle_thinking()
        self._render_state()

    def _show_command_message(self, command_text: str, message: str) -> None:
        self.push_screen(CommandOutputScreen(_command_output_title(command_text), message))

    def _notify(
        self,
        message: str,
        *,
        severity: Literal["information", "warning", "error"] = "information",
    ) -> None:
        self.notify(message, severity=severity, markup=False)

    def _is_compaction_active(self) -> bool:
        worker = self._compaction_worker
        return worker is not None and not worker.is_finished and not worker.is_cancelled

    def _open_theme_picker(self) -> None:
        self.push_screen(
            ThemePickerScreen(self.tui_settings.theme),
            callback=self._handle_theme_picker_result,
        )

    def _handle_theme_picker_result(self, theme: TuiThemeName | None) -> None:
        if theme is not None:
            self._set_tui_theme(theme)

    def _set_tui_theme(self, theme: TuiThemeName) -> None:
        self.tui_settings = TuiSettings(
            keybindings=self.tui_settings.keybindings,
            theme=theme,
            auto_copy_selection=self.tui_settings.auto_copy_selection,
        )
        self.theme = _textual_base_theme(theme)
        save_tui_settings(self.tui_settings)
        self.query_one(
            "#prompt", PromptInput
        ).shell_mode_style = self.tui_settings.resolved_theme.accent
        self.refresh_css(animate=False)
        self._render_state()

    async def _apply_streaming_transcript_event(self, event: AgentEvent) -> None:
        """Update active Markdown widgets without rebuilding prior messages."""
        transcript = self.query_one("#transcript", TranscriptView)
        theme = self.tui_settings.resolved_theme
        follow = transcript.is_vertical_scroll_end or transcript.is_anchored
        if isinstance(event, MessageStartEvent):
            return
        if isinstance(event, ThinkingDeltaEvent):
            await transcript.append_thinking_delta(
                event.delta,
                theme=theme,
                show_thinking=self.state.show_thinking,
                scroll_end=follow,
            )
            return
        if isinstance(event, MessageDeltaEvent):
            await transcript.append_assistant_delta(
                event.delta,
                theme=theme,
                scroll_end=follow,
            )
            return
        if isinstance(event, MessageEndEvent) and event.message.role == "assistant":
            await transcript.finish_assistant_message(event.message.content or None)
            return
        transcript.update_from_state(self.state, theme=theme)

    def _render_state(self, *, redraw_transcript: bool = True) -> None:
        try:
            transcript = self.query_one("#transcript", TranscriptView)
        except NoMatches:
            return
        if redraw_transcript:
            transcript.update_from_state(self.state, theme=self.tui_settings.resolved_theme)
        self._sync_queue_state()
        queued = self.query_one("#queued-messages", Static)
        queued.display = self.state.queued_message_count > 0
        queued.update(_render_queued_messages(self.state, theme=self.tui_settings.resolved_theme))
        self.query_one("#sidebar", SessionSidebar).update_from_session(
            self.session,
            theme=self.tui_settings.resolved_theme,
        )
        self.query_one("#compact-session-info", CompactSessionInfo).update_from_session(
            self.session,
            theme=self.tui_settings.resolved_theme,
        )
        request_usage = self.query_one("#request-context-usage", Static)
        request_usage.display = self._request_context_usage is not None
        if self._request_context_usage is not None:
            request_usage.update(
                render_request_context_usage(
                    self._request_context_usage,
                    turn=self._request_context_turn,
                    theme=self.tui_settings.resolved_theme,
                )
            )
        self._sync_activity_indicator()
        self._refresh_footer_bindings()

    def _capture_request_context_usage(self, turn: int) -> None:
        try:
            usage = getattr(self.session, "context_usage", None)
        except Exception:  # UI telemetry must never interrupt an agent request
            return
        if not isinstance(usage, ContextUsageEstimate):
            return
        self._request_context_usage = usage
        self._request_context_turn = turn

    def _rebuild_completions(self, prompt: PromptInput) -> None:
        prefix = prompt.text[: prompt.cursor_position]
        manager = getattr(self.session, "session_manager", None)
        resume_options: tuple[CompletionOption, ...] = ()
        if manager is not None:
            resume_options = tuple(
                CompletionOption(
                    record.id,
                    f"{record.title or 'Untitled'} · {record.model}",
                )
                for record in manager.list_sessions(self.session.cwd)
            )
        self.completion_state = build_completion_state(
            prefix,
            commands=tuple(
                CompletionCommand(
                    name=command.name,
                    description=command.description,
                    aliases=command.aliases,
                    search_terms=command.search_terms,
                )
                for command in self.command_registry.list_commands()
            ),
            skills=tuple(getattr(self.session, "skills", ())),
            prompt_templates=tuple(getattr(self.session, "prompt_templates", ())),
            argument_options={
                "model": tuple(
                    CompletionOption(model, f"Model on {self._provider_name()}")
                    for model in tuple(getattr(self.session, "available_models", ()))
                ),
                "thinking": tuple(
                    CompletionOption(
                        level,
                        THINKING_LEVEL_DESCRIPTIONS.get(level),
                    )
                    for level in tuple(getattr(self.session, "available_thinking_levels", ()))
                ),
                "login": tuple(
                    CompletionOption(entry.name, entry.display_name)
                    for entry in BUILTIN_PROVIDER_CATALOG
                ),
                "logout": tuple(
                    CompletionOption(entry.name, entry.display_name)
                    for entry in BUILTIN_PROVIDER_CATALOG
                ),
                "theme": tuple(
                    CompletionOption(name, "Set TUI theme") for name in BUILTIN_TUI_THEME_NAMES
                ),
                "resume": resume_options,
            },
            cwd=self.session.cwd,
            shell_paths_enabled=True,
        )
        self._refresh_completions()

    def _sync_queue_state(self) -> None:
        queue_update = getattr(self.session, "queue_update_event", None)
        if callable(queue_update):
            self.adapter.apply(queue_update())

    def _sync_prompt_shell_mode(self, prompt: PromptInput) -> None:
        prompt.set_class(_is_terminal_command_prompt(prompt.text), "-shell-mode")
        prompt.refresh()
        self._apply_activity_indicator()

    def _sync_activity_indicator(self) -> None:
        if self.state.running:
            if self._activity_timer is None:
                self._activity_timer = self.set_interval(
                    ACTIVITY_TICK_SECONDS,
                    self._tick_activity,
                    name="activity-indicator",
                )
            else:
                self._activity_timer.resume()
        else:
            self._activity_frame = 0
            if self._activity_timer is not None:
                self._activity_timer.pause()
        self._apply_activity_indicator()

    def _tick_activity(self) -> None:
        if self.state.running:
            self._activity_frame += 1
            self._apply_activity_indicator()

    def _apply_activity_indicator(self) -> None:
        try:
            prompt = self.query_one("#prompt", PromptInput)
            prefix = self.query_one("#prompt-prefix", Static)
        except NoMatches:
            return
        theme = self.tui_settings.resolved_theme
        prompt.styles.border = (
            "tall",
            _textual_color(
                _activity_prompt_border_color(
                    theme,
                    frame=self._activity_frame,
                    running=self.state.running,
                    shell_mode=_is_terminal_command_prompt(prompt.text),
                )
            ),
        )
        prefix.update(
            _render_activity_indicator(
                theme,
                frame=self._activity_frame,
                running=self.state.running,
            )
        )

    def _refresh_completions(self) -> None:
        try:
            suggestions = self.query_one("#autocomplete", Static)
        except NoMatches:
            return
        suggestions.display = bool(self.completion_state.items)
        suggestions.update(
            render_completion_suggestions(
                self.completion_state,
                theme=self.tui_settings.resolved_theme,
            )
        )
        self._refresh_footer_bindings()

    def _refresh_footer_bindings(self) -> None:
        try:
            prompt = self.query_one("#prompt", PromptInput)
        except NoMatches:
            return
        prompt.set_footer_mode(_prompt_footer_mode(self.state, self.completion_state))

    def _update_responsive_layout(self, width: int, height: int) -> None:
        self.set_class(
            width < SIDEBAR_MIN_WIDTH or height < SIDEBAR_MIN_HEIGHT,
            "-hide-sidebar",
        )


async def run_tui_app(
    session: TuiSession,
    *,
    tui_settings: TuiSettings | None = None,
    startup_message: str | None = None,
    initial_prompt: str | None = None,
    paths: AxisPaths | None = None,
) -> None:
    """Run the basic Axis TUI in the caller's current async loop."""
    await AxisTuiApp(
        session,
        tui_settings=tui_settings if tui_settings is not None else load_tui_settings(),
        startup_message=startup_message,
        initial_prompt=initial_prompt,
        paths=paths,
    ).run_async()


def _prompt_footer_mode(
    state: TuiState,
    completion_state: CompletionState,
) -> Literal["normal", "completion", "running"]:
    if completion_state.items:
        return "completion"
    if state.running:
        return "running"
    return "normal"


def _key_hint(key: str) -> str:
    return "+".join(part.capitalize() for part in key.split("+"))


def _prompt_bindings(
    keybindings: TuiKeybindings,
    *,
    mode: Literal["normal", "completion", "running"],
    voice_active: bool = False,
) -> list[Binding]:
    if mode == "completion":
        visible = [
            Binding(
                keybindings.accept_completion,
                "accept_completion",
                "Complete",
                key_display=f"{_key_hint(keybindings.accept_completion)}/Enter",
                priority=True,
            ),
            Binding(
                keybindings.completion_next,
                "completion_next",
                "Choose",
                key_display=(
                    f"{_key_hint(keybindings.completion_previous)}/"
                    f"{_key_hint(keybindings.completion_next)}"
                ),
                priority=True,
            ),
            Binding(keybindings.cancel, "cancel", "Close", priority=True),
            Binding(
                keybindings.voice_record,
                "toggle_voice",
                "Stop voice" if voice_active else "Voice",
                priority=True,
            ),
        ]
        return [*visible, *_hidden_prompt_bindings(keybindings, visible)]
    if mode == "running":
        visible = [
            Binding("enter", "submit_prompt", "Steer", priority=True),
            Binding(
                keybindings.queue_follow_up,
                "submit_follow_up",
                "Follow-up",
                priority=True,
            ),
            Binding(keybindings.cancel, "cancel", "Cancel", priority=True),
            Binding(
                keybindings.toggle_thinking,
                "toggle_thinking",
                "Thinking",
                priority=True,
            ),
            Binding(
                keybindings.toggle_tool_results,
                "toggle_tool_results",
                "Tools",
                priority=True,
            ),
            Binding(
                keybindings.voice_record,
                "toggle_voice",
                "Stop voice" if voice_active else "Voice",
                priority=True,
            ),
        ]
        return [*visible, *_hidden_prompt_bindings(keybindings, visible)]
    visible = [
        Binding("enter", "submit_prompt", "Submit", priority=True),
        Binding("shift+enter", "insert_newline", "Newline", priority=True),
        Binding(
            keybindings.voice_record,
            "toggle_voice",
            "Stop voice" if voice_active else "Voice",
            priority=True,
        ),
        Binding(
            keybindings.command_palette,
            "open_command_palette",
            "Commands",
            priority=True,
        ),
        Binding(
            keybindings.session_picker,
            "open_session_picker",
            "Sessions",
            priority=True,
        ),
        Binding(keybindings.thinking_cycle, "cycle_thinking", "Thinking", priority=True),
        Binding(keybindings.model_cycle, "cycle_model", "Model", priority=True),
        Binding(keybindings.copy_message, "clear_prompt", "Clear", priority=True),
        Binding(keybindings.quit, "quit", "Quit", priority=True),
    ]
    return [*visible, *_hidden_prompt_bindings(keybindings, visible)]


def _hidden_prompt_bindings(
    keybindings: TuiKeybindings,
    visible: Sequence[Binding],
) -> list[Binding]:
    visible_keys = {key for binding in visible for key in binding.key.split(",")}
    candidates = (
        (keybindings.command_palette, "open_command_palette"),
        (keybindings.session_picker, "open_session_picker"),
        (keybindings.queue_follow_up, "submit_follow_up"),
        (keybindings.thinking_cycle, "cycle_thinking"),
        (keybindings.model_cycle, "cycle_model"),
        (keybindings.toggle_tool_results, "toggle_tool_results"),
        (keybindings.toggle_thinking, "toggle_thinking"),
        (keybindings.copy_message, "clear_prompt"),
        (keybindings.voice_record, "toggle_voice"),
        (keybindings.accept_completion, "accept_completion"),
        (keybindings.completion_next, "completion_next"),
        (keybindings.completion_previous, "completion_previous"),
        (keybindings.quit, "quit"),
    )
    return [
        Binding(key, action, show=False, priority=True)
        for key, action in candidates
        if key not in visible_keys
    ]


def _app_bindings(
    keybindings: TuiKeybindings,
    *,
    voice_active: bool = False,
) -> list[Binding]:
    """Bind only actions implemented by the current incremental frontend."""
    return [
        Binding(keybindings.cancel, "cancel_run", "Cancel", priority=True),
        Binding(
            keybindings.voice_record,
            "toggle_voice",
            "Stop voice" if voice_active else "Voice",
            priority=True,
        ),
        Binding(
            keybindings.command_palette,
            "open_command_palette",
            "Commands",
            priority=True,
        ),
        Binding(
            keybindings.session_picker,
            "open_session_picker",
            "Sessions",
            priority=True,
        ),
        Binding(
            keybindings.toggle_tool_results,
            "toggle_tool_results",
            "Tool results",
            priority=True,
        ),
        Binding(
            keybindings.toggle_thinking,
            "toggle_thinking",
            "Thinking tokens",
            priority=True,
        ),
        Binding(
            keybindings.thinking_cycle,
            "cycle_thinking",
            "Thinking mode",
            priority=True,
        ),
        Binding(
            keybindings.model_cycle,
            "cycle_model",
            "Model",
            priority=True,
        ),
        Binding(keybindings.quit, "exit_app", "Quit", priority=True),
    ]


def _command_name(text: str) -> str:
    return text.split(maxsplit=1)[0].removeprefix("/").casefold()


def _model_choice_label(
    choice: ModelChoice,
    *,
    current: ModelChoice,
    scoped: bool,
) -> str:
    active_marker = "●" if choice == current else " "
    scoped_marker = "★" if scoped else " "
    return f"{active_marker} {scoped_marker} {choice.provider_name}:{choice.model}"


def _command_output_title(text: str) -> str:
    return f"/{_command_name(text) or 'command'}"


def _render_queued_messages(state: TuiState, *, theme: TuiTheme) -> Group:
    """Render queued steering and follow-up prompts as one-line previews."""
    rows: list[Text] = []
    for message in state.queued_steering:
        row = Text("↪ steering · queued: ", style=theme.muted_text)
        row.append(_queued_message_preview(message), style=theme.prompt_text)
        rows.append(row)
    for message in state.queued_follow_up:
        row = Text("↳ follow-up · queued: ", style=theme.muted_text)
        row.append(_queued_message_preview(message), style=theme.prompt_text)
        rows.append(row)
    return Group(*rows)


def _queued_message_preview(message: str) -> str:
    lines = message.splitlines()
    return lines[0] if lines else ""


def _is_terminal_command_prompt(text: str) -> bool:
    return _terminal_command_prefix_span(text) is not None


def _terminal_command_prefix_span(text: str) -> tuple[int, int] | None:
    """Return the leading ``!`` or ``!!`` span, allowing initial whitespace."""
    leading = len(text) - len(text.lstrip())
    stripped = text[leading:]
    if stripped.startswith("!!"):
        return leading, leading + 2
    if stripped.startswith("!"):
        return leading, leading + 1
    return None


def _activity_prompt_border_color(
    theme: TuiTheme,
    *,
    frame: int,
    running: bool,
    shell_mode: bool,
) -> str:
    """Choose a stable border, reserving accent for explicit shell mode."""
    del frame, running
    return theme.accent if shell_mode else theme.prompt_border


def _render_activity_indicator(theme: TuiTheme, *, frame: int, running: bool) -> Text:
    """Render the Axis prompt marker or a vertically moving activity square."""
    if not running:
        return Text("A", style=f"bold {theme.accent}")
    cycle_length = (ACTIVITY_INDICATOR_HEIGHT - 1) * 2
    position = frame % cycle_length
    active_row = position if position < ACTIVITY_INDICATOR_HEIGHT else cycle_length - position
    direction = 1 if position < ACTIVITY_INDICATOR_HEIGHT else -1
    if theme.accent.startswith("#") and theme.screen_background.startswith("#"):
        trail = {
            active_row: theme.accent,
            active_row - direction: _blend_hex_colors(
                theme.accent,
                theme.screen_background,
                fraction=0.35,
            ),
            active_row - (direction * 2): _blend_hex_colors(
                theme.accent,
                theme.screen_background,
                fraction=0.65,
            ),
        }
    else:
        trail = {
            active_row: theme.accent,
            active_row - direction: f"dim {theme.accent}",
            active_row - (direction * 2): f"dim {theme.accent}",
        }
    rendered = Text()
    for row in range(ACTIVITY_INDICATOR_HEIGHT):
        color = trail.get(row)
        rendered.append("■" if color is not None else " ", style=color)
        if row < ACTIVITY_INDICATOR_HEIGHT - 1:
            rendered.append("\n")
    return rendered


def _blend_hex_colors(start: str, end: str, *, fraction: float) -> str:
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    channels = tuple(
        round(left + (right - left) * fraction)
        for left, right in zip(start_rgb, end_rgb, strict=True)
    )
    return f"#{channels[0]:02x}{channels[1]:02x}{channels[2]:02x}"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    normalized = color.removeprefix("#")
    if len(normalized) != 6:
        raise ValueError(f"Expected #rrggbb color, got {color!r}")
    return (
        int(normalized[0:2], 16),
        int(normalized[2:4], 16),
        int(normalized[4:6], 16),
    )


def _theme_css_variables(theme: TuiTheme) -> dict[str, str]:
    """Translate typed theme values into Textual CSS variables."""
    return {
        "axis-screen-background": _textual_color(theme.screen_background),
        "axis-screen-overlay-background": (
            "transparent"
            if theme.screen_background == "default"
            else f"{_textual_color(theme.screen_background)} 70%"
        ),
        "axis-screen-text": _textual_color(theme.screen_text),
        "axis-chrome-background": _textual_color(theme.chrome_background),
        "axis-chrome-text": _textual_color(theme.chrome_text),
        "axis-muted-text": _textual_color(theme.muted_text),
        "axis-sidebar-background": _textual_color(theme.sidebar_background),
        "axis-border": _textual_color(theme.border),
        "axis-transcript-background": _textual_color(theme.transcript_background),
        "axis-prompt-background": _textual_color(theme.prompt_background),
        "axis-prompt-text": _textual_color(theme.prompt_text),
        "axis-prompt-border": _textual_color(theme.prompt_border),
        "axis-autocomplete-background": _textual_color(theme.autocomplete_background),
        "axis-accent": _textual_color(theme.accent),
        "axis-highlight-background": _textual_color(theme.highlight_background),
        "axis-highlight-text": _textual_color(theme.highlight_text),
        "axis-markdown-highlight": _textual_color(theme.markdown_heading),
        "axis-markdown-table-header": _textual_color(theme.markdown_table_header),
        "axis-markdown-table-border": _textual_color(theme.markdown_table_border),
        "axis-markdown-inline-code": _textual_color(theme.markdown_inline_code),
        "axis-markdown-code-block-background": _textual_color(theme.markdown_code_block_background),
        "axis-markdown-link": _textual_color(theme.markdown_link),
        "axis-markdown-bullet": _textual_color(theme.markdown_bullet),
    }


_RICH_TO_TEXTUAL_ANSI_COLORS = {
    "default": "ansi_default",
    "black": "ansi_black",
    "red": "ansi_red",
    "green": "ansi_green",
    "yellow": "ansi_yellow",
    "blue": "ansi_blue",
    "magenta": "ansi_magenta",
    "cyan": "ansi_cyan",
    "white": "ansi_white",
    "bright_black": "ansi_bright_black",
    "bright_red": "ansi_bright_red",
    "bright_green": "ansi_bright_green",
    "bright_yellow": "ansi_bright_yellow",
    "bright_blue": "ansi_bright_blue",
    "bright_magenta": "ansi_bright_magenta",
    "bright_cyan": "ansi_bright_cyan",
    "bright_white": "ansi_bright_white",
}


def _textual_color(color: str) -> str:
    """Translate Rich ANSI names while preserving terminal palette semantics."""
    return _RICH_TO_TEXTUAL_ANSI_COLORS.get(color, color)


def _textual_base_theme(theme: TuiThemeName) -> str:
    """Select Textual's renderer mode beneath an Axis theme."""
    if theme == "terminal-native":
        return "ansi-dark"
    if theme == "axis-light":
        return "textual-light"
    return "textual-dark"
