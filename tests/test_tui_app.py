"""Headless tests for Axis's Tau-style transcript frontend."""

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from rich.console import Console
from textual.geometry import Offset
from textual.selection import SELECT_ALL, Selection
from textual.widgets import Button, Footer, Input, ListView, Static, TextArea

import axis_coding.tui.app as tui_app
from axis_agent import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    ErrorEvent,
    JsonlSessionStorage,
    LeafEntry,
    MessageEntry,
    QueueUpdateEvent,
    ToolCall,
    ToolResultMessage,
    TurnEndEvent,
    TurnStartEvent,
    UserMessage,
)
from axis_agent.types import JSONValue
from axis_ai import (
    FakeProvider,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
    ProviderThinkingDeltaEvent,
)
from axis_coding import (
    AxisPaths,
    AxisResourcePaths,
    CodingSession,
    CodingSessionConfig,
    ContextUsageEstimate,
    FileCredentialStore,
    ModelChoice,
    ProjectContextFile,
    ProviderSettings,
    RequestContextBreakdown,
    RequestContextPart,
    ScopedModelConfig,
    SessionManager,
    TerminalCommandResult,
    builtin_provider_entry,
    load_provider_settings,
)
from axis_coding.credentials import credentials_path
from axis_coding.prompt_templates import PromptTemplate
from axis_coding.skills import Skill
from axis_coding.tui import (
    AxisTuiApp,
    BranchSummaryInstructionsScreen,
    CommandOutputScreen,
    CompactSessionInfo,
    LoginProviderPickerScreen,
    LoginScreen,
    ModelPickerScreen,
    PromptInput,
    SessionPickerScreen,
    SessionSidebar,
    StreamingTranscriptMessageWidget,
    ThemePickerScreen,
    ToolApprovalScreen,
    TranscriptMessageWidget,
    TranscriptView,
    TreePickerScreen,
    VoiceSetupScreen,
    render_compact_session_info,
    render_request_context_usage,
    render_session_sidebar,
)
from axis_coding.tui.config import TuiKeybindings, TuiSettings
from axis_coding.voice import (
    AudioInputDevice,
    VoiceInputEvent,
    load_voice_config,
)
from axis_coding.voice.config import VOLCENGINE_ASR_CREDENTIAL_NAME


async def wait_until(predicate: Callable[[], bool], *, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        await asyncio.sleep(0.01)


def visible_footer_bindings(app: AxisTuiApp) -> dict[str, str]:
    return {
        binding.description: binding.key_display or binding.key
        for _, binding, _enabled, _tooltip in app.screen.active_bindings.values()
        if binding.show
    }


class CancellableSession:
    def __init__(
        self,
        cwd: Path,
        *,
        messages: tuple[UserMessage | AssistantMessage | ToolResultMessage, ...] = (),
    ) -> None:
        self.cwd = cwd
        self.model = "fake"
        self.provider_name = "deepseek"
        self.available_models = ("fake", "fake-fast")
        self.available_providers = ("deepseek",)
        self.available_model_choices = (
            ModelChoice("deepseek", "fake"),
            ModelChoice("deepseek", "fake-fast"),
        )
        self.scoped_model_choices = self.available_model_choices
        self.thinking_level = "high"
        self.available_thinking_levels = ("high", "xhigh")
        self.thinking_unavailable_reason = None
        self.messages = messages
        self.tools = ()
        self.skills = ()
        self.prompt_templates = ()
        self.context_files = ()
        self.context_token_estimate = 12_034
        self.context_window_tokens = 128_000
        self.auto_compact_token_threshold = None
        self._running = False
        self.cancel_called = False
        self._cancel_event = asyncio.Event()
        self.queued_steering_messages: tuple[str, ...] = ()
        self.queued_follow_up_messages: tuple[str, ...] = ()
        self.terminal_commands: list[tuple[str, bool]] = []
        self.model_changes: list[ModelChoice] = []

    @property
    def is_running(self) -> bool:
        return self._running

    async def prompt(
        self,
        content: str,
        *,
        streaming_behavior: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        if streaming_behavior == "steer":
            self.queued_steering_messages = (*self.queued_steering_messages, content)
            yield self.queue_update_event()
            return
        if streaming_behavior == "follow_up":
            self.queued_follow_up_messages = (*self.queued_follow_up_messages, content)
            yield self.queue_update_event()
            return
        self._running = True
        try:
            yield AgentStartEvent()
            yield TurnStartEvent(turn=1)
            await self._cancel_event.wait()
            yield ErrorEvent(message="Agent run cancelled", recoverable=True)
            yield TurnEndEvent(turn=1)
            yield AgentEndEvent()
        finally:
            self._running = False

    async def set_model_choice(self, choice: ModelChoice) -> str:
        self.model_changes.append(choice)
        self.provider_name = choice.provider_name
        self.model = choice.model
        return f"Current model: {choice.provider_name}:{choice.model}"

    async def cycle_scoped_model(self) -> ModelChoice:
        current = ModelChoice(self.provider_name, self.model)
        index = self.scoped_model_choices.index(current)
        selected = self.scoped_model_choices[(index + 1) % len(self.scoped_model_choices)]
        await self.set_model_choice(selected)
        return selected

    async def set_thinking_level(self, level: str) -> str:
        self.thinking_level = level
        return f"Thinking mode: {level}"

    async def cycle_thinking_level(self) -> str:
        index = self.available_thinking_levels.index(self.thinking_level)
        self.thinking_level = self.available_thinking_levels[
            (index + 1) % len(self.available_thinking_levels)
        ]
        return f"Thinking mode: {self.thinking_level}"

    def toggle_scoped_model(self, choice: ModelChoice) -> tuple[ModelChoice, ...]:
        if choice in self.scoped_model_choices:
            self.scoped_model_choices = tuple(
                item for item in self.scoped_model_choices if item != choice
            )
        else:
            self.scoped_model_choices = (*self.scoped_model_choices, choice)
        return self.scoped_model_choices

    def reload_provider_settings(self) -> None:
        return None

    def cancel(self) -> None:
        self.cancel_called = True
        self._cancel_event.set()

    def queue_update_event(self) -> QueueUpdateEvent:
        return QueueUpdateEvent(
            steering=self.queued_steering_messages,
            follow_up=self.queued_follow_up_messages,
        )

    def pop_latest_follow_up_message(self) -> str | None:
        if not self.queued_follow_up_messages:
            return None
        message = self.queued_follow_up_messages[-1]
        self.queued_follow_up_messages = self.queued_follow_up_messages[:-1]
        return message

    async def run_terminal_command(
        self,
        command: str,
        *,
        add_to_context: bool,
    ) -> TerminalCommandResult:
        self.terminal_commands.append((command, add_to_context))
        return TerminalCommandResult(
            command=command,
            output="command output",
            exit_code=0,
            ok=True,
            added_to_context=add_to_context,
        )


class FakeVoiceController:
    def __init__(self, listener: Callable[[VoiceInputEvent], None]) -> None:
        self.listener = listener
        self.state = "idle"
        self.closed = False
        self.context: Callable[[], object] | None = None

    @property
    def active(self) -> bool:
        return self.state in {"connecting", "recording", "finalizing", "polishing"}

    async def start(self, context_provider: Callable[[], object]) -> None:
        self.state = "connecting"
        self.listener(VoiceInputEvent("connecting"))
        self.context = context_provider
        self.state = "recording"
        self.listener(VoiceInputEvent("recording"))

    async def stop(self) -> None:
        self.state = "finalizing"
        self.listener(VoiceInputEvent("finalizing"))
        assert self.context is not None
        self.context()
        self.state = "polishing"
        self.listener(VoiceInputEvent("polishing"))
        self.state = "completed"
        self.listener(
            VoiceInputEvent(
                "completed",
                text=" spoken ",
                raw_text="raw",
                breakdown=RequestContextBreakdown(
                    "Voice polish", (RequestContextPart("raw ASR", 2),)
                ),
            )
        )

    async def cancel(self) -> None:
        self.state = "cancelled"
        self.listener(VoiceInputEvent("cancelled", message="cancelled"))

    async def aclose(self) -> None:
        self.closed = True


def test_session_sidebar_and_compact_info_render_axis_session_facts(
    tmp_path: Path,
) -> None:
    session = CancellableSession(tmp_path)
    session.tools = (
        PromptTemplate(
            name="read",
            path=tmp_path / "read.md",
            content="Read files.",
            description="Read files",
        ),
    )
    session.skills = (
        Skill(
            name="review",
            path=tmp_path / "review" / "SKILL.md",
            content="Review code.",
            description="Review code",
        ),
    )
    session.prompt_templates = (
        PromptTemplate(
            name="explain",
            path=tmp_path / "explain.md",
            content="Explain code.",
            description="Explain code",
        ),
    )
    session.context_files = (
        ProjectContextFile(path=tmp_path / "AGENTS.md", content="Project rules."),
        ProjectContextFile(
            path=tmp_path / ".agents" / "AGENTS.md",
            content="Agent rules.",
        ),
    )
    console = Console(record=True, width=240)
    compact_session = CancellableSession(Path("/workspace/project"))

    console.print(render_session_sidebar(session))
    console.print(render_compact_session_info(compact_session))
    compact_session.auto_compact_token_threshold = 64_000
    console.print(render_compact_session_info(compact_session))

    output = console.export_text()
    assert "A X I S" in output
    assert "provider" in output
    assert "deepseek" in output
    assert "fake" in output
    assert "thinking" in output
    assert "high" in output
    assert "AGENTS.md" in output
    assert ".agents/AGENTS.md" in output
    assert "read" in output
    assert "review" in output
    assert "explain" in output
    assert "12k/128k context" in output
    assert "12k/64k context" in output
    assert "/workspace/project (--)" in output


def test_request_context_usage_renders_token_ratios() -> None:
    rendered = render_request_context_usage(
        ContextUsageEstimate(
            total_tokens=1_000,
            system_tokens=400,
            message_tokens=350,
            tool_tokens=250,
            message_count=3,
            tool_count=4,
        ),
        turn=2,
    )

    assert rendered.plain == (
        "request 2 estimate · total ≈1,000 tokens · "
        "system 40.0% (400) · messages 35.0% (350) · tools 25.0% (250)"
    )

    voice = render_request_context_usage(
        RequestContextBreakdown(
            "Voice polish",
            (RequestContextPart("raw ASR", 25), RequestContextPart("session", 75)),
        )
    )
    assert voice.plain == (
        "voice polish estimate · total ≈100 tokens · raw ASR 25.0% (25) · session 75.0% (75)"
    )


def test_tui_sidebar_responds_to_terminal_width_and_height(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = AxisTuiApp(CancellableSession(tmp_path))

        async with app.run_test(size=(120, 30)) as pilot:
            sidebar = app.query_one("#sidebar", SessionSidebar)
            compact = app.query_one("#compact-session-info", CompactSessionInfo)
            assert sidebar.display is True
            assert compact.display is True
            assert not app.has_class("-hide-sidebar")

            await pilot.resize_terminal(width=80, height=30)
            await pilot.pause()
            assert sidebar.display is False
            assert compact.display is True

            await pilot.resize_terminal(width=120, height=18)
            await pilot.pause()
            assert sidebar.display is False
            assert compact.display is True

            await pilot.resize_terminal(width=120, height=30)
            await pilot.pause()
            assert sidebar.display is True
            assert compact.display is True

    asyncio.run(scenario())


def test_tui_sidebar_fills_workspace_and_compact_info_wraps(tmp_path: Path) -> None:
    compact_console = Console(record=True, width=36)
    compact_console.print(render_compact_session_info(CancellableSession(tmp_path)))
    assert len(compact_console.export_text().splitlines()) > 1

    async def scenario() -> None:
        app = AxisTuiApp(CancellableSession(tmp_path))

        async with app.run_test(size=(120, 30)):
            workspace = app.query_one("#workspace")
            sidebar = app.query_one("#sidebar", SessionSidebar)
            transcript = app.query_one("#transcript", TranscriptView)
            prompt = app.query_one("#prompt", PromptInput)

            assert sidebar.region.height == workspace.region.height
            assert sidebar.outer_size.height == workspace.size.height
            assert transcript.styles.min_width is not None
            assert transcript.styles.min_width.value == 1
            assert prompt.soft_wrap is True

    asyncio.run(scenario())


def test_tui_prompt_grows_to_six_lines_then_scrolls(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = AxisTuiApp(CancellableSession(tmp_path))

        async with app.run_test(size=(120, 30)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            assert prompt.size.height == 1

            prompt.text = "x" * 500
            await pilot.pause()
            assert prompt.size.height == 6

            prompt.text = "x" * 1000
            await pilot.pause()
            assert prompt.size.height == 6
            assert prompt.max_scroll_y > 0

    asyncio.run(scenario())


def test_tui_clicking_transcript_refocuses_prompt(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = AxisTuiApp(CancellableSession(tmp_path))

        async with app.run_test(size=(120, 30)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            transcript = app.query_one("#transcript", TranscriptView)
            transcript.focus()
            await pilot.pause()
            assert app.screen.focused is transcript

            await pilot.click("#transcript")
            await pilot.pause()
            assert app.screen.focused is prompt

    asyncio.run(scenario())


def test_tui_runs_initial_positional_prompt_on_mount(tmp_path: Path) -> None:
    async def scenario() -> None:
        provider = FakeProvider(
            [[ProviderResponseEndEvent(message=AssistantMessage(content="Repository explained"))]]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "initial-prompt.jsonl"),
                cwd=tmp_path,
                tools=[],
            )
        )
        app = AxisTuiApp(session, initial_prompt="explain this repo")

        async with app.run_test(size=(120, 30)):
            await wait_until(lambda: len(provider.calls) == 1 and not session.is_running)

        assert provider.calls[0][2] == [UserMessage(content="explain this repo")]
        assert [(item.role, item.text) for item in app.state.items] == [
            ("user", "explain this repo"),
            ("assistant", "Repository explained"),
        ]

    asyncio.run(scenario())


def test_tui_footer_hints_follow_normal_completion_and_running_modes(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        app = AxisTuiApp(CancellableSession(tmp_path))

        async with app.run_test(size=(120, 30)) as pilot:
            assert app.query_one(Footer) is not None
            assert visible_footer_bindings(app) == {
                "Quit": "ctrl+d",
                "Clear": "ctrl+c",
                "Commands": "ctrl+k",
                "Submit": "enter",
                "Newline": "shift+enter",
                "Sessions": "ctrl+r",
                "Thinking": "shift+tab",
                "Model": "ctrl+p",
                "Cancel": "escape",
                "Voice": "f2",
            }

            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "/se"
            prompt.cursor_position = len(prompt.value)
            await pilot.pause()
            assert visible_footer_bindings(app) == {
                "Choose": "Up/Down",
                "Complete": "Tab/Enter",
                "Close": "escape",
                "Voice": "f2",
            }

            prompt.value = ""
            prompt.cursor_position = 0
            app._rebuild_completions(prompt)
            app.adapter.apply(AgentStartEvent())
            app._render_state()
            assert visible_footer_bindings(app) == {
                "Steer": "enter",
                "Follow-up": "alt+enter",
                "Cancel": "escape",
                "Thinking": "ctrl+t",
                "Tools": "ctrl+o",
                "Voice": "f2",
            }

    asyncio.run(scenario())


def test_tui_keeps_textual_footer_on_short_windows(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = AxisTuiApp(CancellableSession(tmp_path))

        async with app.run_test(size=(120, 18)):
            assert app.query_one(Footer).display is True
            assert app.query_one("#sidebar", SessionSidebar).display is False
            assert app.query_one("#compact-session-info", CompactSessionInfo).display is True

    asyncio.run(scenario())


def test_tui_app_uses_theme_variables_and_configured_keys(tmp_path: Path) -> None:
    settings = TuiSettings(
        keybindings=TuiKeybindings(
            cancel="f4",
            toggle_tool_results="f8",
            toggle_thinking="f9",
            quit="f10",
        ),
        theme="high-contrast",
    )
    app = AxisTuiApp(CancellableSession(tmp_path), tui_settings=settings)

    variables = app.get_theme_variable_defaults()
    assert variables["axis-screen-text"] == "#ffffff"
    assert variables["axis-accent"] == "#ffb454"
    assert variables["axis-markdown-inline-code"] == "#7fffd4"
    assert app._bindings.key_to_bindings["f4"][0].action == "cancel_run"
    assert app._bindings.key_to_bindings["f8"][0].action == "toggle_tool_results"
    assert app._bindings.key_to_bindings["f9"][0].action == "toggle_thinking"
    assert app._bindings.key_to_bindings["f10"][0].action == "exit_app"


def test_tui_app_maps_omni_palette_and_registers_picker_option(tmp_path: Path) -> None:
    app = AxisTuiApp(
        CancellableSession(tmp_path),
        tui_settings=TuiSettings(theme="omni"),
    )

    variables = app.get_theme_variable_defaults()
    assert variables["axis-screen-background"] == "#191622"
    assert variables["axis-sidebar-background"] == "#13111B"
    assert variables["axis-prompt-background"] == "#201B2D"
    assert variables["axis-accent"] == "#FF79C6"
    assert variables["axis-markdown-inline-code"] == "#67E480"
    assert "omni" in ThemePickerScreen("omni").theme_names


def test_tui_app_maps_terminal_native_theme_to_terminal_ansi_colors(tmp_path: Path) -> None:
    app = AxisTuiApp(
        CancellableSession(tmp_path),
        tui_settings=TuiSettings(theme="terminal-native"),
    )

    variables = app.get_theme_variable_defaults()
    assert app.theme == "ansi-dark"
    assert app.native_ansi_color is True
    assert variables["axis-screen-background"] == "ansi_default"
    assert variables["axis-screen-overlay-background"] == "transparent"
    assert variables["axis-screen-text"] == "ansi_default"
    assert variables["axis-prompt-background"] == "ansi_default"
    assert variables["axis-accent"] == "ansi_bright_yellow"
    assert ThemePickerScreen("terminal-native").theme_names[-1] == "terminal-native"


def test_terminal_native_theme_compiles_and_runs_headlessly(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = AxisTuiApp(
            CancellableSession(tmp_path),
            tui_settings=TuiSettings(theme="terminal-native"),
        )
        async with app.run_test(size=(100, 30)):
            assert app.theme == "ansi-dark"
            assert app.native_ansi_color is True
            assert app.query_one("#prompt", PromptInput).styles.background.ansi == -1
            assert app.screen.styles.background.ansi == -1
            rendered = app.screen._compositor.render_update(full=True).render_segments(app.console)
            assert "\x1b[49m" in rendered
            app.adapter.apply(AgentStartEvent())
            app._render_state()
            assert app.state.running is True

    asyncio.run(scenario())


def test_f2_voice_inserts_polished_draft_at_frozen_cursor(tmp_path: Path) -> None:
    async def scenario() -> None:
        controllers: list[FakeVoiceController] = []

        def factory(listener: Callable[[VoiceInputEvent], None]) -> FakeVoiceController:
            controller = FakeVoiceController(listener)
            controllers.append(controller)
            return controller

        session = CancellableSession(tmp_path)
        app = AxisTuiApp(
            session,
            paths=AxisPaths(home=tmp_path / ".axis"),
            voice_controller_factory=factory,  # type: ignore[arg-type]
        )
        async with app.run_test(size=(120, 30)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.text = "before after"
            prompt.cursor_position = 6

            await pilot.press("f2")
            await wait_until(lambda: bool(controllers) and controllers[0].state == "recording")
            assert prompt.disabled is True
            assert prompt.text == "before after"
            assert "Recording" in str(app.query_one("#voice-status", Static).render())

            controllers[0].listener(VoiceInputEvent("partial", text="temporary words"))
            await pilot.pause()
            assert prompt.text == "before after"

            await pilot.press("f2")
            await wait_until(lambda: prompt.disabled is False)
            assert prompt.text == "before spoken  after"
            assert session.messages == ()
            usage = app.query_one("#request-context-usage", Static)
            assert "voice polish estimate" in str(usage.render()).lower()
            assert controllers[0].closed is True

    asyncio.run(scenario())


def test_escape_cancels_voice_before_running_agent(tmp_path: Path) -> None:
    async def scenario() -> None:
        controllers: list[FakeVoiceController] = []

        def factory(listener: Callable[[VoiceInputEvent], None]) -> FakeVoiceController:
            controller = FakeVoiceController(listener)
            controllers.append(controller)
            return controller

        session = CancellableSession(tmp_path)
        session._running = True
        app = AxisTuiApp(session, voice_controller_factory=factory)  # type: ignore[arg-type]
        async with app.run_test(size=(100, 25)) as pilot:
            await pilot.press("f2")
            await wait_until(lambda: bool(controllers) and controllers[0].state == "recording")
            await pilot.press("escape")
            await wait_until(lambda: controllers[0].state == "cancelled")
            assert session.cancel_called is False
            assert app.query_one("#prompt", PromptInput).disabled is False

    asyncio.run(scenario())


def test_voice_setup_masks_tests_and_saves_separate_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_test(config: object, api_key: str, *, listener: object = None) -> str:
        del config, listener
        assert api_key == "volc-key"
        return "测试成功"

    monkeypatch.setattr(
        tui_app,
        "list_audio_input_devices",
        lambda: (AudioInputDevice(4, "Studio Mic", 1, True),),
    )
    monkeypatch.setattr(tui_app, "test_voice_configuration", fake_test)

    async def scenario() -> None:
        paths = AxisPaths(home=tmp_path / ".axis", agents_home=tmp_path / ".agents")
        app = AxisTuiApp(CancellableSession(tmp_path), paths=paths)
        async with app.run_test(size=(100, 32)) as pilot:
            app._open_voice_setup()
            await pilot.pause()
            assert isinstance(app.screen, VoiceSetupScreen)
            key = app.screen.query_one("#voice-api-key", Input)
            assert key.password is True
            key.value = "volc-key"
            app.screen.query_one("#voice-device-list", ListView).index = 1
            await pilot.pause()
            app.screen.query_one("#voice-test", Button).press()
            await wait_until(lambda: app.screen.query_one("#voice-save", Button).disabled is False)
            app.screen.query_one("#voice-save", Button).press()
            await pilot.pause()
            assert load_voice_config(paths).input_device == 4
            assert (
                FileCredentialStore(credentials_path(paths)).get(VOLCENGINE_ASR_CREDENTIAL_NAME)
                == "volc-key"
            )

    asyncio.run(scenario())


def test_tui_submits_and_renders_streamed_thinking_and_markdown(tmp_path: Path) -> None:
    async def scenario() -> None:
        provider = FakeProvider(
            [
                [
                    ProviderResponseStartEvent(model="fake"),
                    ProviderThinkingDeltaEvent(delta="Inspect first."),
                    ProviderTextDeltaEvent(delta="## Done\n\n- one"),
                    ProviderResponseEndEvent(message=AssistantMessage(content="## Done\n\n- one")),
                ]
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
                cwd=tmp_path,
                tools=[],
            )
        )
        app = AxisTuiApp(session)

        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "Hello Axis"
            await pilot.press("enter")
            await wait_until(lambda: len(provider.calls) == 1 and not prompt.disabled)

            assert [(item.role, item.text) for item in app.state.items] == [
                ("user", "Hello Axis"),
                ("thinking", "Inspect first."),
                ("assistant", "## Done\n\n- one"),
            ]
            transcript = app.query_one("#transcript", TranscriptView)
            rendered = "\n".join(line.text for line in transcript.lines)
            assert "Thinking… Press Ctrl+T" in rendered
            assert "Inspect first." not in rendered
            assert "## Done" in rendered
            assert prompt.has_focus is True

    asyncio.run(scenario())


def test_tui_surfaces_prompt_expansion_errors(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
                cwd=tmp_path,
                tools=[],
            )
        )
        app = AxisTuiApp(session)

        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "/skill:missing"
            await pilot.press("enter")
            await wait_until(lambda: app.state.error is not None and not prompt.disabled)

            assert app.state.error == "Unknown skill: missing"
            assert app.state.items[-1].role == "error"
            assert app.state.items[-1].text == "Error: Unknown skill: missing"

    asyncio.run(scenario())


def test_tui_completion_accepts_before_submitting(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = CancellableSession(tmp_path)
        session.skills = (
            Skill(
                name="review",
                path=tmp_path / "review" / "SKILL.md",
                content="Review code.",
                description="Review code",
            ),
        )
        session.prompt_templates = (
            PromptTemplate(
                name="explain",
                path=tmp_path / "explain.md",
                content="Explain code.",
                description="Explain code",
            ),
        )
        app = AxisTuiApp(session)

        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "/skill:r"
            prompt.cursor_position = len(prompt.value)
            await pilot.pause()

            assert [item.display for item in app.completion_state.items] == ["/skill:review"]
            await pilot.press("enter")
            await pilot.pause()
            assert prompt.value == "/skill:review"
            assert session.is_running is False

            prompt.value = "/expl"
            prompt.cursor_position = len(prompt.value)
            await pilot.pause()
            await pilot.press("tab")
            await pilot.pause()
            assert prompt.value == "/explain"

    asyncio.run(scenario())


def test_tui_prompt_is_multiline_and_file_completion_uses_cursor(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "README.md").write_text("# Axis\n", encoding="utf-8")
        app = AxisTuiApp(CancellableSession(tmp_path))

        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "Read @READ after"
            prompt.cursor_position = len("Read @READ")
            app._rebuild_completions(prompt)

            assert [item.display for item in app.completion_state.items] == ["@README.md"]
            await pilot.press("tab")
            await pilot.pause()
            assert prompt.value == "Read @README.md after"

            prompt.value = "first"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("shift+enter")
            await pilot.press("s", "e", "c", "o", "n", "d")
            await pilot.pause()
            assert prompt.value == "first\nsecond"

    asyncio.run(scenario())


def test_tui_tracks_complete_tool_round_trip(tmp_path: Path) -> None:
    async def scenario() -> None:
        async def execute(
            arguments: Mapping[str, JSONValue],
            signal: object | None = None,
        ) -> AgentToolResult:
            del signal
            return AgentToolResult(
                tool_call_id="provider-will-replace",
                name="read",
                ok=True,
                content=f"contents of {arguments['path']}",
            )

        call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
        provider = FakeProvider(
            [
                [ProviderResponseEndEvent(message=AssistantMessage(tool_calls=[call]))],
                [ProviderResponseEndEvent(message=AssistantMessage(content="Finished"))],
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "tool-session.jsonl"),
                cwd=tmp_path,
                tools=[AgentTool("read", "Read", {"type": "object"}, execute)],
            )
        )
        app = AxisTuiApp(session)

        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "Read README"
            await pilot.press("enter")
            await wait_until(lambda: len(provider.calls) == 2 and not prompt.disabled)

            tool_items = [item for item in app.state.items if item.role == "tool"]
            assert len(tool_items) == 1
            assert tool_items[0].text == "→ read README.md"
            assert tool_items[0].tool_call_id == "call-1"
            assert tool_items[0].tool_result_text == "✓ read\ncontents of README.md"
            assert app._request_context_turn == 2
            assert app._request_context_usage is not None
            assert app._request_context_usage.tool_count == 1
            request_usage = app.query_one("#request-context-usage", Static)
            request_text = str(request_usage.render())
            assert "request 2 estimate" in request_text
            assert "system " in request_text
            assert "messages " in request_text
            assert "tools " in request_text

    asyncio.run(scenario())


def test_tui_denies_protected_tool_before_any_side_effect(tmp_path: Path) -> None:
    async def scenario() -> None:
        executed: list[str] = []

        async def execute(
            arguments: Mapping[str, JSONValue],
            signal: object | None = None,
        ) -> AgentToolResult:
            del signal
            executed.append(str(arguments["command"]))
            return AgentToolResult(
                tool_call_id="ignored",
                name="bash",
                ok=True,
                content="unsafe",
            )

        call = ToolCall(
            id="call-approval",
            name="bash",
            arguments={"command": "touch should-not-exist"},
        )
        provider = FakeProvider(
            [
                [ProviderResponseEndEvent(message=AssistantMessage(tool_calls=[call]))],
                [ProviderResponseEndEvent(message=AssistantMessage(content="Understood"))],
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "approval-session.jsonl"),
                cwd=tmp_path,
                tools=[
                    AgentTool(
                        "bash",
                        "Run shell",
                        {"type": "object"},
                        execute,
                        requires_approval=True,
                    )
                ],
            )
        )
        app = AxisTuiApp(session)

        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "Run command"
            await pilot.press("enter")
            await wait_until(lambda: isinstance(app.screen, ToolApprovalScreen))

            assert executed == []
            details = str(app.screen.query_one("#tool-approval-details", Static).render())
            assert "touch should-not-exist" in details
            assert str(tmp_path) in details

            await pilot.press("d")
            await wait_until(lambda: len(provider.calls) == 2 and not session.is_running)

            assert executed == []
            tool_item = next(item for item in app.state.items if item.tool_call_id == call.id)
            assert tool_item.tool_result_text == "✗ bash\nTool call denied by user"

    asyncio.run(scenario())


def test_tui_loads_restored_messages_into_transcript(tmp_path: Path) -> None:
    call = ToolCall(id="call-1", name="edit", arguments={"path": "README.md"})
    session = CancellableSession(
        tmp_path,
        messages=(
            UserMessage(content="Read the file"),
            AssistantMessage(content="Inspecting", tool_calls=[call]),
            ToolResultMessage(
                tool_call_id="call-1",
                name="edit",
                content="Changed",
                data={"patch": "--- README.md\n+++ README.md"},
            ),
        ),
    )

    app = AxisTuiApp(session)

    assert [(item.role, item.text) for item in app.state.items] == [
        ("user", "Read the file"),
        ("assistant", "Inspecting"),
        ("tool", "→ edit README.md"),
    ]
    assert app.state.items[-1].tool_result_text is not None
    assert "Patch:" in app.state.items[-1].tool_result_text


def test_transcript_widget_extracts_plain_text_selection(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = AxisTuiApp(
            CancellableSession(tmp_path, messages=(UserMessage(content="alpha beta\ngamma"),))
        )
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            widget = app.query_one(TranscriptMessageWidget)
            assert widget.get_selection(Selection(Offset(6, 0), Offset(10, 0))) == (
                "beta",
                "\n",
            )

    asyncio.run(scenario())


def test_tui_auto_copy_selection_is_configurable(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = AxisTuiApp(
            CancellableSession(tmp_path, messages=(UserMessage(content="copy this"),)),
            tui_settings=TuiSettings(auto_copy_selection=True),
        )
        copied: list[str] = []
        app.copy_to_clipboard = copied.append  # type: ignore[method-assign]

        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            message = app.query_one(TranscriptMessageWidget)
            app.screen.selections = {message: SELECT_ALL}
            await app.on_text_selected()

        assert copied == ["copy this"]

    asyncio.run(scenario())


def test_tui_toggles_tool_results_and_thinking(tmp_path: Path) -> None:
    async def scenario() -> None:
        call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
        session = CancellableSession(
            tmp_path,
            messages=(
                AssistantMessage(
                    content="answer",
                    tool_calls=[call],
                    provider_data={"reasoning_content": "internal plan"},
                ),
                ToolResultMessage(
                    tool_call_id="call-1",
                    name="read",
                    content="README contents",
                ),
            ),
        )
        app = AxisTuiApp(session)
        app.state.add_thinking_delta("internal plan")

        async with app.run_test(size=(100, 30)) as pilot:
            transcript = app.query_one("#transcript", TranscriptView)

            def visible_text() -> str:
                return "\n".join(line.text for line in transcript.lines)

            assert "README contents" not in visible_text()
            assert "internal plan" not in visible_text()
            await pilot.press("ctrl+o")
            await pilot.press("ctrl+t")
            await pilot.pause()
            assert app.state.show_tool_results is True
            assert app.state.show_thinking is True
            assert "README contents" in visible_text()
            assert "internal plan" in visible_text()

    asyncio.run(scenario())


def test_streaming_widget_is_reused_for_multiple_deltas(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = AxisTuiApp(CancellableSession(tmp_path))
        async with app.run_test(size=(80, 24)) as pilot:
            transcript = app.query_one("#transcript", TranscriptView)
            await transcript.append_assistant_delta("alpha ")
            await transcript.append_assistant_delta("beta")
            await pilot.pause()
            widgets = list(app.query(StreamingTranscriptMessageWidget))
            assert len(widgets) == 1
            assert widgets[0].selection_text == "alpha beta"

    asyncio.run(scenario())


def test_escape_requests_cooperative_cancellation(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = CancellableSession(tmp_path)
        app = AxisTuiApp(session)

        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "Long task"
            await pilot.press("enter")
            await wait_until(lambda: session.is_running)
            await pilot.press("escape")
            await wait_until(lambda: app.state.cancelled and not prompt.disabled)

            assert session.cancel_called is True
            assert app.state.error is None
            assert app.state.running is False
            assert any(item.text == "Agent run cancelled." for item in app.state.items)

    asyncio.run(scenario())


def test_ctrl_d_exits_the_app(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = AxisTuiApp(CancellableSession(tmp_path))
        async with app.run_test() as pilot:
            assert app.is_running is True
            await pilot.press("ctrl+d")
            await wait_until(lambda: not app.is_running)

    asyncio.run(scenario())


def test_terminal_command_prefix_detects_shell_mode() -> None:
    assert tui_app._terminal_command_prefix_span("! pwd") == (0, 1)
    assert tui_app._terminal_command_prefix_span("!! pwd") == (0, 2)
    assert tui_app._terminal_command_prefix_span("  !! pwd") == (2, 4)
    assert tui_app._terminal_command_prefix_span("hello ! pwd") is None


def test_tui_shell_mode_highlights_prefix_and_enables_path_completion(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        (tmp_path / "README.md").write_text("# Axis\n", encoding="utf-8")
        app = AxisTuiApp(CancellableSession(tmp_path))

        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "!! cat READ"
            prompt.cursor_position = len(prompt.value)
            await pilot.pause()

            assert prompt.has_class("-shell-mode")
            assert prompt.get_line(0).spans[-1].start == 0
            assert prompt.get_line(0).spans[-1].end == 2
            assert [item.display for item in app.completion_state.items] == ["README.md"]

            await pilot.press("tab")
            assert prompt.value == "!! cat README.md"

    asyncio.run(scenario())


def test_tui_queues_steering_and_follow_up_while_running(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = CancellableSession(tmp_path)
        app = AxisTuiApp(session)

        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "long task"
            await pilot.press("enter")
            await wait_until(lambda: session.is_running)

            prompt.value = "adjust course\nwith detail"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await pilot.pause()
            assert session.queued_steering_messages == ("adjust course\nwith detail",)
            assert app.state.queued_steering == session.queued_steering_messages

            prompt.value = "after this\nrun tests"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("alt+enter")
            await pilot.pause()
            assert session.queued_follow_up_messages == ("after this\nrun tests",)
            assert app.state.queued_follow_up == session.queued_follow_up_messages

            queue = app.query_one("#queued-messages", Static)
            assert queue.display is True
            rows = [
                str(row)
                for row in tui_app._render_queued_messages(
                    app.state,
                    theme=app.tui_settings.resolved_theme,
                ).renderables
            ]
            assert "↪ steering · queued: adjust course" in rows
            assert "↳ follow-up · queued: after this" in rows
            assert all("with detail" not in row and "run tests" not in row for row in rows)

            await pilot.press("escape")
            await wait_until(lambda: not session.is_running)

    asyncio.run(scenario())


def test_up_arrow_edits_latest_queued_follow_up(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = CancellableSession(tmp_path)
        session.queued_follow_up_messages = ("first", "latest\nwith detail")
        app = AxisTuiApp(session)

        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = ""
            app._render_state()
            await pilot.press("up")
            await pilot.pause()

            assert prompt.value == "latest\nwith detail"
            assert session.queued_follow_up_messages == ("first",)
            assert app.state.queued_follow_up == ("first",)

    asyncio.run(scenario())


def test_tui_runs_terminal_commands_with_context_modes(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = CancellableSession(tmp_path)
        app = AxisTuiApp(session)

        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "! pwd"
            await pilot.press("enter")
            await wait_until(lambda: len(session.terminal_commands) == 1 and not prompt.disabled)
            assert session.terminal_commands == [("pwd", True)]
            assert app.state.items[-1].text == "$ pwd"
            assert app.state.items[-1].tool_result_text == (
                "✓ bash · added to context\ncommand output"
            )
            assert app.state.items[-1].always_show_tool_result is True

            prompt.value = "!! pwd"
            await pilot.press("enter")
            await wait_until(lambda: len(session.terminal_commands) == 2 and not prompt.disabled)
            assert session.terminal_commands[-1] == ("pwd", False)
            assert app.state.items[-1].tool_result_text == (
                "✓ bash · not added to context\ncommand output"
            )

    asyncio.run(scenario())


def test_tui_marks_terminal_exception_as_failed(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = CancellableSession(tmp_path)

        async def fail(
            command: str,
            *,
            add_to_context: bool,
        ) -> TerminalCommandResult:
            del command, add_to_context
            raise RuntimeError("boom")

        session.run_terminal_command = fail  # type: ignore[method-assign]
        app = AxisTuiApp(session)

        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "!! false"
            await pilot.press("enter")
            await wait_until(lambda: not prompt.disabled)

            assert app.state.items[-1].text == "$ false"
            assert app.state.items[-1].tool_result_text == ("✗ bash · not added to context\nboom")

    asyncio.run(scenario())


def test_tui_activity_indicator_tracks_running_state(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = AxisTuiApp(CancellableSession(tmp_path))
        async with app.run_test() as pilot:
            prefix = app.query_one("#prompt-prefix", Static)
            assert prefix.render().plain == "A"

            app.state.running = True
            app._render_state()
            assert "■" in prefix.render().plain
            initial = prefix.render().plain
            app._tick_activity()
            assert prefix.render().plain != initial

            app.state.running = False
            app._render_state()
            assert prefix.render().plain == "A"
            await pilot.pause()

    asyncio.run(scenario())


def test_ctrl_k_opens_registry_backed_command_completion(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = AxisTuiApp(CancellableSession(tmp_path))
        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            await pilot.press("ctrl+k")
            await pilot.pause()

            assert prompt.value == "/"
            assert [item.display for item in app.completion_state.items] == [
                "/compact",
                "/export",
                "/hotkeys",
                "/login",
                "/logout",
                "/model",
                "/name",
                "/new",
                "/quit",
                "/reload",
                "/resume",
                "/scoped-models",
                "/session",
                "/skill:",
                "/theme",
                "/thinking",
                "/tree",
                "/voice",
            ]

            prompt.value = "/sta"
            prompt.cursor_position = len(prompt.value)
            await pilot.pause()
            assert [item.display for item in app.completion_state.items] == ["/session"]

    asyncio.run(scenario())


def test_quit_command_exits_the_app(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = AxisTuiApp(CancellableSession(tmp_path))
        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "/quit"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await wait_until(lambda: not app.is_running)

    asyncio.run(scenario())


def test_exact_command_submits_and_uses_output_modal(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "commands-session.jsonl"),
                cwd=tmp_path,
                tools=[],
            )
        )
        app = AxisTuiApp(session)

        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "/session"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, CommandOutputScreen)
            assert "Model: deepseek:fake" in app.screen.message
            assert f"CWD: {tmp_path.resolve()}" in app.screen.message
            assert app.state.items == []

            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, CommandOutputScreen)

            prompt.value = "/help"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, CommandOutputScreen)
            assert app.screen.message == "Unknown command: /help"

    asyncio.run(scenario())


def test_reload_updates_resources_and_appends_inline_status(tmp_path: Path) -> None:
    async def scenario() -> None:
        axis_home = tmp_path / "axis-home"
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "reload-ui-session.jsonl"),
                cwd=tmp_path,
                tools=[],
                resource_paths=AxisResourcePaths(
                    paths=AxisPaths(
                        home=axis_home,
                        agents_home=tmp_path / "agents-home",
                    )
                ),
            )
        )
        skills_dir = axis_home / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "review.md").write_text(
            "---\ndescription: Review changes\n---\nReview carefully.",
            encoding="utf-8",
        )
        app = AxisTuiApp(session)

        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "/reload"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await pilot.pause()

            assert not isinstance(app.screen, CommandOutputScreen)
            assert [skill.name for skill in session.skills] == ["review"]
            assert [skill.name for skill in app.state.skills] == ["review"]
            assert app.state.items[-1].role == "status"
            assert app.state.items[-1].text.startswith(
                "/reload\nReloaded local coding resources and project context."
            )

    asyncio.run(scenario())


def test_theme_command_argument_persists_and_picker_selects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    async def scenario() -> None:
        app = AxisTuiApp(CancellableSession(tmp_path))
        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "/theme axis-light"
            prompt.cursor_position = len(prompt.value)
            await pilot.pause()
            assert [item.display for item in app.completion_state.items] == ["axis-light"]
            await pilot.press("enter")
            await pilot.pause()

            assert app.tui_settings.theme == "axis-light"
            assert app.get_theme_variable_defaults()["axis-screen-background"] == "#ffffff"
            assert (tmp_path / ".axis" / "tui.json").exists()

            prompt.value = "/theme"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ThemePickerScreen)
            picker = app.screen.query_one("#theme-picker-list", ListView)
            assert picker.index == 1

            await pilot.press("down", "enter")
            await pilot.pause()
            assert app.tui_settings.theme == "high-contrast"
            assert app.theme == "textual-dark"
            assert app.native_ansi_color is False

            app._set_tui_theme("terminal-native")
            assert app.theme == "ansi-dark"
            assert app.native_ansi_color is True

            app._set_tui_theme("axis-dark")
            assert app.theme == "textual-dark"
            assert app.native_ansi_color is False

    asyncio.run(scenario())


def test_ctrl_c_clears_unselected_prompt_text(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = AxisTuiApp(CancellableSession(tmp_path))
        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "discard me"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert prompt.value == ""

    asyncio.run(scenario())


def test_tui_resume_name_export_and_new_session_lifecycle(tmp_path: Path) -> None:
    async def scenario() -> None:
        paths = AxisPaths(home=tmp_path / ".axis", agents_home=tmp_path / ".agents")
        manager = SessionManager(paths)
        first = manager.create_session(cwd=tmp_path, model="fake", session_id="first")
        second = manager.create_session(cwd=tmp_path, model="fake", session_id="second")
        first_storage = JsonlSessionStorage(first.path)
        second_storage = JsonlSessionStorage(second.path)
        await first_storage.append(
            MessageEntry(id="first-root", message=UserMessage(content="First"))
        )
        await first_storage.append(LeafEntry(entry_id="first-root"))
        await second_storage.append(
            MessageEntry(id="second-root", message=UserMessage(content="Second"))
        )
        await second_storage.append(LeafEntry(entry_id="second-root"))
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="fake",
                storage=first_storage,
                cwd=tmp_path,
                tools=[],
                resource_paths=AxisResourcePaths(paths=paths),
                session_id="first",
                session_manager=manager,
            )
        )
        app = AxisTuiApp(session)
        notifications: list[str] = []
        app._notify = lambda message, **_kwargs: notifications.append(message)  # type: ignore[method-assign]

        async with app.run_test(size=(110, 32)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "/resume sec"
            prompt.cursor_position = len(prompt.value)
            await pilot.pause()
            assert [item.display for item in app.completion_state.items] == ["second"]
            prompt.value = "/resume second"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await wait_until(lambda: session.session_id == "second")
            assert [(item.role, item.text) for item in app.state.items] == [("user", "Second")]

            prompt.value = "/name Resumed work"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await pilot.pause()
            assert manager.get_session("second").title == "Resumed work"  # type: ignore[union-attr]

            prompt.value = "/export --format jsonl exported.jsonl"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await pilot.pause()
            assert (tmp_path / "exported.jsonl").exists()

            prompt.value = "/new"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await wait_until(lambda: session.session_id not in {"first", "second"})
            assert app.state.items == []

        assert any(message.startswith("Resumed session") for message in notifications)
        assert "Session renamed: Resumed work" in notifications
        assert any(message.startswith("Exported session") for message in notifications)

    asyncio.run(scenario())


def test_ctrl_r_session_picker_resumes_selected_record(tmp_path: Path) -> None:
    async def scenario() -> None:
        paths = AxisPaths(home=tmp_path / ".axis", agents_home=tmp_path / ".agents")
        manager = SessionManager(paths)
        first = manager.create_session(cwd=tmp_path, model="fake", session_id="first")
        second = manager.create_session(cwd=tmp_path, model="fake", session_id="second")
        for record, content in ((first, "First"), (second, "Second")):
            storage = JsonlSessionStorage(record.path)
            await storage.append(
                MessageEntry(id=f"{record.id}-root", message=UserMessage(content=content))
            )
            await storage.append(LeafEntry(entry_id=f"{record.id}-root"))
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="fake",
                storage=JsonlSessionStorage(first.path),
                cwd=tmp_path,
                tools=[],
                resource_paths=AxisResourcePaths(paths=paths),
                session_id="first",
                session_manager=manager,
            )
        )
        app = AxisTuiApp(session)

        async with app.run_test(size=(110, 32)) as pilot:
            await pilot.press("ctrl+r")
            await pilot.pause()
            assert isinstance(app.screen, SessionPickerScreen)
            selected_id = app.screen.records[0].id
            await pilot.press("enter")
            await wait_until(lambda: session.session_id == selected_id)

    asyncio.run(scenario())


def test_model_picker_and_keyboard_cycles_change_runtime_state(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = CancellableSession(tmp_path)
        app = AxisTuiApp(session)

        async with app.run_test(size=(110, 32)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "/model"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ModelPickerScreen)
            app.screen.query_one("#model-picker-search", Input).value = "fast"
            await pilot.pause()
            assert app.screen.visible_choices == (ModelChoice("deepseek", "fake-fast"),)
            await pilot.press("enter")
            await wait_until(lambda: session.model == "fake-fast")
            assert session.model_changes[-1] == ModelChoice("deepseek", "fake-fast")

            await pilot.press("shift+tab")
            await wait_until(lambda: session.thinking_level == "xhigh")
            await pilot.press("ctrl+p")
            await wait_until(lambda: session.model == "fake")

            prompt.value = "/scoped-models"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ModelPickerScreen)
            assert app.screen.picker_kind == "scoped"

    asyncio.run(scenario())


def test_login_picker_saves_and_logout_removes_private_key(tmp_path: Path) -> None:
    async def scenario() -> None:
        paths = AxisPaths(home=tmp_path / ".axis", agents_home=tmp_path / ".agents")
        session = CancellableSession(tmp_path)
        session.resource_paths = AxisResourcePaths(paths=paths, cwd=tmp_path)
        app = AxisTuiApp(session)
        notifications: list[str] = []
        app._notify = lambda message, **_kwargs: notifications.append(message)  # type: ignore[method-assign]

        async with app.run_test(size=(110, 32)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "/login"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, LoginProviderPickerScreen)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, LoginScreen)
            app.screen.query_one("#login-api-key", Input).value = "super-secret"
            await pilot.press("enter")
            await pilot.pause()

            store = FileCredentialStore(paths.home / "credentials.json")
            assert store.get("deepseek") == "super-secret"
            assert all("super-secret" not in message for message in notifications)

            prompt.value = "/logout deepseek"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await pilot.pause()
            assert store.get("deepseek") is None
            assert any(message.startswith("Removed stored API key") for message in notifications)

    asyncio.run(scenario())


def test_login_preserves_in_memory_custom_models_and_scope(tmp_path: Path) -> None:
    async def scenario() -> None:
        paths = AxisPaths(home=tmp_path / ".axis", agents_home=tmp_path / ".agents")
        default_settings = load_provider_settings(paths)
        default_provider = default_settings.get_provider("deepseek")
        custom_model = "deepseek-v4-custom"
        provider = replace(
            default_provider,
            models=(*default_provider.models, custom_model),
            thinking_models=(*default_provider.thinking_models, custom_model),
        )
        settings = ProviderSettings(
            default_provider="deepseek",
            providers=(provider,),
            scoped_models=(ScopedModelConfig("deepseek", custom_model),),
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model=custom_model,
                storage=JsonlSessionStorage(tmp_path / "login-preserve.jsonl"),
                cwd=tmp_path,
                tools=[],
                resource_paths=AxisResourcePaths(paths=paths),
                provider_name="deepseek",
                provider_settings=settings,
            )
        )
        app = AxisTuiApp(session)
        entry = builtin_provider_entry("deepseek")
        assert entry is not None

        app._handle_login_result(entry, "stored-key")

        restored = load_provider_settings(paths)
        assert custom_model in restored.get_provider("deepseek").models
        assert restored.scoped_models == (ScopedModelConfig("deepseek", custom_model),)
        await session.aclose()

    asyncio.run(scenario())


def test_tree_picker_prefills_user_branch_and_can_summarize(tmp_path: Path) -> None:
    async def scenario() -> None:
        storage = JsonlSessionStorage(tmp_path / "tree-ui.jsonl")
        entries = (
            MessageEntry(id="root", message=UserMessage(content="Root")),
            MessageEntry(
                id="answer",
                parent_id="root",
                message=AssistantMessage(content="Answer"),
            ),
            MessageEntry(
                id="followup",
                parent_id="answer",
                message=UserMessage(content="Try again"),
            ),
            LeafEntry(entry_id="followup"),
        )
        for entry in entries:
            await storage.append(entry)
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="fake",
                storage=storage,
                cwd=tmp_path,
                tools=[],
            )
        )
        app = AxisTuiApp(session)

        async with app.run_test(size=(110, 32)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "/tree"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, TreePickerScreen)
            await pilot.press("enter")
            await wait_until(lambda: prompt.value == "Try again")
            assert session.messages == (
                UserMessage(content="Root"),
                AssistantMessage(content="Answer"),
            )

    asyncio.run(scenario())


def test_tui_compaction_reloads_semantic_summary(tmp_path: Path) -> None:
    async def scenario() -> None:
        storage = JsonlSessionStorage(tmp_path / "compact-ui.jsonl")
        root = MessageEntry(id="root", message=UserMessage(content="Earlier request"))
        answer = MessageEntry(
            id="answer",
            parent_id="root",
            message=AssistantMessage(content="Earlier answer"),
        )
        for entry in (root, answer, LeafEntry(entry_id="answer")):
            await storage.append(entry)
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider(
                    [
                        [
                            ProviderResponseEndEvent(
                                message=AssistantMessage(content="Generated summary")
                            )
                        ]
                    ]
                ),
                model="fake",
                storage=storage,
                cwd=tmp_path,
                tools=[],
            )
        )
        app = AxisTuiApp(session)

        async with app.run_test(size=(110, 32)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "/compact Keep decisions"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await wait_until(lambda: bool(session.state.compaction_entries))
            await pilot.pause()

            assert [(item.role, item.text) for item in app.state.items] == [
                ("compaction_summary", "Compaction summary (Ctrl+O to expand)")
            ]
            assert app.state.items[0].tool_result_text == "Generated summary"

    asyncio.run(scenario())


def test_tree_picker_s_summarizes_abandoned_branch(tmp_path: Path) -> None:
    async def scenario() -> None:
        storage = JsonlSessionStorage(tmp_path / "tree-summary-ui.jsonl")
        entries = (
            MessageEntry(id="root", message=UserMessage(content="Root")),
            MessageEntry(
                id="answer",
                parent_id="root",
                message=AssistantMessage(content="Answer"),
            ),
            MessageEntry(
                id="followup",
                parent_id="answer",
                message=UserMessage(content="Abandoned work"),
            ),
            LeafEntry(entry_id="followup"),
        )
        for entry in entries:
            await storage.append(entry)
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider(
                    [[ProviderResponseEndEvent(message=AssistantMessage(content="Branch summary"))]]
                ),
                model="fake",
                storage=storage,
                cwd=tmp_path,
                tools=[],
            )
        )
        app = AxisTuiApp(session)

        async with app.run_test(size=(110, 32)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "/tree"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, TreePickerScreen)
            await pilot.press("up", "s")
            await wait_until(
                lambda: bool(app.state.items) and app.state.items[0].role == "branch_summary"
            )

            assert app.state.items[0].tool_result_text == "Branch summary"
            assert prompt.disabled is False

    asyncio.run(scenario())


def test_tree_picker_c_uses_custom_summary_instructions(tmp_path: Path) -> None:
    async def scenario() -> None:
        storage = JsonlSessionStorage(tmp_path / "tree-custom-summary-ui.jsonl")
        for entry in (
            MessageEntry(id="root", message=UserMessage(content="Root")),
            MessageEntry(
                id="answer",
                parent_id="root",
                message=AssistantMessage(content="Answer"),
            ),
            MessageEntry(
                id="followup",
                parent_id="answer",
                message=UserMessage(content="Abandoned work"),
            ),
            LeafEntry(entry_id="followup"),
        ):
            await storage.append(entry)
        provider = FakeProvider(
            [[ProviderResponseEndEvent(message=AssistantMessage(content="Focused summary"))]]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=storage,
                cwd=tmp_path,
                tools=[],
            )
        )
        app = AxisTuiApp(session)

        async with app.run_test(size=(110, 32)) as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "/tree"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("up", "c")
            await pilot.pause()
            assert isinstance(app.screen, BranchSummaryInstructionsScreen)
            app.screen.query_one(
                "#branch-summary-instructions-input", TextArea
            ).text = "Focus on failing commands."
            await pilot.press("ctrl+enter")
            await wait_until(
                lambda: bool(app.state.items) and app.state.items[0].role == "branch_summary"
            )

            assert app.state.items[0].tool_result_text == "Focused summary"
            assert "Additional instructions:\nFocus on failing commands." in (
                provider.calls[0][2][0].content
            )

    asyncio.run(scenario())


def test_escape_cancels_active_compaction_and_restores_transcript(tmp_path: Path) -> None:
    async def scenario() -> None:
        started = asyncio.Event()

        class SlowCompactSession(CancellableSession):
            async def compact(self, instructions: str | None = None) -> str:
                del instructions
                started.set()
                await asyncio.Event().wait()
                return "unreachable"

        session = SlowCompactSession(
            tmp_path,
            messages=(UserMessage(content="Earlier"),),
        )
        app = AxisTuiApp(session)
        notifications: list[str] = []
        app._notify = lambda message, **_kwargs: notifications.append(message)  # type: ignore[method-assign]

        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "/compact"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await asyncio.wait_for(started.wait(), timeout=1)
            await pilot.press("escape")
            await pilot.pause()

            assert app._compaction_worker is None
            assert [(item.role, item.text) for item in app.state.items] == [("user", "Earlier")]
            assert notifications == ["Cancelled compaction."]

    asyncio.run(scenario())


def test_tree_picker_toggles_tool_call_entries(tmp_path: Path) -> None:
    async def scenario() -> None:
        storage = JsonlSessionStorage(tmp_path / "tree-tools-ui.jsonl")
        call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
        entries = (
            MessageEntry(id="root", message=UserMessage(content="Read")),
            MessageEntry(
                id="tool-call",
                parent_id="root",
                message=AssistantMessage(tool_calls=[call]),
            ),
            MessageEntry(
                id="tool-result",
                parent_id="tool-call",
                message=ToolResultMessage(
                    tool_call_id="call-1",
                    name="read",
                    content="contents",
                ),
            ),
            MessageEntry(
                id="final",
                parent_id="tool-result",
                message=AssistantMessage(content="Done"),
            ),
            LeafEntry(entry_id="final"),
        )
        for entry in entries:
            await storage.append(entry)
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="fake",
                storage=storage,
                cwd=tmp_path,
                tools=[],
            )
        )
        app = AxisTuiApp(session)

        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", PromptInput)
            prompt.value = "/tree"
            prompt.cursor_position = len(prompt.value)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, TreePickerScreen)
            tree = app.screen.query_one("#tree-picker-list", ListView)
            assert len(tree.children) == 3

            await pilot.press("ctrl+t")
            await pilot.pause()
            assert len(tree.children) == 2
            assert "tool calls hidden" in str(
                app.screen.query_one("#tree-picker-help", Static).render()
            )

    asyncio.run(scenario())
