"""Tests for Axis's non-interactive command-line entry point."""

import asyncio
import json
from io import StringIO
from pathlib import Path

import pytest
from typer.testing import CliRunner

from axis_agent import (
    AssistantMessage,
    ErrorEvent,
    JsonlSessionStorage,
    MessageEndEvent,
    MessageEntry,
    SessionState,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from axis_ai import (
    FakeProvider,
    ProviderErrorEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
)
from axis_coding import (
    AxisPaths,
    CodingSession,
    FileCredentialStore,
    OpenAICompatibleProviderConfig,
    ProviderSettings,
    ResourceError,
    ScopedModelConfig,
    SessionManager,
    ToolApprovalPolicy,
    __version__,
)
from axis_coding.cli import (
    _resolve_tui_startup_selection,
    app,
    run_deepseek_print_mode,
    run_deepseek_tui_mode,
    run_print_mode,
)
from axis_coding.rendering import FinalTextRenderer, PrintOutputMode

runner = CliRunner()


def test_final_text_renderer_prints_only_last_complete_assistant_message() -> None:
    stdout = StringIO()
    stderr = StringIO()
    renderer = FinalTextRenderer(stdout=stdout, stderr=stderr)

    renderer.render(MessageEndEvent(message=UserMessage(content="ignored prompt")))
    renderer.render(MessageEndEvent(message=AssistantMessage(content="first")))
    renderer.render(MessageEndEvent(message=AssistantMessage(content="final")))

    assert stdout.getvalue() == ""
    assert renderer.finish() is True
    assert stdout.getvalue() == "final\n"
    assert stderr.getvalue() == ""


def test_final_text_renderer_reports_non_recoverable_errors() -> None:
    stdout = StringIO()
    stderr = StringIO()
    renderer = FinalTextRenderer(stdout=stdout, stderr=stderr)

    renderer.render(ErrorEvent(message="temporary", recoverable=True))
    renderer.render(ErrorEvent(message="provider failed", recoverable=False))

    assert renderer.finish() is False
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "Error: temporary\nError: provider failed\n"


def test_run_print_mode_uses_harness_and_renders_final_text(tmp_path: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()
    provider = FakeProvider([[ProviderResponseEndEvent(message=AssistantMessage(content="Done"))]])

    succeeded = asyncio.run(
        run_print_mode(
            prompt="Inspect this project",
            model="fake-model",
            cwd=tmp_path,
            provider=provider,
            stdout=stdout,
            stderr=stderr,
        )
    )

    assert succeeded is True
    assert stdout.getvalue() == "Done\n"
    assert stderr.getvalue() == ""
    assert provider.calls[0][0] == "fake-model"
    assert provider.calls[0][2] == [UserMessage(content="Inspect this project")]
    assert [tool.name for tool in provider.calls[0][3]] == ["read", "write", "edit", "bash"]
    assert str(tmp_path) in provider.calls[0][1]


def test_run_print_mode_returns_failure_for_provider_error(tmp_path: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()
    provider = FakeProvider([[ProviderErrorEvent(message="provider failed")]])

    succeeded = asyncio.run(
        run_print_mode(
            prompt="Hello",
            model="fake-model",
            cwd=tmp_path,
            provider=provider,
            stdout=stdout,
            stderr=stderr,
        )
    )

    assert succeeded is False
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "Error: provider failed\n"


def test_print_mode_ask_policy_denies_protected_tool_without_tty(tmp_path: Path) -> None:
    target = tmp_path / "blocked.txt"
    call = ToolCall(
        id="call-write",
        name="write",
        arguments={"path": str(target), "content": "blocked"},
    )
    provider = FakeProvider(
        [
            [ProviderResponseEndEvent(message=AssistantMessage(tool_calls=[call]))],
            [ProviderResponseEndEvent(message=AssistantMessage(content="Denied safely"))],
        ]
    )
    stdout = StringIO()
    stderr = StringIO()

    succeeded = asyncio.run(
        run_print_mode(
            prompt="Write a file",
            model="fake-model",
            cwd=tmp_path,
            provider=provider,
            output=PrintOutputMode.JSON,
            stdin=StringIO("y\n"),
            stdout=stdout,
            stderr=stderr,
        )
    )

    events = [json.loads(line) for line in stdout.getvalue().splitlines()]
    result = next(
        message for message in provider.calls[1][2] if isinstance(message, ToolResultMessage)
    )
    assert succeeded is True
    assert target.exists() is False
    assert result.tool_call_id == "call-write"
    assert result.ok is False
    assert result.error == "Tool call denied by user"
    assert [event["type"] for event in events].count("tool_approval_request") == 1
    assert [event["type"] for event in events].count("tool_approval_resolved") == 1
    assert stderr.getvalue() == ""


def test_print_mode_allow_policy_executes_protected_tool(tmp_path: Path) -> None:
    target = tmp_path / "allowed.txt"
    call = ToolCall(
        id="call-write",
        name="write",
        arguments={"path": str(target), "content": "allowed"},
    )
    provider = FakeProvider(
        [
            [ProviderResponseEndEvent(message=AssistantMessage(tool_calls=[call]))],
            [ProviderResponseEndEvent(message=AssistantMessage(content="Written"))],
        ]
    )

    succeeded = asyncio.run(
        run_print_mode(
            prompt="Write a file",
            model="fake-model",
            cwd=tmp_path,
            provider=provider,
            tool_policy=ToolApprovalPolicy.ALLOW,
            stdout=StringIO(),
            stderr=StringIO(),
        )
    )

    assert succeeded is True
    assert target.read_text(encoding="utf-8") == "allowed"


def test_cli_shows_version_without_provider_configuration() -> None:
    result = runner.invoke(app, ["--version"], env={})

    assert result.exit_code == 0
    assert result.stdout == f"axis {__version__}\n"


def test_cli_help_is_available_without_provider_configuration() -> None:
    result = runner.invoke(app, ["--help"], env={})

    assert result.exit_code == 0
    assert "Axis personal coding agent" in result.stdout
    assert "--prompt" in result.stdout


def test_cli_without_prompt_launches_tui(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[tuple[str, Path]] = []

    async def fake_run_deepseek_tui_mode(
        *,
        model: str | None,
        cwd: Path,
        **_kwargs: object,
    ) -> None:
        observed.append((model, cwd))

    monkeypatch.setattr("axis_coding.cli.run_deepseek_tui_mode", fake_run_deepseek_tui_mode)

    result = runner.invoke(
        app,
        ["--cwd", str(tmp_path), "--model", "deepseek-v4-flash"],
        env={},
    )

    assert result.exit_code == 0
    assert observed == [("deepseek-v4-flash", tmp_path.resolve())]


def test_cli_positional_prompt_is_submitted_immediately_in_tui(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[dict[str, object]] = []

    async def fake_run_deepseek_tui_mode(**kwargs: object) -> None:
        observed.append(kwargs)

    monkeypatch.setattr("axis_coding.cli.run_deepseek_tui_mode", fake_run_deepseek_tui_mode)

    result = runner.invoke(
        app,
        [
            "--cwd",
            str(tmp_path),
            "--provider",
            "deepseek",
            "--model",
            "deepseek-v4-flash",
            "--new-session",
            "--auto-compact-threshold",
            "64000",
            "explain",
            "this",
            "repo",
        ],
        env={},
    )

    assert result.exit_code == 0
    assert observed == [
        {
            "model": "deepseek-v4-flash",
            "cwd": tmp_path.resolve(),
            "session_id": None,
            "new_session": True,
            "provider_name": "deepseek",
            "auto_compact_token_threshold": 64_000,
            "compact_retain_tokens": 20_000,
            "initial_prompt": "explain this repo",
        }
    ]


def test_cli_rejects_resume_with_new_session(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "--cwd",
            str(tmp_path),
            "--resume",
            "session-1",
            "--new-session",
        ],
        env={},
    )

    assert result.exit_code == 1
    assert "--resume and --new-session cannot be used together" in result.stderr


def test_cli_rejects_non_positive_compaction_retention(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["--cwd", str(tmp_path), "--compact-retain-tokens", "0"],
        env={},
    )

    assert result.exit_code == 1
    assert "--compact-retain-tokens must be greater than 0" in result.stderr


def test_tui_startup_falls_back_to_first_credentialed_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("AXIS_MISSING_DEFAULT_KEY", raising=False)
    monkeypatch.delenv("AXIS_LOCAL_KEY", raising=False)
    settings = ProviderSettings(
        default_provider="default",
        providers=(
            OpenAICompatibleProviderConfig(
                name="default",
                base_url="https://default.invalid/v1",
                api_key_env="AXIS_MISSING_DEFAULT_KEY",
                credential_name="default",
                models=("default-model",),
                default_model="default-model",
            ),
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="https://local.invalid/v1",
                api_key_env="AXIS_LOCAL_KEY",
                credential_name="local",
                models=("local-model",),
                default_model="local-model",
            ),
        ),
    )
    credential_store = FileCredentialStore(tmp_path / "credentials.json")
    credential_store.set("local", "stored-key")

    selection, resolved_settings = _resolve_tui_startup_selection(
        settings,
        record=None,
        provider_name=None,
        model=None,
        credential_store=credential_store,
    )

    assert selection.provider.name == "local"
    assert selection.model == "local-model"
    assert resolved_settings == settings


def test_tui_resume_without_provider_prefers_credentialed_scoped_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("AXIS_FIRST_KEY", raising=False)
    monkeypatch.delenv("AXIS_SCOPED_KEY", raising=False)
    settings = ProviderSettings(
        default_provider="first",
        providers=(
            OpenAICompatibleProviderConfig(
                name="first",
                base_url="https://first.invalid/v1",
                api_key_env="AXIS_FIRST_KEY",
                credential_name="first",
                models=("shared-model",),
                default_model="shared-model",
            ),
            OpenAICompatibleProviderConfig(
                name="scoped",
                base_url="https://scoped.invalid/v1",
                api_key_env="AXIS_SCOPED_KEY",
                credential_name="scoped",
                models=("shared-model",),
                default_model="shared-model",
            ),
        ),
        scoped_models=(ScopedModelConfig(provider="scoped", model="shared-model"),),
    )
    manager = SessionManager(
        AxisPaths(
            home=tmp_path / "axis-home",
            agents_home=tmp_path / "agents-home",
        )
    )
    record = manager.create_session(
        cwd=tmp_path,
        model="shared-model",
        provider_name=None,
        session_id="legacy-session",
    )
    credential_store = FileCredentialStore(tmp_path / "credentials.json")
    credential_store.set("scoped", "stored-key")

    selection, _settings = _resolve_tui_startup_selection(
        settings,
        record=record,
        provider_name=None,
        model=None,
        credential_store=credential_store,
    )

    assert selection.provider.name == "scoped"
    assert selection.model == "shared-model"

    credential_store.set("first", "default-key")
    new_selection, _settings = _resolve_tui_startup_selection(
        settings,
        record=None,
        provider_name=None,
        model=None,
        credential_store=credential_store,
    )
    assert new_selection.provider.name == "first"


def test_deepseek_tui_resumes_explicit_session_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ClosingFakeProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    paths = AxisPaths(
        home=tmp_path / "axis-home",
        agents_home=tmp_path / "agents-home",
    )
    manager = SessionManager(paths)
    record = manager.create_session(
        cwd=tmp_path,
        model="resume-model",
        provider_name="deepseek",
        session_id="session-1",
    )
    provider = ClosingFakeProvider()
    observed: list[tuple[str | None, str, str]] = []

    async def fake_run_tui_app(
        session: CodingSession,
        *,
        startup_message: str | None = None,
        initial_prompt: str | None = None,
        paths: AxisPaths | None = None,
    ) -> None:
        del startup_message, initial_prompt, paths
        observed.append((session.session_id, session.provider_name, session.model))

    monkeypatch.setattr(
        "axis_coding.cli.create_model_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr(
        "axis_coding.session.create_model_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr("axis_coding.cli.run_tui_app", fake_run_tui_app)

    asyncio.run(
        run_deepseek_tui_mode(
            model=None,
            cwd=tmp_path,
            session_id=record.id,
            paths=paths,
            session_manager=manager,
        )
    )

    assert observed == [("session-1", "deepseek", "resume-model")]
    assert len(manager.list_sessions(tmp_path)) == 1
    assert provider.closed is True


def test_cli_composes_prompt_cwd_and_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[tuple[str, str, Path, PrintOutputMode, ToolApprovalPolicy]] = []

    async def fake_run_deepseek_print_mode(
        *,
        prompt: str,
        model: str,
        cwd: Path,
        output: PrintOutputMode,
        tool_policy: ToolApprovalPolicy,
    ) -> bool:
        observed.append((prompt, model, cwd, output, tool_policy))
        return True

    monkeypatch.setattr(
        "axis_coding.cli.run_deepseek_print_mode",
        fake_run_deepseek_print_mode,
    )

    result = runner.invoke(
        app,
        [
            "-p",
            "Fix tests",
            "--cwd",
            str(tmp_path),
            "--model",
            "deepseek-v4-flash",
            "--output",
            "transcript",
            "--tool-policy",
            "deny",
        ],
        env={},
    )

    assert result.exit_code == 0
    assert observed == [
        (
            "Fix tests",
            "deepseek-v4-flash",
            tmp_path.resolve(),
            PrintOutputMode.TRANSCRIPT,
            ToolApprovalPolicy.DENY,
        )
    ]


def test_cli_reports_missing_key_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def missing_provider(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("Missing credentials for provider deepseek")

    monkeypatch.setattr("axis_coding.cli.create_model_provider", missing_provider)
    result = runner.invoke(
        app,
        ["-p", "Hello", "--cwd", str(tmp_path)],
        env={"DEEPSEEK_API_KEY": ""},
    )

    assert result.exit_code == 1
    assert "Missing credentials for provider deepseek" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_reports_empty_model_environment_without_traceback(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["-p", "Hello", "--cwd", str(tmp_path)],
        env={"DEEPSEEK_MODEL": "  "},
    )

    assert result.exit_code == 1
    assert "DEEPSEEK_MODEL" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_rejects_invalid_working_directory_before_provider_call(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    result = runner.invoke(
        app,
        ["-p", "Hello", "--cwd", str(missing)],
        env={},
    )

    assert result.exit_code == 2
    assert "Working directory does not exist" in result.stderr


def test_cli_uses_failure_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def failed_run(
        *,
        prompt: str,
        model: str,
        cwd: Path,
        output: PrintOutputMode,
        tool_policy: ToolApprovalPolicy,
    ) -> bool:
        del prompt, model, cwd, output, tool_policy
        return False

    monkeypatch.setattr("axis_coding.cli.run_deepseek_print_mode", failed_run)

    result = runner.invoke(app, ["-p", "Hello", "--cwd", str(tmp_path)], env={})

    assert result.exit_code == 1


def test_cli_reports_resource_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def failed_run(
        *,
        prompt: str,
        model: str,
        cwd: Path,
        output: PrintOutputMode,
        tool_policy: ToolApprovalPolicy,
    ) -> bool:
        del prompt, model, cwd, output, tool_policy
        raise ResourceError("Unknown skill: missing")

    monkeypatch.setattr("axis_coding.cli.run_deepseek_print_mode", failed_run)

    result = runner.invoke(app, ["-p", "/skill:missing", "--cwd", str(tmp_path)], env={})

    assert result.exit_code == 1
    assert result.stderr == "Error: Unknown skill: missing\n"
    assert "Traceback" not in result.stderr


def test_deepseek_print_wrapper_closes_provider(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    class ClosingFakeProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([[ProviderResponseEndEvent(message=AssistantMessage(content="Done"))]])
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    provider = ClosingFakeProvider()
    monkeypatch.setattr(
        "axis_coding.cli.create_model_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr(
        "axis_coding.session.create_model_provider",
        lambda *_args, **_kwargs: provider,
    )

    succeeded = asyncio.run(
        run_deepseek_print_mode(
            prompt="Hello",
            model="fake-model",
            cwd=tmp_path,
            paths=AxisPaths(
                home=tmp_path / "axis-home",
                agents_home=tmp_path / "agents-home",
            ),
        )
    )

    assert succeeded is True
    assert provider.closed is True
    assert capsys.readouterr().out == "Done\n"
    session_files = [
        path
        for path in (tmp_path / "axis-home" / "sessions").rglob("*.jsonl")
        if path.name != "index.jsonl"
    ]
    assert len(session_files) == 1


def test_deepseek_tui_wrapper_owns_provider_and_persistent_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ClosingFakeProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__(
                [[ProviderResponseEndEvent(message=AssistantMessage(content="TUI done"))]]
            )
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    provider = ClosingFakeProvider()
    observed_models: list[str] = []

    async def fake_run_tui_app(
        session: CodingSession,
        *,
        startup_message: str | None = None,
        initial_prompt: str | None = None,
        paths: AxisPaths | None = None,
    ) -> None:
        del startup_message, initial_prompt, paths
        observed_models.append(session.model)
        async for _event in session.prompt("TUI prompt"):
            pass

    monkeypatch.setattr(
        "axis_coding.cli.create_model_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr(
        "axis_coding.session.create_model_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr("axis_coding.cli.run_tui_app", fake_run_tui_app)
    paths = AxisPaths(
        home=tmp_path / "axis-home",
        agents_home=tmp_path / "agents-home",
    )

    asyncio.run(
        run_deepseek_tui_mode(
            model="fake-model",
            cwd=tmp_path,
            paths=paths,
        )
    )

    session_files = [
        path for path in paths.sessions_dir.rglob("*.jsonl") if path.name != "index.jsonl"
    ]
    assert observed_models == ["fake-model"]
    assert provider.closed is True
    assert len(session_files) == 1
    state = asyncio.run(SessionState.from_storage(JsonlSessionStorage(session_files[0])))
    assert state.messages == (
        UserMessage(content="TUI prompt"),
        AssistantMessage(content="TUI done"),
    )


def test_deepseek_tui_opens_login_capable_ui_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[tuple[str, str | None]] = []

    def missing_provider(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("missing key")

    async def fake_run_tui_app(
        session: CodingSession,
        *,
        startup_message: str | None = None,
        initial_prompt: str | None = None,
        paths: AxisPaths | None = None,
    ) -> None:
        del initial_prompt, paths
        observed.append((session.provider_name, startup_message))

    monkeypatch.setattr("axis_coding.cli.create_model_provider", missing_provider)
    monkeypatch.setattr("axis_coding.cli.run_tui_app", fake_run_tui_app)
    paths = AxisPaths(home=tmp_path / ".axis", agents_home=tmp_path / ".agents")

    asyncio.run(
        run_deepseek_tui_mode(
            model="deepseek-v4-pro",
            cwd=tmp_path,
            paths=paths,
        )
    )

    assert observed == [
        (
            "deepseek",
            "Login required. Run /login to choose a provider, or /login deepseek to continue.",
        )
    ]


def test_run_print_mode_can_emit_json_events(tmp_path: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()
    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="fake-model"),
                ProviderTextDeltaEvent(delta="Hello"),
                ProviderResponseEndEvent(message=AssistantMessage(content="Hello")),
            ]
        ]
    )

    succeeded = asyncio.run(
        run_print_mode(
            prompt="Say hello",
            model="fake-model",
            cwd=tmp_path,
            provider=provider,
            output=PrintOutputMode.JSON,
            stdout=stdout,
            stderr=stderr,
        )
    )

    events = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert succeeded is True
    assert events[0]["type"] == "memory_context"
    assert events[0]["warnings"] == [
        "Memory Bank is not initialized. Run /memory init to enable it."
    ]
    assert events[1] == {"type": "agent_start"}
    assert events[-1] == {"type": "agent_end"}
    assert any(event["type"] == "message_delta" for event in events)
    assert stderr.getvalue() == ""


def test_run_print_mode_can_emit_live_transcript(tmp_path: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()
    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="fake-model"),
                ProviderTextDeltaEvent(delta="Hel"),
                ProviderTextDeltaEvent(delta="lo"),
                ProviderResponseEndEvent(message=AssistantMessage(content="Hello")),
            ]
        ]
    )

    succeeded = asyncio.run(
        run_print_mode(
            prompt="Say hello",
            model="fake-model",
            cwd=tmp_path,
            provider=provider,
            output=PrintOutputMode.TRANSCRIPT,
            stdout=stdout,
            stderr=stderr,
        )
    )

    assert succeeded is True
    assert stdout.getvalue() == "Hello\n"
    assert stderr.getvalue() == (
        "… Memory warning: Memory Bank is not initialized. Run /memory init to enable it.\n"
    )


def test_run_print_mode_persists_to_injected_storage(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "print-session.jsonl")
    provider = FakeProvider(
        [[ProviderResponseEndEvent(message=AssistantMessage(content="Persisted"))]]
    )

    succeeded = asyncio.run(
        run_print_mode(
            prompt="Remember this",
            model="fake-model",
            cwd=tmp_path,
            provider=provider,
            storage=storage,
            stdout=StringIO(),
            stderr=StringIO(),
        )
    )

    entries = asyncio.run(storage.read_all())
    state = SessionState.from_entries(entries)
    assert succeeded is True
    assert [entry.message for entry in entries if isinstance(entry, MessageEntry)] == [
        UserMessage(content="Remember this"),
        AssistantMessage(content="Persisted"),
    ]
    assert state.messages == (
        UserMessage(content="Remember this"),
        AssistantMessage(content="Persisted"),
    )
