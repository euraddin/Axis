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
    UserMessage,
)
from axis_ai import (
    FakeProvider,
    OpenAICompatibleConfig,
    ProviderErrorEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
)
from axis_coding import AxisPaths, CodingSession, ResourceError, __version__
from axis_coding.cli import (
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

    async def fake_run_deepseek_tui_mode(*, model: str, cwd: Path) -> None:
        observed.append((model, cwd))

    monkeypatch.setattr("axis_coding.cli.run_deepseek_tui_mode", fake_run_deepseek_tui_mode)

    result = runner.invoke(
        app,
        ["--cwd", str(tmp_path), "--model", "deepseek-v4-flash"],
        env={},
    )

    assert result.exit_code == 0
    assert observed == [("deepseek-v4-flash", tmp_path.resolve())]


def test_cli_composes_prompt_cwd_and_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[tuple[str, str, Path, PrintOutputMode]] = []

    async def fake_run_deepseek_print_mode(
        *,
        prompt: str,
        model: str,
        cwd: Path,
        output: PrintOutputMode,
    ) -> bool:
        observed.append((prompt, model, cwd, output))
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
        )
    ]


def test_cli_reports_missing_key_without_traceback(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["-p", "Hello", "--cwd", str(tmp_path)],
        env={"DEEPSEEK_API_KEY": ""},
    )

    assert result.exit_code == 1
    assert "Missing required environment variable: DEEPSEEK_API_KEY" in result.stderr
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
    ) -> bool:
        del prompt, model, cwd, output
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
    ) -> bool:
        del prompt, model, cwd, output
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
        "axis_coding.cli.deepseek_v4_config_from_env",
        lambda: OpenAICompatibleConfig(api_key="test-key"),
    )
    monkeypatch.setattr(
        "axis_coding.cli.OpenAICompatibleProvider",
        lambda _config: provider,
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
    assert len(list((tmp_path / "axis-home" / "sessions").rglob("*.jsonl"))) == 1


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

    async def fake_run_tui_app(session: CodingSession) -> None:
        observed_models.append(session.model)
        async for _event in session.prompt("TUI prompt"):
            pass

    monkeypatch.setattr(
        "axis_coding.cli.deepseek_v4_config_from_env",
        lambda: OpenAICompatibleConfig(api_key="test-key"),
    )
    monkeypatch.setattr(
        "axis_coding.cli.OpenAICompatibleProvider",
        lambda _config: provider,
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

    session_files = list(paths.sessions_dir.rglob("*.jsonl"))
    assert observed_models == ["fake-model"]
    assert provider.closed is True
    assert len(session_files) == 1
    state = asyncio.run(SessionState.from_storage(JsonlSessionStorage(session_files[0])))
    assert state.messages == (
        UserMessage(content="TUI prompt"),
        AssistantMessage(content="TUI done"),
    )


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
    assert events[0] == {"type": "agent_start"}
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
    assert stderr.getvalue() == ""


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
