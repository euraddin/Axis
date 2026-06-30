"""Command-line entry point for Axis."""

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Annotated, TextIO

import typer

from axis_agent import JsonlSessionStorage, SessionEntry, SessionStorage
from axis_ai import (
    ModelProvider,
    deepseek_model_from_env,
)
from axis_coding import __version__
from axis_coding.credentials import FileCredentialStore, credentials_path
from axis_coding.paths import AxisPaths
from axis_coding.provider_config import (
    OpenAICompatibleProviderConfig,
    ProviderConfigError,
    ProviderSettings,
    load_provider_settings,
    provider_default_thinking_level,
    upsert_provider,
)
from axis_coding.provider_runtime import LoginRequiredProvider, create_model_provider
from axis_coding.rendering import PrintOutputMode, create_event_renderer
from axis_coding.resources import AxisResourcePaths, ResourceError
from axis_coding.session import CodingSession, CodingSessionConfig
from axis_coding.session_manager import SessionManager
from axis_coding.thinking import ThinkingLevel
from axis_coding.tui import run_tui_app
from axis_coding.tui.config import TuiConfigError

app = typer.Typer(
    name="axis",
    help="Axis personal coding agent.",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", "-p", help="Run one non-interactive prompt."),
    ] = None,
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Working directory for coding tools."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="DeepSeek model name."),
    ] = None,
    output: Annotated[
        PrintOutputMode,
        typer.Option("--output", "-o", help="Output format for print mode."),
    ] = PrintOutputMode.TEXT,
    version: Annotated[
        bool,
        typer.Option("--version", help="Show Axis's version and exit."),
    ] = False,
) -> None:
    """Run Axis in interactive TUI or non-interactive print mode."""
    if version:
        typer.echo(f"axis {__version__}")
        raise typer.Exit()
    root = _resolve_cwd(cwd)
    try:
        selected_model = model or deepseek_model_from_env()
        if prompt is None:
            asyncio.run(
                run_deepseek_tui_mode(
                    model=selected_model,
                    cwd=root,
                )
            )
            return
        succeeded = asyncio.run(
            run_deepseek_print_mode(
                prompt=prompt,
                model=selected_model,
                cwd=root,
                output=output,
            )
        )
    except (ProviderConfigError, ResourceError, RuntimeError, TuiConfigError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    except KeyboardInterrupt:
        typer.echo("Cancelled.", err=True)
        raise typer.Exit(code=130) from None
    if not succeeded:
        raise typer.Exit(code=1)


async def run_deepseek_tui_mode(
    *,
    model: str,
    cwd: Path,
    paths: AxisPaths | None = None,
) -> None:
    """Create one DeepSeek coding session and run the interactive Textual app."""
    runtime_paths = paths if paths is not None else AxisPaths()
    manager = SessionManager(runtime_paths)
    settings = load_provider_settings(runtime_paths)
    provider_config = settings.get_provider("deepseek")
    if model not in provider_config.models:
        provider_config = replace(
            provider_config,
            models=(*provider_config.models, model),
            default_model=model,
            thinking_models=(*provider_config.thinking_models, model),
        )
        settings = upsert_provider(settings, provider_config)
    thinking_level = provider_default_thinking_level(provider_config, model=model)
    if thinking_level is None:
        raise RuntimeError(f"DeepSeek model does not declare thinking support: {model}")
    credential_store = FileCredentialStore(credentials_path(runtime_paths))
    startup_message: str | None = None
    try:
        provider = create_model_provider(
            provider_config,
            credential_store=credential_store,
            model=model,
            thinking_level=thinking_level,
        )
        runtime_provider_config = provider_config
    except RuntimeError:
        startup_message = (
            "Login required. Run /login or /login deepseek to save a DeepSeek API key."
        )
        provider = LoginRequiredProvider(startup_message)
        runtime_provider_config = None
    record = manager.create_session(
        cwd=cwd,
        model=model,
        provider_name=provider_config.name,
    )
    session: CodingSession | None = None
    try:
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model=model,
                storage=JsonlSessionStorage(record.path),
                cwd=record.cwd,
                resource_paths=AxisResourcePaths(paths=runtime_paths, cwd=record.cwd),
                session_id=record.id,
                session_manager=manager,
                provider_name=provider_config.name,
                provider_settings=settings,
                runtime_provider_config=runtime_provider_config,
                thinking_level=thinking_level,
            )
        )
        await run_tui_app(session, startup_message=startup_message)
    finally:
        if session is not None:
            await session.aclose()
        await provider.aclose()


async def run_deepseek_print_mode(
    *,
    prompt: str,
    model: str,
    cwd: Path,
    output: PrintOutputMode = PrintOutputMode.TEXT,
    paths: AxisPaths | None = None,
) -> bool:
    """Run print mode with the environment-configured DeepSeek provider."""
    runtime_paths = paths if paths is not None else AxisPaths()
    settings = load_provider_settings(runtime_paths)
    provider_config = settings.get_provider("deepseek")
    if model not in provider_config.models:
        provider_config = replace(
            provider_config,
            models=(*provider_config.models, model),
            default_model=model,
            thinking_models=(*provider_config.thinking_models, model),
        )
        settings = upsert_provider(settings, provider_config)
    thinking_level = provider_default_thinking_level(provider_config, model=model)
    if thinking_level is None:
        raise RuntimeError(f"DeepSeek model does not declare thinking support: {model}")
    provider = create_model_provider(
        provider_config,
        credential_store=FileCredentialStore(credentials_path(runtime_paths)),
        model=model,
        thinking_level=thinking_level,
    )
    manager = SessionManager(runtime_paths)
    record = manager.create_session(
        cwd=cwd,
        model=model,
        provider_name=provider_config.name,
    )
    try:
        return await run_print_mode(
            prompt=prompt,
            model=model,
            cwd=record.cwd,
            provider=provider,
            storage=JsonlSessionStorage(record.path),
            resource_paths=AxisResourcePaths(paths=runtime_paths, cwd=record.cwd),
            session_id=record.id,
            session_manager=manager,
            provider_name=provider_config.name,
            provider_settings=settings,
            runtime_provider_config=provider_config,
            thinking_level=thinking_level,
            output=output,
        )
    finally:
        await provider.aclose()


async def run_print_mode(
    *,
    prompt: str,
    model: str,
    cwd: Path,
    provider: ModelProvider,
    storage: SessionStorage | None = None,
    resource_paths: AxisResourcePaths | None = None,
    session_id: str | None = None,
    session_manager: SessionManager | None = None,
    provider_name: str = "deepseek",
    provider_settings: ProviderSettings | None = None,
    runtime_provider_config: OpenAICompatibleProviderConfig | None = None,
    thinking_level: ThinkingLevel = "xhigh",
    output: PrintOutputMode = PrintOutputMode.TEXT,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> bool:
    """Run one persistent session prompt through the selected event renderer."""
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model=model,
            storage=storage if storage is not None else _MemorySessionStorage(),
            cwd=cwd,
            resource_paths=resource_paths,
            session_id=session_id,
            session_manager=session_manager,
            provider_name=provider_name,
            provider_settings=provider_settings,
            runtime_provider_config=runtime_provider_config,
            thinking_level=thinking_level,
        )
    )
    renderer = create_event_renderer(output, stdout=stdout, stderr=stderr)
    try:
        async for event in session.prompt(prompt):
            renderer.render(event)
        return renderer.finish()
    finally:
        await session.aclose()


def _resolve_cwd(cwd: Path | None) -> Path:
    root = (Path.cwd() if cwd is None else cwd).expanduser().resolve()
    if not root.exists():
        raise typer.BadParameter(f"Working directory does not exist: {root}", param_hint="--cwd")
    if not root.is_dir():
        raise typer.BadParameter(
            f"Working directory is not a directory: {root}", param_hint="--cwd"
        )
    return root


class _MemorySessionStorage:
    """Ephemeral default for the provider-injected, reusable print-mode core."""

    def __init__(self) -> None:
        self.entries: list[SessionEntry] = []

    async def append(self, entry: SessionEntry) -> None:
        self.entries.append(entry)

    async def read_all(self) -> list[SessionEntry]:
        return list(self.entries)
