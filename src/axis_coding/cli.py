"""Command-line entry point for Axis."""

import asyncio
from pathlib import Path
from typing import Annotated, TextIO

import typer

from axis_agent import JsonlSessionStorage, SessionEntry, SessionStorage
from axis_ai import (
    ModelProvider,
    OpenAICompatibleProvider,
    deepseek_model_from_env,
    deepseek_v4_config_from_env,
)
from axis_coding import __version__
from axis_coding.paths import AxisPaths
from axis_coding.rendering import PrintOutputMode, create_event_renderer
from axis_coding.resources import AxisResourcePaths, ResourceError
from axis_coding.session import CodingSession, CodingSessionConfig
from axis_coding.tui import run_tui_app

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
    except (ResourceError, RuntimeError) as exc:
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
    provider = OpenAICompatibleProvider(deepseek_v4_config_from_env())
    try:
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model=model,
                storage=JsonlSessionStorage(runtime_paths.new_session_path(cwd)),
                cwd=cwd,
                resource_paths=AxisResourcePaths(paths=runtime_paths, cwd=cwd),
            )
        )
        await run_tui_app(session)
    finally:
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
    provider = OpenAICompatibleProvider(deepseek_v4_config_from_env())
    try:
        return await run_print_mode(
            prompt=prompt,
            model=model,
            cwd=cwd,
            provider=provider,
            storage=JsonlSessionStorage(runtime_paths.new_session_path(cwd)),
            resource_paths=AxisResourcePaths(paths=runtime_paths, cwd=cwd),
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
        )
    )
    renderer = create_event_renderer(output, stdout=stdout, stderr=stderr)
    async for event in session.prompt(prompt):
        renderer.render(event)
    return renderer.finish()


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
