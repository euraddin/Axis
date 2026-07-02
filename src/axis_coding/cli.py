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
from axis_coding.permissions import (
    ToolApprovalPolicy,
    approval_handler_for_policy,
)
from axis_coding.provider_config import (
    OpenAICompatibleProviderConfig,
    ProviderConfigError,
    ProviderSelection,
    ProviderSettings,
    load_provider_settings,
    provider_default_thinking_level,
    provider_has_usable_credentials,
    resolve_provider_selection,
    upsert_provider,
)
from axis_coding.provider_runtime import LoginRequiredProvider, create_model_provider
from axis_coding.rendering import PrintOutputMode, create_event_renderer
from axis_coding.resources import AxisResourcePaths, ResourceError
from axis_coding.session import CodingSession, CodingSessionConfig
from axis_coding.session_manager import CodingSessionRecord, SessionManager
from axis_coding.thinking import DEFAULT_THINKING_LEVEL, ThinkingLevel
from axis_coding.tui import run_tui_app
from axis_coding.tui.config import TuiConfigError

app = typer.Typer(
    name="axis",
    help="Axis personal coding agent.",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(
    prompt_args: Annotated[
        list[str] | None,
        typer.Argument(help="Optional prompt submitted immediately in TUI mode."),
    ] = None,
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
        typer.Option("--model", "-m", help="Model name to request from the provider."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Configured OpenAI-compatible provider name."),
    ] = None,
    resume: Annotated[
        str | None,
        typer.Option("--resume", help="Resume an indexed session id in TUI mode."),
    ] = None,
    new_session: Annotated[
        bool,
        typer.Option("--new-session", help="Force a new TUI session."),
    ] = False,
    auto_compact_threshold: Annotated[
        int | None,
        typer.Option(
            "--auto-compact-threshold",
            help="Context threshold displayed by the TUI compact status.",
        ),
    ] = None,
    output: Annotated[
        PrintOutputMode,
        typer.Option("--output", "-o", help="Output format for print mode."),
    ] = PrintOutputMode.TEXT,
    tool_policy: Annotated[
        ToolApprovalPolicy,
        typer.Option(
            "--tool-policy",
            help="Protected-tool policy for print mode: ask, deny, or allow.",
        ),
    ] = ToolApprovalPolicy.ASK,
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
        if prompt is None:
            initial_prompt = " ".join(prompt_args or []).strip() or None
            asyncio.run(
                run_deepseek_tui_mode(
                    model=model,
                    cwd=root,
                    session_id=resume,
                    new_session=new_session,
                    provider_name=provider,
                    auto_compact_token_threshold=auto_compact_threshold,
                    initial_prompt=initial_prompt,
                )
            )
            return
        if prompt_args:
            raise RuntimeError("Positional prompts are only supported in TUI mode")
        selected_model = model or deepseek_model_from_env()
        succeeded = asyncio.run(
            run_deepseek_print_mode(
                prompt=prompt,
                model=selected_model,
                cwd=root,
                output=output,
                tool_policy=tool_policy,
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


def _explicit_resume_record(
    manager: SessionManager,
    session_id: str | None,
) -> CodingSessionRecord | None:
    if session_id is None:
        return None
    record = manager.get_session(session_id)
    if record is None:
        raise RuntimeError(f"Unknown session: {session_id}")
    return record


def _resolve_tui_startup_selection(
    settings: ProviderSettings,
    *,
    record: CodingSessionRecord | None,
    provider_name: str | None,
    model: str | None,
    credential_store: FileCredentialStore,
) -> tuple[ProviderSelection, ProviderSettings]:
    if provider_name is not None or model is not None:
        target_provider = provider_name or settings.default_provider
        target_model = model
        return _configured_selection(settings, target_provider, target_model)

    if record is not None:
        record_provider = getattr(record, "provider_name", None)
        record_model = getattr(record, "model", None)
        if isinstance(record_model, str) and record_model:
            if isinstance(record_provider, str) and record_provider:
                try:
                    return _configured_selection(settings, record_provider, record_model)
                except ProviderConfigError:
                    pass
            for choice in settings.scoped_models:
                if choice.model != record_model:
                    continue
                try:
                    provider = settings.get_provider(choice.provider)
                except ProviderConfigError:
                    continue
                if record_model not in provider.models:
                    continue
                if provider_has_usable_credentials(
                    provider,
                    credential_reader=credential_store,
                ):
                    return ProviderSelection(provider, record_model), settings
            for provider in settings.providers:
                if record_model in provider.models:
                    return ProviderSelection(provider, record_model), settings

    default_provider = settings.get_provider()
    default_model = (
        deepseek_model_from_env()
        if default_provider.name == "deepseek"
        else default_provider.default_model
    )
    selection, settings = _configured_selection(
        settings,
        default_provider.name,
        default_model,
    )
    if provider_has_usable_credentials(
        selection.provider,
        credential_reader=credential_store,
    ):
        return selection, settings
    for provider in settings.providers:
        if provider_has_usable_credentials(provider, credential_reader=credential_store):
            return ProviderSelection(provider, provider.default_model), settings
    return selection, settings


def _configured_selection(
    settings: ProviderSettings,
    provider_name: str,
    model: str | None,
) -> tuple[ProviderSelection, ProviderSettings]:
    provider = settings.get_provider(provider_name)
    selected_model = model or provider.default_model
    if selected_model not in provider.models:
        thinking_models = provider.thinking_models
        if thinking_models:
            thinking_models = (*thinking_models, selected_model)
        provider = replace(
            provider,
            models=(*provider.models, selected_model),
            default_model=selected_model,
            thinking_models=thinking_models,
        )
        settings = upsert_provider(settings, provider)
    return resolve_provider_selection(
        settings,
        provider_name=provider.name,
        model=selected_model,
    ), settings


async def run_deepseek_tui_mode(
    *,
    model: str | None,
    cwd: Path,
    session_id: str | None = None,
    new_session: bool = False,
    provider_name: str | None = None,
    auto_compact_token_threshold: int | None = None,
    initial_prompt: str | None = None,
    paths: AxisPaths | None = None,
    session_manager: SessionManager | None = None,
) -> None:
    """Resolve startup provider/session state and run the interactive TUI."""
    if new_session and session_id is not None:
        raise RuntimeError("--resume and --new-session cannot be used together")
    if auto_compact_token_threshold is not None and auto_compact_token_threshold <= 0:
        raise RuntimeError("--auto-compact-threshold must be greater than 0")
    runtime_paths = paths if paths is not None else AxisPaths()
    manager = session_manager or SessionManager(runtime_paths)
    settings = load_provider_settings(runtime_paths)
    record = _explicit_resume_record(manager, session_id)
    credential_store = FileCredentialStore(credentials_path(runtime_paths))
    selection, settings = _resolve_tui_startup_selection(
        settings,
        record=record,
        provider_name=provider_name,
        model=model,
        credential_store=credential_store,
    )
    provider_config = selection.provider
    selected_model = selection.model
    thinking_level = (
        provider_default_thinking_level(provider_config, model=selected_model)
        or DEFAULT_THINKING_LEVEL
    )
    startup_message: str | None = None
    try:
        provider = create_model_provider(
            provider_config,
            credential_store=credential_store,
            model=selected_model,
            thinking_level=thinking_level,
        )
        runtime_provider_config = provider_config
    except RuntimeError:
        startup_message = (
            "Login required. Run /login to choose a provider, or "
            f"/login {provider_config.name} to continue."
        )
        provider = LoginRequiredProvider(startup_message)
        runtime_provider_config = None
    if record is None:
        record = manager.create_session(
            cwd=cwd,
            model=selected_model,
            provider_name=provider_config.name,
        )
    session: CodingSession | None = None
    try:
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model=record.model or selected_model,
                storage=JsonlSessionStorage(record.path),
                cwd=record.cwd,
                resource_paths=AxisResourcePaths(paths=runtime_paths, cwd=record.cwd),
                session_id=record.id,
                session_manager=manager,
                provider_name=provider_config.name,
                provider_settings=settings,
                runtime_provider_config=runtime_provider_config,
                thinking_level=thinking_level,
                auto_compact_token_threshold=auto_compact_token_threshold,
            )
        )
        await run_tui_app(
            session,
            startup_message=startup_message,
            initial_prompt=initial_prompt,
        )
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
    tool_policy: ToolApprovalPolicy = ToolApprovalPolicy.ASK,
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
            tool_policy=tool_policy,
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
    tool_policy: ToolApprovalPolicy = ToolApprovalPolicy.ASK,
    stdin: TextIO | None = None,
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
            tool_approval_handler=approval_handler_for_policy(
                tool_policy,
                cwd=cwd,
                stdin=stdin,
                stderr=stderr,
            ),
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
