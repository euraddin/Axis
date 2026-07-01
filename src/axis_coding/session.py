"""Persistent coding-session composition around the portable AgentHarness."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Literal

from axis_agent import (
    AgentEvent,
    AgentHarness,
    AgentHarnessConfig,
    AgentMessage,
    AgentTool,
    AssistantMessage,
    BranchSummaryEntry,
    CompactionEntry,
    JsonlSessionStorage,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    QueueUpdateEvent,
    SessionEntry,
    SessionInfoEntry,
    SessionState,
    SessionStorage,
    ThinkingLevelChangeEntry,
    UserMessage,
    path_to_entry,
)
from axis_ai import ModelProvider
from axis_coding.commands import (
    CommandRegistry,
    CommandResult,
    create_default_command_registry,
)
from axis_coding.context import (
    ProjectContextFile,
    discover_project_context_with_diagnostics,
)
from axis_coding.context_window import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    ContextUsageEstimate,
    estimate_context_usage,
)
from axis_coding.credentials import FileCredentialStore, credentials_path
from axis_coding.prompt_templates import (
    PromptTemplate,
    expand_prompt_template_command,
    load_prompt_templates_with_diagnostics,
)
from axis_coding.provider_config import (
    OpenAICompatibleProviderConfig,
    ProviderConfigError,
    ProviderSettings,
    load_provider_settings,
    provider_default_thinking_level,
    provider_has_usable_credentials,
    provider_thinking_levels,
    save_provider_settings,
    set_default_provider_model,
    toggle_scoped_model,
)
from axis_coding.provider_runtime import ClosableModelProvider, create_model_provider
from axis_coding.reload import CodingReloadSummary, ReloadCategorySummary
from axis_coding.resources import (
    AxisResourcePaths,
    ResourceDiagnostic,
    resource_paths_with_cwd,
)
from axis_coding.session_export import (
    default_session_export_path,
    export_session_artifact,
    normalize_export_format,
)
from axis_coding.session_manager import CodingSessionRecord, SessionManager
from axis_coding.skills import Skill, expand_skill_command, load_skills_with_diagnostics
from axis_coding.system_prompt import BuildSystemPromptOptions, build_system_prompt
from axis_coding.thinking import (
    DEFAULT_THINKING_LEVEL,
    THINKING_LEVELS,
    ThinkingLevel,
    next_thinking_level,
    normalize_thinking_level,
)
from axis_coding.tools import create_bash_tool, create_coding_tools

type StreamingBehavior = Literal["steer", "follow_up"]


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """A selectable model and the OpenAI-compatible provider serving it."""

    provider_name: str
    model: str


@dataclass(frozen=True, slots=True)
class TerminalCommandRequest:
    """Parsed input-bar terminal command."""

    command: str
    add_to_context: bool


@dataclass(frozen=True, slots=True)
class TerminalCommandResult:
    """Result of one input-bar terminal command."""

    command: str
    output: str
    exit_code: int | None
    ok: bool
    added_to_context: bool


@dataclass(frozen=True, slots=True)
class SessionTreeChoice:
    """One branchable entry displayed by the tree picker."""

    entry_id: str
    label: str
    active: bool = False
    is_tool_call: bool = False


@dataclass(frozen=True, slots=True)
class SessionTreeBranchResult:
    """Visible branch result plus optional prompt prefill."""

    message: str
    input_prefill: str | None = None


class _TerminalCancellationToken:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled


class CodingSessionError(RuntimeError):
    """A coding session cannot be safely created or restored."""


@dataclass(frozen=True, slots=True)
class CodingSessionConfig:
    """Dependencies and defaults used to load one coding session."""

    provider: ModelProvider
    model: str
    storage: SessionStorage
    cwd: Path
    system: str | None = None
    tools: list[AgentTool] | None = None
    resource_paths: AxisResourcePaths | None = None
    session_id: str | None = None
    session_manager: SessionManager | None = None
    provider_name: str = "deepseek"
    provider_settings: ProviderSettings | None = None
    runtime_provider_config: OpenAICompatibleProviderConfig | None = None
    thinking_level: ThinkingLevel = DEFAULT_THINKING_LEVEL
    auto_compact_token_threshold: int | None = None


class CodingSession:
    """Axis coding environment with durable append-only transcript state."""

    def __init__(
        self,
        config: CodingSessionConfig,
        *,
        cwd: Path,
        state: SessionState,
        harness: AgentHarness,
        resource_paths: AxisResourcePaths,
        context_files: tuple[ProjectContextFile, ...] = (),
        skills: tuple[Skill, ...] = (),
        prompt_templates: tuple[PromptTemplate, ...] = (),
        resource_diagnostics: tuple[ResourceDiagnostic, ...] = (),
        pending_initial_entries: tuple[SessionEntry, ...] = (),
    ) -> None:
        self._config = config
        self._cwd = cwd
        self._state = state
        self._harness = harness
        self._resource_paths = resource_paths
        self._context_files = context_files
        self._skills = skills
        self._prompt_templates = prompt_templates
        self._resource_diagnostics = resource_diagnostics
        self._provider_name = config.provider_name
        self._provider_settings = config.provider_settings
        self._runtime_provider_config = config.runtime_provider_config
        self._thinking_level = normalize_thinking_level(
            state.thinking_level or config.thinking_level
        )
        self._owned_providers: list[ClosableModelProvider] = []
        self._credential_store = FileCredentialStore(credentials_path(resource_paths.paths))
        self._current_entry_id = state.active_leaf_id
        self._persisted_message_count = len(state.messages)
        self._pending_initial_entries = pending_initial_entries
        self._terminal_signal: _TerminalCancellationToken | None = None
        self._command_registry = create_default_command_registry()

    @classmethod
    async def load(cls, config: CodingSessionConfig) -> CodingSession:
        """Load an existing session or prepare a new deferred session."""
        cwd = config.cwd.expanduser().resolve()
        if not cwd.exists():
            raise CodingSessionError(f"Working directory does not exist: {cwd}")
        if not cwd.is_dir():
            raise CodingSessionError(f"Working directory is not a directory: {cwd}")

        stored_entries = await config.storage.read_all()
        pending_initial_entries: tuple[SessionEntry, ...] = ()
        restored_state: SessionState | None
        if stored_entries:
            restored_state = SessionState.from_entries(stored_entries)
            _validate_restored_cwd(restored_state, cwd)
        else:
            restored_state = None

        tools = config.tools if config.tools is not None else create_coding_tools(cwd=cwd)
        resource_paths = resource_paths_with_cwd(config.resource_paths, cwd)
        context_files, resource_diagnostics = discover_project_context_with_diagnostics(
            resource_paths
        )
        skills, skill_diagnostics = load_skills_with_diagnostics(resource_paths)
        prompt_templates, prompt_diagnostics = load_prompt_templates_with_diagnostics(
            resource_paths
        )
        resource_diagnostics = (
            *resource_diagnostics,
            *skill_diagnostics,
            *prompt_diagnostics,
        )
        stored_system = (
            restored_state.session_info.system
            if restored_state is not None and restored_state.session_info is not None
            else None
        )
        if stored_system is not None:
            system = stored_system
        elif config.system is not None:
            system = config.system
        else:
            system = build_system_prompt(
                BuildSystemPromptOptions(
                    cwd=cwd,
                    current_date=date.today(),
                    tools=tools,
                    skills=skills,
                    context_files=context_files,
                )
            )

        if restored_state is None:
            record = (
                config.session_manager.get_session(config.session_id)
                if config.session_manager is not None and config.session_id is not None
                else None
            )
            info = SessionInfoEntry(
                cwd=str(cwd),
                title=record.title if record is not None else None,
                system=system,
            )
            model = ModelChangeEntry(parent_id=info.id, model=config.model)
            initial_entries: list[SessionEntry] = [info, model]
            if config.provider_settings is not None:
                thinking = ThinkingLevelChangeEntry(
                    parent_id=model.id,
                    thinking_level=config.thinking_level,
                )
                initial_entries.append(thinking)
            state = SessionState.from_entries(initial_entries)
            pending_initial_entries = tuple(initial_entries)
        else:
            state = restored_state
        harness = AgentHarness(
            AgentHarnessConfig(
                provider=config.provider,
                model=state.model or config.model,
                system=system,
                tools=tools,
            ),
            messages=state.messages,
        )
        session = cls(
            config,
            cwd=cwd,
            state=state,
            harness=harness,
            resource_paths=resource_paths,
            context_files=context_files,
            skills=skills,
            prompt_templates=prompt_templates,
            resource_diagnostics=resource_diagnostics,
            pending_initial_entries=pending_initial_entries,
        )
        session._sync_thinking_level_to_active_model()
        if config.runtime_provider_config is not None:
            session._refresh_runtime_provider()
        return session

    @property
    def cwd(self) -> Path:
        """Return the resolved working directory bound to coding tools."""
        return self._cwd

    @property
    def model(self) -> str:
        """Return the restored or configured active model."""
        return self._harness.config.model

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def provider_settings(self) -> ProviderSettings | None:
        return self._provider_settings

    @property
    def available_providers(self) -> tuple[str, ...]:
        if self._provider_settings is None:
            return (self._provider_name,)
        return tuple(provider.name for provider in self._usable_provider_configs())

    @property
    def available_models(self) -> tuple[str, ...]:
        provider = self._active_provider_config()
        if provider is None:
            return (self.model,)
        if not self._provider_is_usable(provider):
            return ()
        return provider.models

    @property
    def available_model_choices(self) -> tuple[ModelChoice, ...]:
        if self._provider_settings is None:
            return (ModelChoice(self._provider_name, self.model),)
        return tuple(
            ModelChoice(provider.name, model)
            for provider in self._usable_provider_configs()
            for model in provider.models
        )

    @property
    def scoped_model_choices(self) -> tuple[ModelChoice, ...]:
        if self._provider_settings is None:
            return ()
        available = set(self.available_model_choices)
        return tuple(
            choice
            for choice in (
                ModelChoice(item.provider, item.model)
                for item in self._provider_settings.scoped_models
            )
            if choice in available
        )

    @property
    def thinking_level(self) -> ThinkingLevel:
        return self._thinking_level

    @property
    def available_thinking_levels(self) -> tuple[ThinkingLevel, ...]:
        provider = self._active_provider_config()
        if provider is None:
            return THINKING_LEVELS if self._provider_settings is None else ()
        return provider_thinking_levels(provider, model=self.model)

    @property
    def thinking_unavailable_reason(self) -> str | None:
        if self.available_thinking_levels:
            return None
        provider = self._active_provider_config()
        if provider is None:
            return "Active provider settings are not available"
        if provider.thinking_levels is None:
            return f"Provider {provider.name} does not declare thinking_levels"
        return f"{provider.name}:{self.model} is not declared in thinking_models"

    @property
    def context_token_estimate(self) -> int:
        return self.context_usage.total_tokens

    @property
    def context_usage(self) -> ContextUsageEstimate:
        """Estimate the current system/messages/tools request snapshot."""
        return estimate_context_usage(
            system=self.system,
            messages=self.messages,
            tools=self.tools,
        )

    @property
    def context_window_tokens(self) -> int:
        provider = self._active_provider_config()
        if provider is None:
            return DEFAULT_CONTEXT_WINDOW_TOKENS
        return provider.context_windows.get(self.model, DEFAULT_CONTEXT_WINDOW_TOKENS)

    @property
    def auto_compact_token_threshold(self) -> int | None:
        return self._config.auto_compact_token_threshold

    @property
    def system(self) -> str:
        """Return the system prompt used for future provider calls."""
        return self._harness.config.system

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        """Return the coding tools available to the harness."""
        return tuple(self._harness.config.tools)

    @property
    def resource_paths(self) -> AxisResourcePaths:
        """Return resource locations bound to this session cwd."""
        return self._resource_paths

    @property
    def context_files(self) -> tuple[ProjectContextFile, ...]:
        """Return discovered AGENTS.md instruction files."""
        return self._context_files

    @property
    def skills(self) -> tuple[Skill, ...]:
        """Return discovered skills after precedence resolution."""
        return self._skills

    @property
    def prompt_templates(self) -> tuple[PromptTemplate, ...]:
        """Return discovered prompt templates after precedence resolution."""
        return self._prompt_templates

    @property
    def resource_diagnostics(self) -> tuple[ResourceDiagnostic, ...]:
        """Return non-fatal resource loading problems."""
        return self._resource_diagnostics

    @property
    def command_registry(self) -> CommandRegistry:
        """Return this session's slash-command source of truth."""
        return self._command_registry

    @property
    def session_id(self) -> str | None:
        return self._config.session_id

    @property
    def session_manager(self) -> SessionManager | None:
        return self._config.session_manager

    @property
    def session_title(self) -> str | None:
        manager = self._config.session_manager
        if manager is not None and self.session_id is not None:
            record = manager.get_session(self.session_id)
            if record is not None:
                return record.title
        return self._state.session_info.title if self._state.session_info is not None else None

    @property
    def storage(self) -> SessionStorage:
        return self._config.storage

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        """Return the current authoritative Harness transcript."""
        return self._harness.messages

    @property
    def state(self) -> SessionState:
        """Return the most recently replayed durable state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Return whether an agent or input-bar terminal command is active."""
        return self._harness.is_running or self._terminal_signal is not None

    @property
    def queued_steering_messages(self) -> tuple[str, ...]:
        """Return queued steering text for UI display."""
        return tuple(message.content for message in self._harness.queued_messages.steering)

    @property
    def queued_follow_up_messages(self) -> tuple[str, ...]:
        """Return queued follow-up text for UI display."""
        return tuple(message.content for message in self._harness.queued_messages.follow_up)

    def cancel(self) -> None:
        """Request cancellation of the active agent or terminal command."""
        self._harness.cancel()
        if self._terminal_signal is not None:
            self._terminal_signal.cancel()

    def queue_update_event(self) -> QueueUpdateEvent:
        """Return the authoritative queue snapshot."""
        return self._harness.queue_update_event()

    def pop_latest_follow_up_message(self) -> str | None:
        """Remove and return the newest queued follow-up for editing."""
        message = self._harness.pop_latest_follow_up()
        return None if message is None else message.content

    async def set_model(self, model: str) -> str:
        """Switch models within the active provider and persist the choice."""
        return await self.set_model_choice(ModelChoice(self.provider_name, model))

    async def set_model_choice(self, choice: ModelChoice) -> str:
        """Switch provider/model atomically for future turns."""
        if self.is_running:
            raise RuntimeError("Cannot switch models during an active operation")
        provider = (
            None
            if self._provider_settings is None and self._runtime_provider_config is None
            else self._provider_config_for_choice(choice)
        )
        if provider is None and choice.provider_name != self.provider_name:
            raise ProviderConfigError("Provider settings are not available for this session")
        thinking = (
            _coerced_thinking_level(
                provider,
                model=choice.model,
                current=self._thinking_level,
            )
            if provider is not None
            else self._thinking_level
        )
        replacement = (
            self._create_runtime_provider(provider, choice.model, thinking)
            if provider is not None
            else None
        )
        if replacement is not None:
            self._owned_providers.append(replacement)
            self._harness.config.provider = replacement
            self._runtime_provider_config = provider
        self._provider_name = choice.provider_name
        self._harness.config.model = choice.model
        self._thinking_level = thinking
        await self._append_state_entry(ModelChangeEntry(model=choice.model))
        self._persist_default_model_choice()
        self._touch_session()
        return f"Current model: {choice.provider_name}:{choice.model}"

    async def set_provider(self, provider_name: str) -> str:
        """Select one provider and its configured default model."""
        if self._provider_settings is None:
            raise ProviderConfigError("Provider settings are not available for this session")
        provider = self._provider_settings.get_provider(provider_name)
        return await self.set_model_choice(ModelChoice(provider.name, provider.default_model))

    def toggle_scoped_model(self, choice: ModelChoice) -> tuple[ModelChoice, ...]:
        """Toggle one provider/model pair in the Ctrl+P cycle."""
        if self._provider_settings is None:
            raise ProviderConfigError("Provider settings are not available for this session")
        if choice not in set(self.available_model_choices):
            raise ProviderConfigError(
                f"Model is not available: {choice.provider_name}:{choice.model}"
            )
        self._provider_settings = toggle_scoped_model(
            self._provider_settings,
            provider_name=choice.provider_name,
            model=choice.model,
        )
        save_provider_settings(self._provider_settings, self._resource_paths.paths)
        return self.scoped_model_choices

    async def cycle_scoped_model(self, *, reverse: bool = False) -> ModelChoice:
        """Activate the next configured scoped model."""
        choices = self.scoped_model_choices
        if not choices:
            raise ProviderConfigError("No scoped models configured.")
        current = ModelChoice(self.provider_name, self.model)
        try:
            index = choices.index(current)
        except ValueError:
            index = 0 if reverse else -1
        selected = choices[(index + (-1 if reverse else 1)) % len(choices)]
        await self.set_model_choice(selected)
        return selected

    async def set_thinking_level(self, level: str) -> str:
        """Activate and persist a supported reasoning-effort level."""
        if self.is_running:
            raise RuntimeError("Cannot change thinking mode during an active operation")
        normalized = normalize_thinking_level(level)
        available = self.available_thinking_levels
        if not available:
            reason = self.thinking_unavailable_reason
            suffix = f": {reason}" if reason else ""
            raise ValueError(
                f"Thinking controls are unavailable for {self.provider_name}:{self.model}{suffix}"
            )
        if normalized not in available:
            raise ValueError(
                f"Thinking mode {normalized} is not available for "
                f"{self.provider_name}:{self.model}. Available modes: {', '.join(available)}"
            )
        if normalized == self._thinking_level:
            return f"Thinking mode: {normalized}"
        provider = self._active_provider_config()
        replacement = (
            self._create_runtime_provider(provider, self.model, normalized)
            if provider is not None
            else None
        )
        if replacement is not None:
            self._owned_providers.append(replacement)
            self._harness.config.provider = replacement
            self._runtime_provider_config = provider
        self._thinking_level = normalized
        await self._append_state_entry(ThinkingLevelChangeEntry(thinking_level=normalized))
        return f"Thinking mode: {normalized}"

    async def cycle_thinking_level(self) -> str:
        return await self.set_thinking_level(
            next_thinking_level(
                self._thinking_level,
                available=self.available_thinking_levels,
            )
        )

    def reload_provider_settings(self) -> None:
        """Reload provider catalog/settings after login or external edits."""
        if self._provider_settings is None:
            return
        settings = load_provider_settings(self._resource_paths.paths)
        provider = settings.get_provider(self.provider_name)
        previous = self._provider_settings
        self._provider_settings = settings
        try:
            self._thinking_level = _coerced_thinking_level(
                provider,
                model=self.model,
                current=self._thinking_level,
            )
            if not self._provider_is_usable(provider):
                self._runtime_provider_config = None
                return
            replacement = self._create_runtime_provider(
                provider,
                self.model,
                self._thinking_level,
            )
        except ProviderConfigError, RuntimeError:
            self._provider_settings = previous
            raise
        self._runtime_provider_config = provider
        if replacement is not None:
            self._owned_providers.append(replacement)
            self._harness.config.provider = replacement

    def handle_command(self, text: str) -> CommandResult:
        """Resolve a slash command while preserving prompt-template directives."""
        if expand_prompt_template_command(text, self._prompt_templates) is not None:
            return CommandResult(handled=False)
        return self._command_registry.execute(self, text)

    async def reload(self) -> CodingReloadSummary:
        """Reload local resources and persist a rebuilt next-turn system prompt."""
        if self.is_running:
            raise RuntimeError("Cannot reload resources during an active operation")

        before_context = self._context_files
        before_skills = self._skills
        before_templates = self._prompt_templates
        before_diagnostics = self._resource_diagnostics

        context_files, context_diagnostics = discover_project_context_with_diagnostics(
            self._resource_paths
        )
        skills, skill_diagnostics = load_skills_with_diagnostics(self._resource_paths)
        prompt_templates, prompt_diagnostics = load_prompt_templates_with_diagnostics(
            self._resource_paths
        )
        diagnostics = (*context_diagnostics, *skill_diagnostics, *prompt_diagnostics)

        self._context_files = context_files
        self._skills = skills
        self._prompt_templates = prompt_templates
        self._resource_diagnostics = diagnostics

        system_inputs_changed = before_context != context_files or before_skills != skills
        system_rebuilt = self._config.system is None and system_inputs_changed
        if system_rebuilt:
            self._harness.config.system = build_system_prompt(
                BuildSystemPromptOptions(
                    cwd=self.cwd,
                    current_date=date.today(),
                    tools=self.tools,
                    skills=skills,
                    context_files=context_files,
                )
            )
            await self._persist_system_snapshot()

        return CodingReloadSummary(
            skills=_reload_category(before_skills, skills),
            prompt_templates=_reload_category(before_templates, prompt_templates),
            context_files=_reload_category(before_context, context_files),
            diagnostics=_reload_category(before_diagnostics, diagnostics),
            system_prompt_rebuilt=system_rebuilt,
        )

    async def resume(self, session_id: str) -> str:
        """Replace this object with an indexed session while preserving provider ownership."""
        manager = self._config.session_manager
        if manager is None:
            raise ValueError("Session manager is not available")
        if self.is_running:
            raise RuntimeError("Cannot resume during an active operation")
        record = manager.get_session(session_id)
        if record is None:
            raise ValueError(f"Unknown session: {session_id}")
        provider_name = record.provider_name or self.provider_name
        runtime_provider = self._runtime_provider_config
        if self._provider_settings is not None:
            runtime_provider = self._provider_settings.get_provider(provider_name)
        replacement = await type(self).load(
            replace(
                self._config,
                model=record.model,
                cwd=record.cwd,
                storage=JsonlSessionStorage(record.path),
                session_id=record.id,
                provider=self._harness.config.provider,
                provider_name=provider_name,
                provider_settings=self._provider_settings,
                runtime_provider_config=runtime_provider,
                thinking_level=self._thinking_level,
            )
        )
        self._adopt(replacement)
        manager.touch_session(
            record.id,
            model=self.model,
            provider_name=self.provider_name,
        )
        return f"Resumed session: {record.id}"

    async def new_session(self) -> str:
        """Create and adopt a fresh indexed session for the same cwd."""
        manager = self._config.session_manager
        if manager is None:
            raise ValueError("Session manager is not available")
        if self.is_running:
            raise RuntimeError("Cannot create a session during an active operation")
        record = manager.create_session(
            cwd=self.cwd,
            model=self.model,
            provider_name=self.provider_name,
        )
        replacement = await type(self).load(
            replace(
                self._config,
                storage=JsonlSessionStorage(record.path),
                session_id=record.id,
                provider=self._harness.config.provider,
                provider_name=self.provider_name,
                provider_settings=self._provider_settings,
                runtime_provider_config=self._runtime_provider_config,
                thinking_level=self._thinking_level,
            )
        )
        self._adopt(replacement)
        return f"Started new session: {record.id}"

    async def rename(self, title: str) -> str:
        """Rename the current indexed session and append a metadata snapshot."""
        normalized = " ".join(title.split())
        if not normalized:
            raise ValueError("Session name cannot be empty")
        if len(normalized) > 120:
            raise ValueError("Session name must be at most 120 characters")
        manager = self._config.session_manager
        if manager is None or self.session_id is None:
            raise ValueError("Session manager is not available")
        updated = manager.touch_session(
            self.session_id,
            model=self.model,
            provider_name=self.provider_name,
            title=normalized,
        )
        if updated is None:
            raise ValueError(f"Unknown session: {self.session_id}")
        await self._persist_session_info(title=normalized)
        return f"Session renamed: {normalized}"

    async def export(
        self,
        destination: Path | None = None,
        *,
        format: str | None = None,
    ) -> Path:
        """Export the complete append-only tree as HTML or JSONL."""
        entries = await self._config.storage.read_all()
        inferred = destination.suffix if destination is not None else None
        export_format = normalize_export_format(format or inferred or "html")
        session_name = self.session_id or _storage_stem(self._config.storage) or "axis-session"
        output = destination or default_session_export_path(
            destination_dir=self.cwd,
            session_name=session_name,
            format=export_format,
        )
        if not output.is_absolute():
            output = self.cwd / output
        source = _storage_path(self._config.storage)
        return export_session_artifact(
            entries,
            output,
            title=self.session_title or "Axis Session Export",
            source=str(source) if source is not None else self.session_id,
            format=export_format,
        )

    async def tree_choices(self) -> tuple[SessionTreeChoice, ...]:
        """Return branchable entries in deterministic tree display order."""
        entries = await self._config.storage.read_all()
        indents = _tree_branch_indents(entries)
        return tuple(
            SessionTreeChoice(
                entry_id=entry.id,
                label=f"{'  ' * indents.get(entry.id, 0)}{_tree_entry_title(entry)}",
                active=entry.id == self._state.active_leaf_id,
                is_tool_call=(
                    isinstance(entry, MessageEntry)
                    and isinstance(entry.message, AssistantMessage)
                    and bool(entry.message.tool_calls)
                ),
            )
            for entry in _ordered_tree_entries(entries)
            if _is_branchable_tree_entry(entry)
        )

    async def branch_to_entry(
        self,
        entry_id: str,
        *,
        summarize: bool = False,
        custom_instructions: str | None = None,
    ) -> SessionTreeBranchResult:
        """Move the active leaf without deleting the abandoned physical branch."""
        if self.is_running:
            raise RuntimeError("Cannot branch during an active operation")
        entries = await self._config.storage.read_all()
        selected = next((entry for entry in entries if entry.id == entry_id), None)
        if selected is None:
            raise ValueError(f"Unknown session entry: {entry_id}")
        if not _is_branchable_tree_entry(selected):
            raise ValueError(f"Session entry cannot be branched from: {entry_id}")

        target_id: str | None = entry_id
        input_prefill: str | None = None
        summarized = False
        if summarize:
            abandoned = _messages_after_entry(entries, entry_id, self._state.active_leaf_id)
            if abandoned:
                summary = await self._summarize_messages(
                    abandoned,
                    purpose="abandoned branch",
                    instructions=custom_instructions,
                )
                summary_entry = BranchSummaryEntry(
                    parent_id=entry_id,
                    branch_root_id=entry_id,
                    summary=summary,
                )
                await self._config.storage.append(summary_entry)
                target_id = summary_entry.id
                summarized = True
        elif isinstance(selected, MessageEntry) and isinstance(selected.message, UserMessage):
            target_id = selected.parent_id
            input_prefill = selected.message.content

        await self._config.storage.append(LeafEntry(parent_id=target_id, entry_id=target_id))
        await self._restore_active_leaf(target_id)
        suffix = " with branch summary" if summarized else ""
        if input_prefill is not None:
            return SessionTreeBranchResult(
                message=f"Branched session before {entry_id}.",
                input_prefill=input_prefill,
            )
        return SessionTreeBranchResult(message=f"Branched session at {target_id}{suffix}.")

    async def compact(self, instructions: str | None = None) -> str:
        """Summarize all active context entries and replace them during replay."""
        if self.is_running:
            raise RuntimeError("Cannot compact during an active operation")
        if not self._state.messages:
            raise ValueError("No active context messages to compact")
        summary = await self._summarize_messages(
            self._state.messages,
            purpose="conversation context",
            instructions=instructions,
        )
        replaced_ids = list(self._state.context_entry_ids)
        entry = CompactionEntry(
            parent_id=self._current_entry_id,
            summary=summary,
            replaces_entry_ids=replaced_ids,
        )
        await self._config.storage.append(entry)
        await self._config.storage.append(LeafEntry(parent_id=entry.id, entry_id=entry.id))
        await self._restore_active_leaf(entry.id)
        return f"Compacted {len(replaced_ids)} context entries."

    async def run_terminal_command(
        self,
        command: str,
        *,
        add_to_context: bool,
    ) -> TerminalCommandResult:
        """Run a local shell command and optionally persist its output as user context."""
        normalized = command.strip()
        if not normalized:
            raise ValueError("Terminal command cannot be empty")
        if self._harness.is_running:
            raise RuntimeError("Cannot run a terminal command during an active agent run")
        if self._terminal_signal is not None:
            raise RuntimeError("A terminal command is already running")

        signal = _TerminalCancellationToken()
        self._terminal_signal = signal
        try:
            result = await create_bash_tool(cwd=self.cwd).execute(
                {"command": normalized},
                signal,
            )
        finally:
            if self._terminal_signal is signal:
                self._terminal_signal = None

        exit_code: int | None = None
        if result.data is not None:
            raw_exit_code = result.data.get("exit_code")
            if isinstance(raw_exit_code, int):
                exit_code = raw_exit_code

        if add_to_context:
            self._harness.append_message(
                UserMessage(content=_terminal_command_context_message(normalized, result.content))
            )
            await self._persist_new_messages()

        return TerminalCommandResult(
            command=normalized,
            output=result.content,
            exit_code=exit_code,
            ok=result.ok,
            added_to_context=add_to_context,
        )

    async def prompt(
        self,
        content: str,
        *,
        streaming_behavior: StreamingBehavior | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run a user prompt while durably following Harness messages."""
        expanded = self._expand_prompt(content)
        if self._terminal_signal is not None:
            raise RuntimeError("Cannot start an agent prompt during an active terminal command")
        if self._harness.is_running:
            if streaming_behavior == "steer":
                yield self._harness.steer(expanded)
                return
            if streaming_behavior == "follow_up":
                yield self._harness.follow_up(expanded)
                return
            raise RuntimeError(
                "CodingSession is already running; choose steering or follow-up queueing."
            )
        async for event in self._persisting_events(self._harness.prompt(expanded)):
            yield event

    async def continue_(self) -> AsyncIterator[AgentEvent]:
        """Continue restored context while durably following new messages."""
        async for event in self._persisting_events(self._harness.continue_()):
            yield event

    async def _persisting_events(
        self,
        events: AsyncIterator[AgentEvent],
    ) -> AsyncIterator[AgentEvent]:
        persistence_failed = False
        try:
            async for event in events:
                try:
                    await self._persist_new_messages()
                except BaseException:
                    persistence_failed = True
                    raise
                yield event
        finally:
            if not persistence_failed:
                await self._persist_new_messages()

    def _expand_prompt(self, content: str) -> str:
        template = expand_prompt_template_command(content, self._prompt_templates)
        if template is not None:
            return template
        skill = expand_skill_command(content, self._skills)
        return skill if skill is not None else content

    def _active_provider_config(self) -> OpenAICompatibleProviderConfig | None:
        if self._provider_settings is None:
            return None
        try:
            return self._provider_settings.get_provider(self.provider_name)
        except ProviderConfigError:
            return None

    def _usable_provider_configs(self) -> tuple[OpenAICompatibleProviderConfig, ...]:
        if self._provider_settings is None:
            return ()
        return tuple(
            provider
            for provider in self._provider_settings.providers
            if self._provider_is_usable(provider)
        )

    def _provider_is_usable(self, provider: OpenAICompatibleProviderConfig) -> bool:
        return provider_has_usable_credentials(
            provider,
            credential_reader=self._credential_store,
        )

    def _provider_config_for_choice(
        self,
        choice: ModelChoice,
    ) -> OpenAICompatibleProviderConfig:
        if self._provider_settings is None:
            if choice.provider_name != self.provider_name or choice.model != self.model:
                raise ProviderConfigError("Provider settings are not available for this session")
            if self._runtime_provider_config is None:
                raise ProviderConfigError("Runtime provider settings are not available")
            return self._runtime_provider_config
        provider = self._provider_settings.get_provider(choice.provider_name)
        if choice.model not in provider.models:
            raise ProviderConfigError(
                f"Model is not configured: {choice.provider_name}:{choice.model}"
            )
        if not self._provider_is_usable(provider):
            raise ProviderConfigError(
                f"Missing credentials for provider {provider.name}. Run /login {provider.name}."
            )
        return provider

    def _create_runtime_provider(
        self,
        provider: OpenAICompatibleProviderConfig,
        model: str,
        thinking_level: ThinkingLevel,
    ) -> ClosableModelProvider | None:
        if self._provider_settings is None and self._runtime_provider_config is None:
            return None
        try:
            return create_model_provider(
                provider,
                credential_store=self._credential_store,
                model=model,
                thinking_level=thinking_level,
            )
        except RuntimeError as exc:
            raise ProviderConfigError(str(exc)) from exc

    def _sync_thinking_level_to_active_model(self) -> None:
        provider = self._active_provider_config()
        if provider is None:
            return
        self._thinking_level = _coerced_thinking_level(
            provider,
            model=self.model,
            current=self._thinking_level,
        )

    def _refresh_runtime_provider(self) -> None:
        provider = self._active_provider_config() or self._runtime_provider_config
        if provider is None:
            return
        replacement = self._create_runtime_provider(
            provider,
            self.model,
            self._thinking_level,
        )
        if replacement is not None:
            self._owned_providers.append(replacement)
            self._harness.config.provider = replacement
            self._runtime_provider_config = provider

    def _persist_default_model_choice(self) -> None:
        if self._provider_settings is None:
            return
        self._provider_settings = set_default_provider_model(
            self._provider_settings,
            provider_name=self.provider_name,
            model=self.model,
        )
        save_provider_settings(self._provider_settings, self._resource_paths.paths)

    async def _append_state_entry(
        self,
        entry: ModelChangeEntry | ThinkingLevelChangeEntry,
    ) -> None:
        await self._ensure_initialized()
        persisted = entry.model_copy(update={"parent_id": self._current_entry_id})
        await self._config.storage.append(persisted)
        await self._config.storage.append(LeafEntry(parent_id=persisted.id, entry_id=persisted.id))
        self._current_entry_id = persisted.id
        self._state = await SessionState.from_storage(self._config.storage)

    async def _persist_new_messages(self) -> None:
        new_messages = self._harness.messages[self._persisted_message_count :]
        if not new_messages:
            return

        await self._ensure_initialized()
        for message in new_messages:
            entry = MessageEntry(
                parent_id=self._current_entry_id,
                message=message,
            )
            await self._config.storage.append(entry)
            leaf = LeafEntry(parent_id=entry.id, entry_id=entry.id)
            await self._config.storage.append(leaf)
            self._current_entry_id = entry.id
            self._persisted_message_count += 1

        self._state = await SessionState.from_storage(self._config.storage)
        self._touch_session()

    async def _summarize_messages(
        self,
        messages: tuple[AgentMessage, ...],
        *,
        purpose: str,
        instructions: str | None,
    ) -> str:
        transcript = "\n\n".join(f"{message.role}: {message.content}" for message in messages)
        custom = (
            f"\n\nAdditional instructions:\n{instructions.strip()}"
            if instructions and instructions.strip()
            else ""
        )
        prompt = (
            f"Summarize this {purpose} for a coding agent that will continue the work. "
            "Preserve decisions, files, commands, failures, unresolved tasks and user intent. "
            "Return only the summary.\n\n"
            f"<conversation>\n{transcript}\n</conversation>{custom}"
        )
        summarizer = AgentHarness(
            AgentHarnessConfig(
                provider=self._harness.config.provider,
                model=self.model,
                system="You create concise, factual coding-session summaries.",
                tools=[],
            )
        )
        async for _event in summarizer.prompt(prompt):
            pass
        summary = next(
            (
                message.content.strip()
                for message in reversed(summarizer.messages)
                if isinstance(message, AssistantMessage) and message.content.strip()
            ),
            "",
        )
        if not summary:
            raise RuntimeError("Session summarization returned an empty summary")
        return summary

    async def _restore_active_leaf(self, leaf_id: str | None) -> None:
        self._state = await SessionState.from_storage(
            self._config.storage,
            leaf_id=leaf_id,
        )
        self._harness.replace_messages(self._state.messages)
        self._harness.config.model = self._state.model or self._config.model
        self._thinking_level = normalize_thinking_level(
            self._state.thinking_level or self._config.thinking_level
        )
        self._sync_thinking_level_to_active_model()
        self._refresh_runtime_provider()
        self._current_entry_id = leaf_id
        self._persisted_message_count = len(self._state.messages)
        self._pending_initial_entries = ()
        self._touch_session()

    def _adopt(self, replacement: CodingSession) -> None:
        owned_providers = [*self._owned_providers, *replacement._owned_providers]
        self._config = replacement._config
        self._cwd = replacement._cwd
        self._state = replacement._state
        self._harness = replacement._harness
        self._resource_paths = replacement._resource_paths
        self._context_files = replacement._context_files
        self._skills = replacement._skills
        self._prompt_templates = replacement._prompt_templates
        self._resource_diagnostics = replacement._resource_diagnostics
        self._provider_name = replacement._provider_name
        self._provider_settings = replacement._provider_settings
        self._runtime_provider_config = replacement._runtime_provider_config
        self._thinking_level = replacement._thinking_level
        self._owned_providers = owned_providers
        self._credential_store = replacement._credential_store
        self._current_entry_id = replacement._current_entry_id
        self._persisted_message_count = replacement._persisted_message_count
        self._pending_initial_entries = replacement._pending_initial_entries
        self._terminal_signal = None
        self._command_registry = replacement._command_registry

    def _touch_session(self) -> CodingSessionRecord | None:
        manager = self._config.session_manager
        if manager is None or self.session_id is None:
            return None
        return manager.touch_session(
            self.session_id,
            model=self.model,
            provider_name=self.provider_name,
        )

    async def aclose(self) -> None:
        """Close provider clients created by model/thinking switches."""
        for provider in self._owned_providers:
            await provider.aclose()
        self._owned_providers.clear()

    async def _ensure_initialized(self) -> None:
        while self._pending_initial_entries:
            entry = self._pending_initial_entries[0]
            await self._config.storage.append(entry)
            self._pending_initial_entries = self._pending_initial_entries[1:]

    async def _persist_system_snapshot(self) -> None:
        await self._persist_session_info(title=self.session_title)

    async def _persist_session_info(self, *, title: str | None) -> None:
        await self._ensure_initialized()
        previous = self._state.session_info
        if previous is None:
            info = SessionInfoEntry(
                parent_id=self._current_entry_id,
                cwd=str(self.cwd),
                title=title,
                system=self.system,
            )
        else:
            info = SessionInfoEntry(
                parent_id=self._current_entry_id,
                created_at=previous.created_at,
                cwd=str(self.cwd),
                title=title,
                system=self.system,
            )
        await self._config.storage.append(info)
        await self._config.storage.append(LeafEntry(parent_id=info.id, entry_id=info.id))
        self._current_entry_id = info.id
        self._state = await SessionState.from_storage(self._config.storage)
        self._touch_session()


def _validate_restored_cwd(state: SessionState, cwd: Path) -> None:
    if state.session_info is None or state.session_info.cwd is None:
        return
    stored_cwd = Path(state.session_info.cwd).expanduser().resolve()
    if stored_cwd != cwd:
        raise CodingSessionError(f"Session cwd mismatch: stored {stored_cwd}, requested {cwd}")


def parse_terminal_command(text: str) -> TerminalCommandRequest | None:
    """Parse leading ``!``/``!!`` input-bar shell syntax."""
    stripped = text.strip()
    if stripped.startswith("!!"):
        command = stripped[2:].strip()
        return TerminalCommandRequest(command, False) if command else None
    if stripped.startswith("!"):
        command = stripped[1:].strip()
        return TerminalCommandRequest(command, True) if command else None
    return None


def _terminal_command_context_message(command: str, output: str) -> str:
    return (
        "Terminal command executed by the user.\n\n"
        f"Command:\n```bash\n{command}\n```\n\n"
        f"Output:\n```text\n{output}\n```"
    )


def _reload_category(
    before: tuple[object, ...], after: tuple[object, ...]
) -> ReloadCategorySummary:
    return ReloadCategorySummary(
        before=len(before),
        after=len(after),
        changed=before != after,
    )


def _coerced_thinking_level(
    provider: OpenAICompatibleProviderConfig,
    *,
    model: str,
    current: ThinkingLevel,
) -> ThinkingLevel:
    levels = provider_thinking_levels(provider, model=model)
    if not levels:
        return current
    if current in levels:
        return current
    return provider_default_thinking_level(provider, model=model) or levels[0]


def _storage_path(storage: SessionStorage) -> Path | None:
    path = getattr(storage, "path", None)
    return Path(path) if isinstance(path, str | Path) else None


def _storage_stem(storage: SessionStorage) -> str | None:
    path = _storage_path(storage)
    return path.stem if path is not None else None


def _is_branchable_tree_entry(entry: SessionEntry) -> bool:
    if isinstance(entry, CompactionEntry | BranchSummaryEntry):
        return True
    return isinstance(entry, MessageEntry) and isinstance(
        entry.message,
        UserMessage | AssistantMessage,
    )


def _ordered_tree_entries(entries: list[SessionEntry]) -> tuple[SessionEntry, ...]:
    children: dict[str | None, list[SessionEntry]] = {}
    for entry in entries:
        if not isinstance(entry, LeafEntry):
            children.setdefault(entry.parent_id, []).append(entry)
    ordered: list[SessionEntry] = []
    seen: set[str] = set()

    def append_descendants(parent_id: str | None) -> None:
        descendants = children.get(parent_id, [])
        for child in descendants:
            if child.id not in seen:
                ordered.append(child)
                seen.add(child.id)
        for child in descendants:
            append_descendants(child.id)

    append_descendants(None)
    for entry in entries:
        if not isinstance(entry, LeafEntry) and entry.id not in seen:
            ordered.append(entry)
            seen.add(entry.id)
            append_descendants(entry.id)
    return tuple(ordered)


def _tree_branch_indents(entries: list[SessionEntry]) -> dict[str, int]:
    children: dict[str | None, list[str]] = {}
    for entry in entries:
        if not isinstance(entry, LeafEntry):
            children.setdefault(entry.parent_id, []).append(entry.id)
    sibling_index = {
        child_id: index
        for child_ids in children.values()
        for index, child_id in enumerate(child_ids)
    }
    indents: dict[str, int] = {}
    for entry in _ordered_tree_entries(entries):
        parent_indent = indents.get(entry.parent_id, 0) if entry.parent_id else 0
        indents[entry.id] = parent_indent + (1 if sibling_index.get(entry.id, 0) else 0)
    return indents


def _tree_entry_title(entry: SessionEntry) -> str:
    if isinstance(entry, MessageEntry):
        message = entry.message
        if isinstance(message, AssistantMessage) and message.tool_calls and not message.content:
            return f"tool call: {', '.join(call.name for call in message.tool_calls)}"
        return f"{message.role}: {_short_preview(message.content)}"
    if isinstance(entry, CompactionEntry):
        return f"compaction summary: {_short_preview(entry.summary)}"
    if isinstance(entry, BranchSummaryEntry):
        return f"branch summary: {_short_preview(entry.summary)}"
    return entry.type


def _short_preview(text: str, *, limit: int = 72) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else f"{normalized[: limit - 1]}…"


def _messages_after_entry(
    entries: list[SessionEntry],
    entry_id: str,
    active_leaf_id: str | None,
) -> tuple[AgentMessage, ...]:
    if active_leaf_id is None:
        return ()
    active_path = path_to_entry(entries, active_leaf_id)
    selected_index = next(
        (index for index, entry in enumerate(active_path) if entry.id == entry_id),
        None,
    )
    if selected_index is None:
        return ()
    return tuple(
        entry.message
        for entry in active_path[selected_index + 1 :]
        if isinstance(entry, MessageEntry)
    )
