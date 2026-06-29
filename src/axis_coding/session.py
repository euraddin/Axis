"""Persistent coding-session composition around the portable AgentHarness."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from axis_agent import (
    AgentEvent,
    AgentHarness,
    AgentHarnessConfig,
    AgentMessage,
    AgentTool,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionEntry,
    SessionInfoEntry,
    SessionState,
    SessionStorage,
)
from axis_ai import ModelProvider
from axis_coding.context import (
    ProjectContextFile,
    discover_project_context_with_diagnostics,
)
from axis_coding.prompt_templates import (
    PromptTemplate,
    expand_prompt_template_command,
    load_prompt_templates_with_diagnostics,
)
from axis_coding.resources import (
    AxisResourcePaths,
    ResourceDiagnostic,
    resource_paths_with_cwd,
)
from axis_coding.skills import Skill, expand_skill_command, load_skills_with_diagnostics
from axis_coding.system_prompt import BuildSystemPromptOptions, build_system_prompt
from axis_coding.tools import create_coding_tools


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
        self._current_entry_id = state.active_leaf_id
        self._persisted_message_count = len(state.messages)
        self._pending_initial_entries = pending_initial_entries

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
            info = SessionInfoEntry(cwd=str(cwd), system=system)
            model = ModelChangeEntry(parent_id=info.id, model=config.model)
            initial_entries: list[SessionEntry] = [info, model]
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
        return cls(
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

    @property
    def cwd(self) -> Path:
        """Return the resolved working directory bound to coding tools."""
        return self._cwd

    @property
    def model(self) -> str:
        """Return the restored or configured active model."""
        return self._harness.config.model

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
    def messages(self) -> tuple[AgentMessage, ...]:
        """Return the current authoritative Harness transcript."""
        return self._harness.messages

    @property
    def state(self) -> SessionState:
        """Return the most recently replayed durable state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Return whether a prompt or continuation is active."""
        return self._harness.is_running

    def cancel(self) -> None:
        """Request cancellation of the active run."""
        self._harness.cancel()

    async def prompt(self, content: str) -> AsyncIterator[AgentEvent]:
        """Run a user prompt while durably following Harness messages."""
        expanded = self._expand_prompt(content)
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

    async def _ensure_initialized(self) -> None:
        while self._pending_initial_entries:
            entry = self._pending_initial_entries[0]
            await self._config.storage.append(entry)
            self._pending_initial_entries = self._pending_initial_entries[1:]


def _validate_restored_cwd(state: SessionState, cwd: Path) -> None:
    if state.session_info is None or state.session_info.cwd is None:
        return
    stored_cwd = Path(state.session_info.cwd).expanduser().resolve()
    if stored_cwd != cwd:
        raise CodingSessionError(f"Session cwd mismatch: stored {stored_cwd}, requested {cwd}")
