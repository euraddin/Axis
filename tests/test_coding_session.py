"""Tests for the persistent Axis CodingSession composition layer."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from axis_agent import (
    AgentEvent,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    BranchSummaryEntry,
    CompactionEntry,
    ContextCompactionEvent,
    ErrorEvent,
    JsonlSessionStorage,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionInfoEntry,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from axis_agent.types import JSONValue
from axis_ai import FakeProvider, ProviderErrorEvent, ProviderResponseEndEvent
from axis_coding import (
    AxisPaths,
    AxisResourcePaths,
    CodingSession,
    CodingSessionConfig,
    CodingSessionError,
    FileCredentialStore,
    ModelChoice,
    OpenAICompatibleProviderConfig,
    ProviderSettings,
    ReloadCategorySummary,
    ResourceError,
    SessionManager,
    SessionTreeBranchResult,
    load_provider_settings,
    parse_terminal_command,
)


async def collect_events(stream: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [event async for event in stream]


VALID_COMPACTION_SUMMARY = """## Goal
Keep working on Axis.

## Constraints & Preferences
Keep recent turns verbatim.

## Progress
### Done
Reviewed the older work.

### In Progress
Continue implementation.

### Blocked
None

## Key Decisions
Use partial compaction.

## Next Steps
Run tests.

## Critical Context
The session is append-only.

<read-files>
src/axis_coding/session.py
</read-files>

<modified-files>
None
</modified-files>"""


def test_new_coding_session_prepares_metadata_without_creating_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.jsonl"
    session = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="deepseek-v4-pro",
                storage=JsonlSessionStorage(path),
                cwd=tmp_path,
            )
        )
    )

    assert path.exists() is False
    assert session.cwd == tmp_path.resolve()
    assert session.model == "deepseek-v4-pro"
    assert session.messages == ()
    names = [tool.name for tool in session.tools]
    assert "read" in names and "write" in names and "bash" in names
    assert "git_status" in names and "web_search" in names
    assert "lint" in names and "task" in names
    assert len(names) >= 12
    assert session.state.session_info is not None
    assert session.state.session_info.cwd == str(tmp_path.resolve())


def test_prompt_persists_metadata_messages_leafs_and_restores_process_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.jsonl"
    storage = JsonlSessionStorage(path)
    assistant = AssistantMessage(content="Hello")
    session = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([[ProviderResponseEndEvent(message=assistant)]]),
                model="deepseek-v4-pro",
                storage=storage,
                cwd=tmp_path,
            )
        )
    )

    asyncio.run(collect_events(session.prompt("Hi")))
    entries = asyncio.run(storage.read_all())

    assert [entry.type for entry in entries] == [
        "session_info",
        "model_change",
        "message",
        "leaf",
        "message",
        "leaf",
    ]
    info, model, user, user_leaf, saved_assistant, assistant_leaf = entries
    assert isinstance(info, SessionInfoEntry)
    assert isinstance(model, ModelChangeEntry)
    assert isinstance(user, MessageEntry)
    assert isinstance(user_leaf, LeafEntry)
    assert isinstance(saved_assistant, MessageEntry)
    assert isinstance(assistant_leaf, LeafEntry)
    assert model.parent_id == info.id
    assert info.system == session.system
    assert user.parent_id == model.id
    assert user.message == UserMessage(content="Hi")
    assert user_leaf.parent_id == user.id
    assert user_leaf.entry_id == user.id
    assert saved_assistant.parent_id == user.id
    assert saved_assistant.message == assistant
    assert assistant_leaf.entry_id == saved_assistant.id

    restarted = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="different-default",
                storage=JsonlSessionStorage(path),
                cwd=tmp_path,
            )
        )
    )

    assert restarted.messages == (UserMessage(content="Hi"), assistant)
    assert restarted.model == "deepseek-v4-pro"
    assert restarted.system == session.system
    assert restarted.state.active_leaf_id == saved_assistant.id


def test_tool_round_trip_persists_exact_harness_transcript(tmp_path: Path) -> None:
    async def executor(
        arguments: Mapping[str, JSONValue],
        signal: object | None = None,
    ) -> AgentToolResult:
        del signal
        return AgentToolResult(
            tool_call_id="ignored",
            name="read",
            ok=True,
            content=f"contents of {arguments['path']}",
            data={"path": arguments["path"]},
        )

    tool = AgentTool("read", "Read.", {"type": "object"}, executor)
    tool_call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    tool_request = AssistantMessage(
        tool_calls=[tool_call],
        provider_data={"reasoning_content": "I should inspect README."},
    )
    final = AssistantMessage(content="Done")
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider(
                    [
                        [ProviderResponseEndEvent(message=tool_request)],
                        [ProviderResponseEndEvent(message=final)],
                    ]
                ),
                model="deepseek-v4-pro",
                storage=storage,
                cwd=tmp_path,
                tools=[tool],
            )
        )
    )

    asyncio.run(collect_events(session.prompt("Read README")))

    persisted_messages = [
        entry.message
        for entry in asyncio.run(storage.read_all())
        if isinstance(entry, MessageEntry)
    ]
    assert persisted_messages == list(session.messages)
    assert persisted_messages == [
        UserMessage(content="Read README"),
        tool_request,
        ToolResultMessage(
            tool_call_id="call-1",
            name="read",
            content="contents of README.md",
            data={"path": "README.md"},
        ),
        final,
    ]


def test_coding_session_rejects_restored_cwd_mismatch(tmp_path: Path) -> None:
    original_cwd = tmp_path / "original"
    requested_cwd = tmp_path / "requested"
    original_cwd.mkdir()
    requested_cwd.mkdir()
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")

    async def seed() -> None:
        info = SessionInfoEntry(cwd=str(original_cwd))
        model = ModelChangeEntry(parent_id=info.id, model="deepseek-v4-pro")
        await storage.append(info)
        await storage.append(model)

    asyncio.run(seed())

    with pytest.raises(CodingSessionError, match="Session cwd mismatch"):
        asyncio.run(
            CodingSession.load(
                CodingSessionConfig(
                    provider=FakeProvider([]),
                    model="deepseek-v4-pro",
                    storage=storage,
                    cwd=requested_cwd,
                )
            )
        )


def test_coding_session_rejects_missing_working_directory(tmp_path: Path) -> None:
    with pytest.raises(CodingSessionError, match="does not exist"):
        asyncio.run(
            CodingSession.load(
                CodingSessionConfig(
                    provider=FakeProvider([]),
                    model="deepseek-v4-pro",
                    storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
                    cwd=tmp_path / "missing",
                )
            )
        )


def test_explicit_empty_system_and_tools_are_preserved(tmp_path: Path) -> None:
    session = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="deepseek-v4-pro",
                storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
                cwd=tmp_path,
                system="",
                tools=[],
            )
        )
    )

    assert session.system == ""
    assert session.tools == ()


def test_explicit_empty_system_is_persisted_and_wins_on_restart(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    original = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider(
                    [[ProviderResponseEndEvent(message=AssistantMessage(content="Done"))]]
                ),
                model="deepseek-v4-pro",
                storage=JsonlSessionStorage(path),
                cwd=tmp_path,
                system="",
                tools=[],
            )
        )
    )
    asyncio.run(collect_events(original.prompt("No system")))

    restarted = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="different-default",
                storage=JsonlSessionStorage(path),
                cwd=tmp_path,
                system="replacement must not win",
                tools=[],
            )
        )
    )

    assert restarted.system == ""
    assert restarted.state.session_info is not None
    assert restarted.state.session_info.system == ""


def test_closing_event_stream_early_still_persists_user_prompt(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="deepseek-v4-pro",
                storage=storage,
                cwd=tmp_path,
            )
        )
    )

    async def consume_one_event() -> None:
        stream = session.prompt("Persist before UI stops")
        await anext(stream)
        await stream.aclose()

    asyncio.run(consume_one_event())

    messages = [
        entry.message
        for entry in asyncio.run(storage.read_all())
        if isinstance(entry, MessageEntry)
    ]
    assert messages == [UserMessage(content="Persist before UI stops")]
    assert session.is_running is False


def test_continue_after_restart_persists_new_assistant_message(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    first = AssistantMessage(content="First")
    original = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([[ProviderResponseEndEvent(message=first)]]),
                model="deepseek-v4-pro",
                storage=JsonlSessionStorage(path),
                cwd=tmp_path,
            )
        )
    )
    asyncio.run(collect_events(original.prompt("Start")))

    second = AssistantMessage(content="Continued")
    restarted = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([[ProviderResponseEndEvent(message=second)]]),
                model="different-default",
                storage=JsonlSessionStorage(path),
                cwd=tmp_path,
            )
        )
    )
    asyncio.run(collect_events(restarted.continue_()))

    assert restarted.messages == (UserMessage(content="Start"), first, second)
    persisted_messages = [
        entry.message
        for entry in asyncio.run(JsonlSessionStorage(path).read_all())
        if isinstance(entry, MessageEntry)
    ]
    assert persisted_messages == list(restarted.messages)


def test_coding_session_discovers_and_injects_hierarchical_agents_files(
    tmp_path: Path,
) -> None:
    user_axis = tmp_path / "user" / ".axis"
    user_agents = tmp_path / "user" / ".agents"
    project = tmp_path / "project"
    cwd = project / "src"
    cwd.mkdir(parents=True)
    user_axis.mkdir(parents=True)
    user_agents.mkdir(parents=True)
    (project / ".axis").mkdir()
    (project / ".agents").mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    files = [
        (user_axis / "AGENTS.md", "user axis rules"),
        (user_agents / "AGENTS.md", "user agents rules"),
        (project / "AGENTS.md", "project rules"),
        (cwd / "AGENTS.md", "nested rules"),
        (project / ".axis" / "AGENTS.md", "axis project rules"),
        (project / ".agents" / "AGENTS.md", "agents project rules"),
    ]
    for path, content in files:
        path.write_text(content, encoding="utf-8")

    provider = FakeProvider([[ProviderResponseEndEvent(message=AssistantMessage(content="Done"))]])
    session = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="deepseek-v4-pro",
                storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
                cwd=cwd,
                resource_paths=AxisResourcePaths(
                    paths=AxisPaths(home=user_axis, agents_home=user_agents)
                ),
            )
        )
    )

    assert [(item.path, item.content) for item in session.context_files] == files
    assert session.resource_paths.cwd == cwd.resolve()
    positions = [session.system.index(content) for _path, content in files]
    assert positions == sorted(positions)
    for path, _content in files:
        assert f'<project_instructions path="{path}">' in session.system
    asyncio.run(collect_events(session.prompt("Follow the rules")))
    assert provider.calls[0][1] == session.system


def test_explicit_system_overrides_discovered_context_prompt(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("project rules", encoding="utf-8")
    session = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="deepseek-v4-pro",
                storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
                cwd=tmp_path,
                system="custom system",
                resource_paths=AxisResourcePaths(
                    paths=AxisPaths(
                        home=tmp_path / "missing-axis-home",
                        agents_home=tmp_path / "missing-agents-home",
                    )
                ),
            )
        )
    )

    assert len(session.context_files) == 1
    assert session.system == "custom system"


def test_coding_session_loads_and_expands_skills_and_prompt_templates(
    tmp_path: Path,
) -> None:
    axis_home = tmp_path / "axis-home"
    skill_path = axis_home / "skills" / "testing" / "SKILL.md"
    template_path = axis_home / "prompts" / "review.md"
    skill_path.parent.mkdir(parents=True)
    template_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\ndescription: Test Python code\n---\n# Testing\nRun pytest.",
        encoding="utf-8",
    )
    template_path.write_text("Review {{ arguments }}.", encoding="utf-8")
    provider = FakeProvider(
        [
            [ProviderResponseEndEvent(message=AssistantMessage(content="Reviewed"))],
            [ProviderResponseEndEvent(message=AssistantMessage(content="Tested"))],
        ]
    )
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="deepseek-v4-pro",
                storage=storage,
                cwd=tmp_path,
                resource_paths=AxisResourcePaths(
                    paths=AxisPaths(
                        home=axis_home,
                        agents_home=tmp_path / "missing-agents-home",
                    )
                ),
            )
        )
    )

    asyncio.run(collect_events(session.prompt("/review src/app.py")))
    asyncio.run(collect_events(session.prompt("/skill:testing add parser tests")))

    assert [skill.name for skill in session.skills] == ["testing"]
    assert [template.name for template in session.prompt_templates] == ["review"]
    assert "You are Axis, the user's personal coding agent" in session.system
    assert "<available_skills>" in session.system
    assert "Test Python code" in session.system
    assert f"<location>{skill_path}</location>" in session.system
    assert "# Testing\nRun pytest." not in session.system
    assert "review.md" not in session.system
    assert provider.calls[0][2][-1] == UserMessage(content="Review src/app.py.")
    skill_prompt = provider.calls[1][2][-1]
    assert isinstance(skill_prompt, UserMessage)
    assert f'<skill name="testing" location="{skill_path}">' in skill_prompt.content
    assert "References are relative to" in skill_prompt.content
    assert skill_prompt.content.endswith("</skill>\n\nadd parser tests")
    persisted_messages = [
        entry.message
        for entry in asyncio.run(storage.read_all())
        if isinstance(entry, MessageEntry)
    ]
    assert persisted_messages == list(session.messages)


def test_restart_restores_exact_system_snapshot_across_date_and_resource_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FirstDate(date):
        @classmethod
        def today(cls) -> date:
            return cls(2026, 6, 29)

    class SecondDate(date):
        @classmethod
        def today(cls) -> date:
            return cls(2026, 6, 30)

    axis_home = tmp_path / "axis-home"
    skill_path = axis_home / "skills" / "testing.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\ndescription: Original skill\n---\nOriginal body",
        encoding="utf-8",
    )
    agents_path = tmp_path / "AGENTS.md"
    agents_path.write_text("Original project rules.", encoding="utf-8")
    paths = AxisResourcePaths(
        paths=AxisPaths(
            home=axis_home,
            agents_home=tmp_path / "missing-agents-home",
        )
    )
    session_path = tmp_path / "session.jsonl"
    monkeypatch.setattr("axis_coding.session.date", FirstDate)
    original = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider(
                    [[ProviderResponseEndEvent(message=AssistantMessage(content="Done"))]]
                ),
                model="deepseek-v4-pro",
                storage=JsonlSessionStorage(session_path),
                cwd=tmp_path,
                resource_paths=paths,
            )
        )
    )
    asyncio.run(collect_events(original.prompt("Create snapshot")))
    original_system = original.system

    skill_path.write_text(
        "---\ndescription: Changed skill\n---\nChanged body",
        encoding="utf-8",
    )
    agents_path.write_text("Changed project rules.", encoding="utf-8")
    monkeypatch.setattr("axis_coding.session.date", SecondDate)
    restarted = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="different-default",
                storage=JsonlSessionStorage(session_path),
                cwd=tmp_path,
                resource_paths=paths,
            )
        )
    )

    assert "Current date: 2026-06-29" in original_system
    assert "Original skill" in original_system
    assert "Original project rules." in original_system
    assert restarted.system == original_system
    assert "2026-06-30" not in restarted.system
    assert "Changed skill" not in restarted.system
    assert "Changed project rules." not in restarted.system


def test_coding_session_aggregates_resource_diagnostics(tmp_path: Path) -> None:
    axis_home = tmp_path / "axis-home"
    skills_dir = axis_home / "skills"
    prompts_dir = axis_home / "prompts"
    skills_dir.mkdir(parents=True)
    prompts_dir.mkdir(parents=True)
    (skills_dir / "bad skill.md").write_text("invalid", encoding="utf-8")
    (prompts_dir / "bad prompt.md").write_text("invalid", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_bytes(b"\xff")

    session = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="deepseek-v4-pro",
                storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
                cwd=tmp_path,
                resource_paths=AxisResourcePaths(
                    paths=AxisPaths(
                        home=axis_home,
                        agents_home=tmp_path / "missing-agents-home",
                    )
                ),
            )
        )
    )

    assert session.context_files == ()
    assert session.skills == ()
    assert session.prompt_templates == ()
    assert [diagnostic.kind for diagnostic in session.resource_diagnostics] == [
        "context",
        "skill",
        "prompt",
    ]


def test_unknown_skill_does_not_start_or_persist_a_session(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    provider = FakeProvider([])
    session = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="deepseek-v4-pro",
                storage=JsonlSessionStorage(path),
                cwd=tmp_path,
            )
        )
    )

    with pytest.raises(ResourceError, match="Unknown skill: missing"):
        asyncio.run(collect_events(session.prompt("/skill:missing")))

    assert provider.calls == []
    assert session.messages == ()
    assert session.is_running is False
    assert path.exists() is False


def test_parse_terminal_command_distinguishes_context_modes() -> None:
    add = parse_terminal_command("  !  pwd  ")
    local_only = parse_terminal_command("!! git status")

    assert add is not None
    assert add.command == "pwd"
    assert add.add_to_context is True
    assert local_only is not None
    assert local_only.command == "git status"
    assert local_only.add_to_context is False
    assert parse_terminal_command("!") is None
    assert parse_terminal_command("normal prompt") is None


def test_coding_session_exposes_harness_queues_during_an_active_run(tmp_path: Path) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class BlockingProvider:
            def stream_response(self, **kwargs: object):  # type: ignore[no-untyped-def]
                del kwargs

                async def events():  # type: ignore[no-untyped-def]
                    started.set()
                    await release.wait()
                    yield ProviderResponseEndEvent(
                        message=AssistantMessage(content="Initial task complete")
                    )

                return events()

        session = await CodingSession.load(
            CodingSessionConfig(
                provider=BlockingProvider(),
                model="deepseek-v4-pro",
                storage=JsonlSessionStorage(tmp_path / "queue-session.jsonl"),
                cwd=tmp_path,
                tools=[],
            )
        )
        active = asyncio.create_task(collect_events(session.prompt("Initial task")))
        await started.wait()

        follow_events = await collect_events(
            session.prompt("After this", streaming_behavior="follow_up")
        )
        steer_events = await collect_events(
            session.prompt("Adjust now", streaming_behavior="steer")
        )

        assert [event.type for event in follow_events] == ["queue_update"]
        assert [event.type for event in steer_events] == ["queue_update"]
        assert session.queued_steering_messages == ("Adjust now",)
        assert session.queued_follow_up_messages == ("After this",)
        assert session.queue_update_event().follow_up == ("After this",)
        assert session.pop_latest_follow_up_message() == "After this"
        assert session.queued_follow_up_messages == ()

        release.set()
        await active

    asyncio.run(scenario())


def test_terminal_command_can_persist_frozen_user_context(tmp_path: Path) -> None:
    async def scenario() -> None:
        storage = JsonlSessionStorage(tmp_path / "terminal-session.jsonl")
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="deepseek-v4-pro",
                storage=storage,
                cwd=tmp_path,
            )
        )

        result = await session.run_terminal_command(
            "printf 'axis-output'",
            add_to_context=True,
        )

        assert result.ok is True
        assert result.output == "axis-output"
        assert result.added_to_context is True
        assert session.messages == (
            UserMessage(
                content=(
                    "Terminal command executed by the user.\n\n"
                    "Command:\n```bash\nprintf 'axis-output'\n```\n\n"
                    "Output:\n```text\naxis-output\n```"
                )
            ),
        )
        persisted = [
            entry.message for entry in await storage.read_all() if isinstance(entry, MessageEntry)
        ]
        assert persisted == list(session.messages)

    asyncio.run(scenario())


def test_terminal_command_cancellation_terminates_process_group(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="deepseek-v4-pro",
                storage=JsonlSessionStorage(tmp_path / "cancel-session.jsonl"),
                cwd=tmp_path,
            )
        )
        task = asyncio.create_task(session.run_terminal_command("sleep 10", add_to_context=False))
        for _ in range(100):
            if session.is_running:
                break
            await asyncio.sleep(0.01)
        assert session.is_running is True

        session.cancel()
        result = await asyncio.wait_for(task, timeout=2)

        assert result.ok is False
        assert "Command cancelled" in result.output
        assert session.is_running is False

    asyncio.run(scenario())


def test_reload_refreshes_resources_and_persists_rebuilt_system_snapshot(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        axis_home = tmp_path / "axis-home"
        paths = AxisResourcePaths(
            paths=AxisPaths(
                home=axis_home,
                agents_home=tmp_path / "agents-home",
            )
        )
        storage = JsonlSessionStorage(tmp_path / "reload-session.jsonl")
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="deepseek-v4-pro",
                storage=storage,
                cwd=tmp_path,
                resource_paths=paths,
            )
        )
        original_system = session.system

        skills_dir = axis_home / "skills"
        prompts_dir = axis_home / "prompts"
        skills_dir.mkdir(parents=True)
        prompts_dir.mkdir(parents=True)
        (skills_dir / "review.md").write_text(
            "---\ndescription: Review changes\n---\nReview carefully.",
            encoding="utf-8",
        )
        (prompts_dir / "explain.md").write_text("Explain {{args}}.", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("Use deterministic tests.", encoding="utf-8")

        summary = await session.reload()

        assert summary.skills == ReloadCategorySummary(0, 1, True)
        assert summary.prompt_templates == ReloadCategorySummary(0, 1, True)
        assert summary.context_files == ReloadCategorySummary(0, 1, True)
        assert summary.system_prompt_rebuilt is True
        assert [skill.name for skill in session.skills] == ["review"]
        assert [template.name for template in session.prompt_templates] == ["explain"]
        assert "Review changes" in session.system
        assert "Use deterministic tests." in session.system
        assert session.system != original_system

        entries = await storage.read_all()
        assert [entry.type for entry in entries] == [
            "session_info",
            "model_change",
            "session_info",
            "leaf",
        ]
        restarted = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="different-default",
                storage=storage,
                cwd=tmp_path,
                resource_paths=paths,
            )
        )
        assert restarted.system == session.system
        assert restarted.model == "deepseek-v4-pro"

    asyncio.run(scenario())


def test_reload_never_rewrites_caller_owned_system_prompt(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="deepseek-v4-pro",
                storage=JsonlSessionStorage(tmp_path / "custom-system-session.jsonl"),
                cwd=tmp_path,
                system="",
                resource_paths=AxisResourcePaths(
                    paths=AxisPaths(
                        home=tmp_path / "axis-home",
                        agents_home=tmp_path / "agents-home",
                    )
                ),
            )
        )
        (tmp_path / "AGENTS.md").write_text("New project rules.", encoding="utf-8")

        summary = await session.reload()

        assert summary.context_files.changed is True
        assert summary.system_prompt_rebuilt is False
        assert session.system == ""

    asyncio.run(scenario())


def test_indexed_sessions_can_rename_resume_and_start_new(tmp_path: Path) -> None:
    async def scenario() -> None:
        paths = AxisPaths(home=tmp_path / ".axis", agents_home=tmp_path / ".agents")
        manager = SessionManager(paths)
        first_record = manager.create_session(cwd=tmp_path, model="fake", session_id="first")
        first = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider(
                    [[ProviderResponseEndEvent(message=AssistantMessage(content="First answer"))]]
                ),
                model="fake",
                storage=JsonlSessionStorage(first_record.path),
                cwd=tmp_path,
                session_id=first_record.id,
                session_manager=manager,
                resource_paths=AxisResourcePaths(paths=paths),
                tools=[],
            )
        )
        await collect_events(first.prompt("First prompt"))
        assert await first.rename("Primary work") == "Session renamed: Primary work"
        assert manager.get_session("first").title == "Primary work"  # type: ignore[union-attr]

        second_record = manager.create_session(cwd=tmp_path, model="fake", session_id="second")
        second = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider(
                    [[ProviderResponseEndEvent(message=AssistantMessage(content="Second answer"))]]
                ),
                model="fake",
                storage=JsonlSessionStorage(second_record.path),
                cwd=tmp_path,
                session_id=second_record.id,
                session_manager=manager,
                resource_paths=AxisResourcePaths(paths=paths),
                tools=[],
            )
        )
        await collect_events(second.prompt("Second prompt"))

        assert await first.resume("second") == "Resumed session: second"
        assert first.session_id == "second"
        assert first.messages == second.messages

        message = await first.new_session()
        assert message.startswith("Started new session: ")
        assert first.session_id not in {"first", "second"}
        assert first.messages == ()

    asyncio.run(scenario())


def test_session_tree_branching_preserves_history_and_prefills_user_message(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        storage = JsonlSessionStorage(tmp_path / "branch-session.jsonl")
        root = MessageEntry(id="root", message=UserMessage(content="Root"))
        answer = MessageEntry(
            id="answer",
            parent_id="root",
            message=AssistantMessage(content="Answer"),
        )
        followup = MessageEntry(
            id="followup",
            parent_id="answer",
            message=UserMessage(content="Try again"),
        )
        for entry in (root, answer, followup, LeafEntry(entry_id="followup")):
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

        choices = await session.tree_choices()
        result = await session.branch_to_entry("followup")

        assert [choice.entry_id for choice in choices] == ["root", "answer", "followup"]
        assert result == SessionTreeBranchResult(
            message="Branched session before followup.",
            input_prefill="Try again",
        )
        assert session.messages == (
            UserMessage(content="Root"),
            AssistantMessage(content="Answer"),
        )
        entries = await storage.read_all()
        assert [entry.id for entry in entries if isinstance(entry, MessageEntry)] == [
            "root",
            "answer",
            "followup",
        ]
        assert isinstance(entries[-1], LeafEntry)
        assert entries[-1].entry_id == "answer"

    asyncio.run(scenario())


def test_branch_summary_and_compaction_rebuild_provider_context(tmp_path: Path) -> None:
    async def scenario() -> None:
        storage = JsonlSessionStorage(tmp_path / "summary-session.jsonl")
        root = MessageEntry(id="root", message=UserMessage(content="Root"))
        answer = MessageEntry(
            id="answer",
            parent_id="root",
            message=AssistantMessage(content="Old direction"),
        )
        followup = MessageEntry(
            id="followup",
            parent_id="answer",
            message=UserMessage(content="More old work"),
        )
        for entry in (root, answer, followup, LeafEntry(entry_id="followup")):
            await storage.append(entry)
        provider = FakeProvider(
            [
                [
                    ProviderResponseEndEvent(
                        message=AssistantMessage(content="The abandoned branch went left.")
                    )
                ],
                [ProviderResponseEndEvent(message=AssistantMessage(content="New direction"))],
                [
                    ProviderResponseEndEvent(
                        message=AssistantMessage(content=VALID_COMPACTION_SUMMARY)
                    )
                ],
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=storage,
                cwd=tmp_path,
                tools=[],
                compact_retain_tokens=1,
            )
        )

        branch = await session.branch_to_entry("root", summarize=True)
        assert "with branch summary" in branch.message
        assert len(session.messages) == 1
        assert "The abandoned branch went left." in session.messages[0].content

        await collect_events(session.prompt("Continue here"))
        compacted = await session.compact("Keep implementation decisions.")
        assert compacted == "Compacted 1 context entries; retained 2 verbatim."
        assert session.messages[0] == UserMessage(
            content=f"Previous conversation summary:\n{VALID_COMPACTION_SUMMARY}"
        )
        assert session.messages[1:] == (
            UserMessage(content="Continue here"),
            AssistantMessage(content="New direction"),
        )
        entries = await storage.read_all()
        assert any(isinstance(entry, BranchSummaryEntry) for entry in entries)
        assert any(isinstance(entry, CompactionEntry) for entry in entries)
        assert "Additional instructions" in provider.calls[2][2][0].content
        assert "conversation-json" in provider.calls[2][2][0].content

    asyncio.run(scenario())


def test_auto_compaction_runs_before_provider_and_preserves_structured_tool_context(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        storage = JsonlSessionStorage(tmp_path / "auto-compact.jsonl")
        call = ToolCall(id="call-old", name="read", arguments={"path": "README.md"})
        entries = (
            MessageEntry(id="old-user", message=UserMessage(content="Inspect the project")),
            MessageEntry(
                id="old-assistant",
                parent_id="old-user",
                message=AssistantMessage(
                    tool_calls=[call],
                    provider_data={"reasoning_content": "secret reasoning"},
                ),
            ),
            MessageEntry(
                id="old-result",
                parent_id="old-assistant",
                message=ToolResultMessage(
                    tool_call_id=call.id,
                    name=call.name,
                    content="README contents",
                ),
            ),
            MessageEntry(
                id="old-final",
                parent_id="old-result",
                message=AssistantMessage(content="Inspection complete"),
            ),
            LeafEntry(entry_id="old-final"),
        )
        for entry in entries:
            await storage.append(entry)
        provider = FakeProvider(
            [
                [
                    ProviderResponseEndEvent(
                        message=AssistantMessage(content=VALID_COMPACTION_SUMMARY)
                    )
                ],
                [ProviderResponseEndEvent(message=AssistantMessage(content="Fresh answer"))],
                [
                    ProviderResponseEndEvent(
                        message=AssistantMessage(content=VALID_COMPACTION_SUMMARY)
                    )
                ],
                [ProviderResponseEndEvent(message=AssistantMessage(content="Second answer"))],
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=storage,
                cwd=tmp_path,
                system="",
                tools=[],
                auto_compact_token_threshold=1,
                compact_retain_tokens=1,
            )
        )

        events = await collect_events(session.prompt("Newest request"))

        compacted = next(event for event in events if isinstance(event, ContextCompactionEvent))
        assert compacted.compacted_entries == 4
        assert compacted.retained_entries == 1
        assert [event.type for event in events[:4]] == [
            "memory_context",
            "agent_start",
            "context_compaction",
            "turn_start",
        ]
        summary_prompt = provider.calls[0][2][0].content
        assert '"name":"read"' in summary_prompt
        assert '"path":"README.md"' in summary_prompt
        assert "README contents" in summary_prompt
        assert "secret reasoning" not in summary_prompt
        assert provider.calls[1][2] == [
            UserMessage(content=f"Previous conversation summary:\n{VALID_COMPACTION_SUMMARY}"),
            UserMessage(content="Newest request"),
        ]
        persisted = await storage.read_all()
        assert [entry.id for entry in persisted if isinstance(entry, MessageEntry)][:4] == [
            "old-user",
            "old-assistant",
            "old-result",
            "old-final",
        ]

        await collect_events(session.prompt("Second request"))
        assert "Previous conversation summary" in provider.calls[2][2][0].content
        persisted = await storage.read_all()
        assert len([entry for entry in persisted if isinstance(entry, CompactionEntry)]) == 2
        assert provider.calls[3][2][-1] == UserMessage(content="Second request")

    asyncio.run(scenario())


def test_auto_compaction_aborts_at_hard_window_when_no_older_turn_exists(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        provider_config = OpenAICompatibleProviderConfig(
            name="local",
            base_url="https://local.invalid/v1",
            api_key_env="AXIS_LOCAL_KEY",
            credential_name="local",
            models=("fake",),
            default_model="fake",
            context_windows={"fake": 10},
        )
        provider = FakeProvider([])
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "auto-hard-window.jsonl"),
                cwd=tmp_path,
                system="",
                tools=[],
                provider_name="local",
                provider_settings=ProviderSettings(
                    default_provider="local",
                    providers=(provider_config,),
                ),
                auto_compact_token_threshold=1,
            )
        )

        events = await collect_events(session.prompt("x" * 100))

        error = next(event for event in events if isinstance(event, ErrorEvent))
        assert "no complete older user turn" in error.message
        assert provider.calls == []
        assert session.messages == (UserMessage(content="x" * 100),)

    asyncio.run(scenario())


def test_auto_compaction_retries_invalid_format_once_then_succeeds(tmp_path: Path) -> None:
    async def scenario() -> None:
        storage = JsonlSessionStorage(tmp_path / "auto-retry.jsonl")
        old = MessageEntry(id="old", message=UserMessage(content="Old request"))
        await storage.append(old)
        await storage.append(LeafEntry(entry_id=old.id))
        provider = FakeProvider(
            [
                [ProviderResponseEndEvent(message=AssistantMessage(content="invalid"))],
                [
                    ProviderResponseEndEvent(
                        message=AssistantMessage(content=VALID_COMPACTION_SUMMARY)
                    )
                ],
                [ProviderResponseEndEvent(message=AssistantMessage(content="Done"))],
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=storage,
                cwd=tmp_path,
                system="",
                tools=[],
                auto_compact_token_threshold=1,
                compact_retain_tokens=1,
            )
        )

        events = await collect_events(session.prompt("New request"))

        assert any(isinstance(event, ContextCompactionEvent) for event in events)
        assert len(provider.calls) == 3
        assert "Rewrite your previous response" in provider.calls[1][2][-1].content
        assert provider.calls[2][2][-1] == UserMessage(content="New request")

    asyncio.run(scenario())


def test_auto_compaction_aborts_provider_request_after_two_invalid_summaries(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        storage = JsonlSessionStorage(tmp_path / "auto-invalid.jsonl")
        old = MessageEntry(id="old", message=UserMessage(content="Old request"))
        await storage.append(old)
        await storage.append(LeafEntry(entry_id=old.id))
        provider = FakeProvider(
            [
                [ProviderResponseEndEvent(message=AssistantMessage(content="invalid one"))],
                [ProviderResponseEndEvent(message=AssistantMessage(content="invalid two"))],
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=storage,
                cwd=tmp_path,
                system="",
                tools=[],
                auto_compact_token_threshold=1,
                compact_retain_tokens=1,
            )
        )

        events = await collect_events(session.prompt("New request"))

        error = next(event for event in events if isinstance(event, ErrorEvent))
        assert error.recoverable is True
        assert error.data == {"kind": "auto_compaction", "request_aborted": True}
        assert len(provider.calls) == 2
        assert not any(isinstance(entry, CompactionEntry) for entry in await storage.read_all())
        assert session.messages[-1] == UserMessage(content="New request")
        assert session.is_running is False

    asyncio.run(scenario())


def test_auto_compaction_does_not_retry_provider_error(tmp_path: Path) -> None:
    async def scenario() -> None:
        storage = JsonlSessionStorage(tmp_path / "auto-provider-error.jsonl")
        old = MessageEntry(id="old", message=UserMessage(content="Old request"))
        await storage.append(old)
        await storage.append(LeafEntry(entry_id=old.id))
        provider = FakeProvider([[ProviderErrorEvent(message="summary unavailable")]])
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=storage,
                cwd=tmp_path,
                system="",
                tools=[],
                auto_compact_token_threshold=1,
                compact_retain_tokens=1,
            )
        )

        events = await collect_events(session.prompt("New request"))

        error = next(event for event in events if isinstance(event, ErrorEvent))
        assert "summary unavailable" in error.message
        assert len(provider.calls) == 1
        assert not any(isinstance(entry, CompactionEntry) for entry in await storage.read_all())

    asyncio.run(scenario())


def test_cancel_stops_automatic_compaction_without_persisting_summary(tmp_path: Path) -> None:
    class BlockingSummaryProvider:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.calls = 0

        def stream_response(self, **kwargs: object) -> AsyncIterator[object]:
            signal = kwargs["signal"]
            self.calls += 1

            async def iterator() -> AsyncIterator[object]:
                self.started.set()
                while not signal.is_cancelled():  # type: ignore[union-attr]
                    await asyncio.sleep(0)
                if False:
                    yield ProviderResponseEndEvent(message=AssistantMessage(content="unused"))

            return iterator()

    async def scenario() -> None:
        storage = JsonlSessionStorage(tmp_path / "auto-cancel.jsonl")
        old = MessageEntry(id="old", message=UserMessage(content="Old request"))
        await storage.append(old)
        await storage.append(LeafEntry(entry_id=old.id))
        provider = BlockingSummaryProvider()
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,  # type: ignore[arg-type]
                model="fake",
                storage=storage,
                cwd=tmp_path,
                system="",
                tools=[],
                auto_compact_token_threshold=1,
                compact_retain_tokens=1,
            )
        )

        running = asyncio.create_task(collect_events(session.prompt("New request")))
        await provider.started.wait()
        session.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

        assert provider.calls == 1
        assert not any(isinstance(entry, CompactionEntry) for entry in await storage.read_all())
        assert session.is_running is False

    asyncio.run(scenario())


def test_auto_compaction_checks_again_before_tool_continuation(tmp_path: Path) -> None:
    async def executor(
        arguments: Mapping[str, JSONValue],
        signal: object | None = None,
    ) -> AgentToolResult:
        del arguments, signal
        return AgentToolResult(
            tool_call_id="wrong",
            name="read",
            ok=True,
            content="x" * 4_000,
        )

    async def scenario() -> None:
        storage = JsonlSessionStorage(tmp_path / "auto-tool-turn.jsonl")
        old_user = MessageEntry(id="old-user", message=UserMessage(content="Old request"))
        old_answer = MessageEntry(
            id="old-answer",
            parent_id=old_user.id,
            message=AssistantMessage(content="Old answer"),
        )
        for entry in (old_user, old_answer, LeafEntry(entry_id=old_answer.id)):
            await storage.append(entry)
        tool = AgentTool(
            name="read",
            description="Read a file.",
            input_schema={"type": "object"},
            executor=executor,
        )
        call = ToolCall(id="call-new", name="read", arguments={"path": "large.txt"})
        provider = FakeProvider(
            [
                [
                    ProviderResponseEndEvent(
                        message=AssistantMessage(tool_calls=[call]),
                        finish_reason="tool_calls",
                    )
                ],
                [
                    ProviderResponseEndEvent(
                        message=AssistantMessage(content=VALID_COMPACTION_SUMMARY)
                    )
                ],
                [ProviderResponseEndEvent(message=AssistantMessage(content="Finished"))],
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=storage,
                cwd=tmp_path,
                system="",
                tools=[tool],
                auto_compact_token_threshold=500,
                compact_retain_tokens=1,
            )
        )

        events = await collect_events(session.prompt("Read the large file"))

        compacted = [event for event in events if isinstance(event, ContextCompactionEvent)]
        assert len(compacted) == 1
        assert len(provider.calls) == 3
        assert "Old request" in provider.calls[1][2][0].content
        continued_messages = provider.calls[2][2]
        assert continued_messages[0] == UserMessage(
            content=f"Previous conversation summary:\n{VALID_COMPACTION_SUMMARY}"
        )
        assert any(isinstance(message, ToolResultMessage) for message in continued_messages)

    asyncio.run(scenario())


@pytest.mark.parametrize("streaming_behavior", ["steer", "follow_up"])
def test_auto_compaction_checks_queued_user_messages_before_provider(
    tmp_path: Path,
    streaming_behavior: str,
) -> None:
    class QueueAwareProvider:
        def __init__(self) -> None:
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()
            self.calls: list[tuple[str, str, list[object], list[object]]] = []

        def stream_response(
            self,
            *,
            model: str,
            system: str,
            messages: list[object],
            tools: list[object],
            signal: object | None = None,
        ) -> AsyncIterator[object]:
            del signal
            call_index = len(self.calls)
            self.calls.append((model, system, list(messages), list(tools)))

            async def iterator() -> AsyncIterator[object]:
                if call_index == 0:
                    self.first_started.set()
                    await self.release_first.wait()
                    yield ProviderResponseEndEvent(message=AssistantMessage(content="First answer"))
                elif call_index == 1:
                    yield ProviderResponseEndEvent(
                        message=AssistantMessage(content=VALID_COMPACTION_SUMMARY)
                    )
                else:
                    yield ProviderResponseEndEvent(message=AssistantMessage(content="Final answer"))

            return iterator()

    async def scenario() -> None:
        storage = JsonlSessionStorage(tmp_path / f"auto-queued-{streaming_behavior}.jsonl")
        old_user = MessageEntry(id="old-user", message=UserMessage(content="Old request"))
        old_answer = MessageEntry(
            id="old-answer",
            parent_id=old_user.id,
            message=AssistantMessage(content="Old answer"),
        )
        for entry in (old_user, old_answer, LeafEntry(entry_id=old_answer.id)):
            await storage.append(entry)
        provider = QueueAwareProvider()
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,  # type: ignore[arg-type]
                model="fake",
                storage=storage,
                cwd=tmp_path,
                system="",
                tools=[],
                auto_compact_token_threshold=500,
                compact_retain_tokens=1,
            )
        )

        running = asyncio.create_task(collect_events(session.prompt("Initial request")))
        await provider.first_started.wait()
        queued = await collect_events(
            session.prompt(
                "queued " + "x" * 4_000,
                streaming_behavior=streaming_behavior,  # type: ignore[arg-type]
            )
        )
        provider.release_first.set()
        events = await running

        assert [event.type for event in queued] == ["queue_update"]
        assert len([event for event in events if isinstance(event, ContextCompactionEvent)]) == 1
        assert len(provider.calls) == 3
        assert isinstance(provider.calls[2][2][0], UserMessage)
        assert provider.calls[2][2][-1] == UserMessage(content="queued " + "x" * 4_000)

    asyncio.run(scenario())


def test_session_persists_model_and_thinking_changes(tmp_path: Path) -> None:
    async def scenario() -> None:
        paths = AxisPaths(home=tmp_path / ".axis", agents_home=tmp_path / ".agents")
        settings = load_provider_settings(paths)
        original = settings.get_provider("deepseek")
        alternate_model = "deepseek-v4-fast"
        provider_config = replace(
            original,
            models=(*original.models, alternate_model),
            thinking_models=(*original.thinking_models, alternate_model),
        )
        settings = ProviderSettings(
            default_provider="deepseek",
            providers=(provider_config,),
        )
        FileCredentialStore(paths.home / "credentials.json").set("deepseek", "test-key")
        storage = JsonlSessionStorage(tmp_path / "model-thinking.jsonl")
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model=original.default_model,
                storage=storage,
                cwd=tmp_path,
                tools=[],
                resource_paths=AxisResourcePaths(paths=paths),
                provider_name="deepseek",
                provider_settings=settings,
                thinking_level="xhigh",
            )
        )

        assert await session.set_model(alternate_model) == (
            f"Current model: deepseek:{alternate_model}"
        )
        assert await session.set_thinking_level("high") == "Thinking mode: high"
        assert session.model == alternate_model
        assert session.thinking_level == "high"
        entries = await storage.read_all()
        assert any(
            isinstance(entry, ModelChangeEntry) and entry.model == alternate_model
            for entry in entries
        )
        assert any(
            entry.type == "thinking_level_change" and entry.thinking_level == "high"
            for entry in entries
        )

        restored = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model=original.default_model,
                storage=storage,
                cwd=tmp_path,
                tools=[],
                resource_paths=AxisResourcePaths(paths=paths),
                provider_name="deepseek",
                provider_settings=settings,
                thinking_level="xhigh",
            )
        )
        assert restored.model == alternate_model
        assert restored.thinking_level == "high"
        await session.aclose()

    asyncio.run(scenario())


def test_session_scoped_model_cycle_and_manager_track_provider(tmp_path: Path) -> None:
    async def scenario() -> None:
        paths = AxisPaths(home=tmp_path / ".axis", agents_home=tmp_path / ".agents")
        manager = SessionManager(paths)
        settings = load_provider_settings(paths)
        original = settings.get_provider("deepseek")
        alternate_model = "deepseek-v4-fast"
        provider_config = replace(
            original,
            models=(*original.models, alternate_model),
            thinking_models=(*original.thinking_models, alternate_model),
        )
        settings = ProviderSettings(
            default_provider="deepseek",
            providers=(provider_config,),
        )
        FileCredentialStore(paths.home / "credentials.json").set("deepseek", "test-key")
        record = manager.create_session(
            cwd=tmp_path,
            model=original.default_model,
            provider_name="deepseek",
            session_id="scoped",
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model=original.default_model,
                storage=JsonlSessionStorage(record.path),
                cwd=tmp_path,
                tools=[],
                resource_paths=AxisResourcePaths(paths=paths),
                session_id=record.id,
                session_manager=manager,
                provider_name="deepseek",
                provider_settings=settings,
            )
        )

        session.toggle_scoped_model(ModelChoice("deepseek", original.default_model))
        session.toggle_scoped_model(ModelChoice("deepseek", alternate_model))
        selected = await session.cycle_scoped_model()

        assert selected == ModelChoice("deepseek", alternate_model)
        assert session.model == alternate_model
        updated = manager.get_session(record.id)
        assert updated is not None
        assert updated.provider_name == "deepseek"
        assert updated.model == alternate_model
        await session.aclose()

    asyncio.run(scenario())
