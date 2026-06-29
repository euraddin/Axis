"""Tests for the persistent Axis CodingSession composition layer."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from datetime import date
from pathlib import Path

import pytest

from axis_agent import (
    AgentEvent,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
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
from axis_ai import FakeProvider, ProviderResponseEndEvent
from axis_coding import (
    AxisPaths,
    AxisResourcePaths,
    CodingSession,
    CodingSessionConfig,
    CodingSessionError,
    ResourceError,
)


async def collect_events(stream: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [event async for event in stream]


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
    assert [tool.name for tool in session.tools] == ["read", "write", "edit", "bash"]
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
