"""Integration tests for task memory loading and reviewable session proposals."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

from axis_agent import (
    AgentEndEvent,
    AgentEvent,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    JsonlSessionStorage,
    MemoryContextEvent,
    MemoryProposalDecisionEntry,
    MemoryProposalEntry,
    MemoryProposalEvent,
    MessageEntry,
    SessionInfoEntry,
    ToolCall,
    UserMessage,
)
from axis_agent.types import JSONValue
from axis_ai import FakeProvider, ProviderErrorEvent, ProviderResponseEndEvent
from axis_coding import CodingSession, CodingSessionConfig


async def _collect(stream: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [event async for event in stream]


def _proposal_json(content: str = "- Auto Memory MVP implemented.") -> str:
    return (
        '{"proposals":[{'
        '"target_file":"progress.md",'
        '"operation":"append",'
        '"section_heading":null,'
        '"reason":"A reusable milestone was completed.",'
        f'"proposed_content":"{content}",'
        '"confidence":0.95,'
        '"requires_user_approval":true}]}'
    )


def test_successful_task_loads_dynamic_memory_and_persists_reviewable_proposal(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        storage = JsonlSessionStorage(tmp_path / "session.jsonl")
        provider = FakeProvider(
            [
                [ProviderResponseEndEvent(message=AssistantMessage(content="Implemented."))],
                [ProviderResponseEndEvent(message=AssistantMessage(content=_proposal_json()))],
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                system="Base system.",
                storage=storage,
                cwd=tmp_path,
                tools=[],
            )
        )
        session.initialize_memory()

        events = await _collect(session.prompt("实现 Memory Bank loader"))

        memory_context = next(event for event in events if isinstance(event, MemoryContextEvent))
        proposal_event = next(event for event in events if isinstance(event, MemoryProposalEvent))
        assert memory_context.task_type == "implementation"
        assert memory_context.loaded_files
        assert memory_context.estimated_tokens > 0
        assert proposal_event.status == "generated"
        assert isinstance(events[-1], AgentEndEvent)

        assert provider.calls[0][1].startswith("Base system.\n\n<project_memory")
        assert provider.calls[0][2] == [UserMessage(content="实现 Memory Bank loader")]
        assert provider.calls[1][3] == []
        generator_prompt = provider.calls[1][2][-1].content
        assert "Task evidence:" in generator_prompt
        assert "Implemented." in generator_prompt

        entries = await storage.read_all()
        info = next(entry for entry in entries if isinstance(entry, SessionInfoEntry))
        assert info.system == "Base system."
        assert "<project_memory" not in (info.system or "")
        assert [entry.message for entry in entries if isinstance(entry, MessageEntry)] == [
            UserMessage(content="实现 Memory Bank loader"),
            AssistantMessage(content="Implemented."),
        ]
        assert len(session.pending_memory_proposals) == 1
        proposal = session.pending_memory_proposals[0]
        assert proposal.id in proposal_event.proposal_ids
        assert proposal not in session.messages
        assert proposal.id in session.review_memory_proposals()

        restarted = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="different-default",
                storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
                cwd=tmp_path,
                tools=[],
            )
        )
        assert [item.id for item in restarted.pending_memory_proposals] == [proposal.id]
        applied = await restarted.apply_memory_proposal(proposal.id)
        assert "Applied memory proposal" in applied
        assert "Auto Memory MVP implemented" in (
            tmp_path / ".agent-memory" / "progress.md"
        ).read_text(encoding="utf-8")
        assert restarted.pending_memory_proposals == ()
        assert any(
            isinstance(entry, MemoryProposalDecisionEntry) and entry.decision == "applied"
            for entry in await storage.read_all()
        )

    asyncio.run(scenario())


def test_invalid_generator_json_is_corrected_once(tmp_path: Path) -> None:
    async def scenario() -> None:
        provider = FakeProvider(
            [
                [ProviderResponseEndEvent(message=AssistantMessage(content="Done."))],
                [ProviderResponseEndEvent(message=AssistantMessage(content="not json"))],
                [
                    ProviderResponseEndEvent(
                        message=AssistantMessage(content=_proposal_json("- Corrected proposal."))
                    )
                ],
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "corrected.jsonl"),
                cwd=tmp_path,
                tools=[],
            )
        )
        session.initialize_memory()

        events = await _collect(session.prompt("实现功能"))

        assert len(provider.calls) == 3
        assert "previous response was invalid" in provider.calls[2][2][-1].content
        assert any(
            isinstance(event, MemoryProposalEvent) and event.status == "generated"
            for event in events
        )
        assert len(session.pending_memory_proposals) == 1

    asyncio.run(scenario())


def test_generator_failure_warns_without_failing_main_task(tmp_path: Path) -> None:
    async def scenario() -> None:
        provider = FakeProvider(
            [
                [ProviderResponseEndEvent(message=AssistantMessage(content="Main task done."))],
                [ProviderErrorEvent(message="memory model unavailable")],
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "failure.jsonl"),
                cwd=tmp_path,
                tools=[],
            )
        )
        session.initialize_memory()

        events = await _collect(session.prompt("实现功能"))

        warnings = [
            event
            for event in events
            if isinstance(event, MemoryProposalEvent) and event.status == "warning"
        ]
        assert len(warnings) == 1
        assert "memory model unavailable" in warnings[0].message
        assert isinstance(events[-1], AgentEndEvent)
        assert session.messages[-1] == AssistantMessage(content="Main task done.")
        assert session.pending_memory_proposals == ()

    asyncio.run(scenario())


def test_generator_evidence_keeps_tool_metadata_but_omits_output_and_reasoning(
    tmp_path: Path,
) -> None:
    async def executor(
        arguments: Mapping[str, JSONValue],
        signal: object | None = None,
    ) -> AgentToolResult:
        del arguments, signal
        return AgentToolResult(
            tool_call_id="ignored",
            name="read",
            ok=True,
            content="FULL PRIVATE FILE BODY must not reach memory generation",
            data={"path": "src/app.py"},
        )

    async def scenario() -> None:
        call = ToolCall(id="call-1", name="read", arguments={"path": "src/app.py"})
        provider = FakeProvider(
            [
                [
                    ProviderResponseEndEvent(
                        message=AssistantMessage(
                            tool_calls=[call],
                            provider_data={"reasoning_content": "HIDDEN CHAIN OF THOUGHT"},
                        )
                    )
                ],
                [ProviderResponseEndEvent(message=AssistantMessage(content="Read completed."))],
                [ProviderResponseEndEvent(message=AssistantMessage(content='{"proposals":[]}'))],
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "evidence.jsonl"),
                cwd=tmp_path,
                tools=[AgentTool("read", "Read file", {"type": "object"}, executor)],
            )
        )
        session.initialize_memory()

        await _collect(session.prompt("Read src/app.py"))

        generator_prompt = provider.calls[2][2][-1].content
        assert '"read_files":["src/app.py"]' in generator_prompt
        assert '"tool":"read"' in generator_prompt
        assert '"ok":true' in generator_prompt
        assert "Read completed." in generator_prompt
        assert "FULL PRIVATE FILE BODY" not in generator_prompt
        assert "HIDDEN CHAIN OF THOUGHT" not in generator_prompt
        assert provider.calls[2][3] == []

    asyncio.run(scenario())


def test_memory_type_override_applies_to_one_top_level_task(tmp_path: Path) -> None:
    async def scenario() -> None:
        provider = FakeProvider(
            [
                [ProviderResponseEndEvent(message=AssistantMessage(content="First."))],
                [ProviderResponseEndEvent(message=AssistantMessage(content="Second."))],
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "types.jsonl"),
                cwd=tmp_path,
                tools=[],
                auto_memory_enabled=False,
            )
        )
        session.initialize_memory()
        session.set_next_memory_task_type("debug")

        first = await _collect(session.prompt("解释一下项目"))
        second = await _collect(session.prompt("解释一下项目"))

        assert next(
            event for event in first if isinstance(event, MemoryContextEvent)
        ).task_type == ("debug")
        assert next(
            event for event in second if isinstance(event, MemoryContextEvent)
        ).task_type == ("default")
        assert session.next_memory_task_type is None

    asyncio.run(scenario())


def test_pending_proposal_can_be_discarded_and_is_not_replayed_as_message(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        provider = FakeProvider(
            [
                [ProviderResponseEndEvent(message=AssistantMessage(content="Done."))],
                [ProviderResponseEndEvent(message=AssistantMessage(content=_proposal_json()))],
            ]
        )
        storage = JsonlSessionStorage(tmp_path / "discard.jsonl")
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=storage,
                cwd=tmp_path,
                tools=[],
            )
        )
        session.initialize_memory()
        await _collect(session.prompt("实现功能"))
        proposal = session.pending_memory_proposals[0]

        assert isinstance(proposal, MemoryProposalEntry)
        await session.discard_memory_proposal(proposal.id)
        assert session.pending_memory_proposals == ()
        assert session.messages == (
            UserMessage(content="实现功能"),
            AssistantMessage(content="Done."),
        )

    asyncio.run(scenario())
