"""Tests for the stateful reusable AgentHarness."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from axis_agent import (
    AgentEvent,
    AssistantMessage,
    ErrorEvent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from axis_agent.harness import AgentHarness, AgentHarnessConfig
from axis_ai import (
    FakeProvider,
    ProviderEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
)


async def collect_events(stream: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [event async for event in stream]


def test_prompt_owns_transcript_and_emits_user_message_events() -> None:
    assistant = AssistantMessage(content="Hello")
    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="fake-model"),
                ProviderTextDeltaEvent(delta="Hello"),
                ProviderResponseEndEvent(message=assistant),
            ]
        ]
    )
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=provider,
            model="fake-model",
            system="You are Axis.",
        )
    )

    events = asyncio.run(collect_events(harness.prompt("Say hello")))

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_delta",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    assert events[2].message_role == "user"  # type: ignore[union-attr]
    assert harness.messages == (UserMessage(content="Say hello"), assistant)
    assert provider.calls[0][2] == [UserMessage(content="Say hello")]
    assert harness.is_running is False


def test_continue_runs_existing_transcript_without_new_user_message() -> None:
    existing = UserMessage(content="Continue this")
    assistant = AssistantMessage(content="Continued")
    provider = FakeProvider([[ProviderResponseEndEvent(message=assistant)]])
    harness = AgentHarness(
        AgentHarnessConfig(provider=provider, model="fake-model", system="You are Axis."),
        messages=[existing],
    )

    events = asyncio.run(collect_events(harness.continue_()))

    assert harness.messages == (existing, assistant)
    assert [event.type for event in events].count("message_end") == 1


def test_messages_are_snapshots_with_explicit_restore_operations() -> None:
    provider = FakeProvider([])
    harness = AgentHarness(
        AgentHarnessConfig(provider=provider, model="fake-model", system="You are Axis."),
        messages=[UserMessage(content="original")],
    )
    snapshot = harness.messages

    harness.append_message(AssistantMessage(content="later"))

    assert snapshot == (UserMessage(content="original"),)
    assert harness.messages == (
        UserMessage(content="original"),
        AssistantMessage(content="later"),
    )

    harness.replace_messages([UserMessage(content="restored")])

    assert harness.messages == (UserMessage(content="restored"),)


def test_subscribed_listener_receives_events_and_can_unsubscribe() -> None:
    provider = FakeProvider(
        [
            [ProviderResponseEndEvent(message=AssistantMessage(content="first"))],
            [ProviderResponseEndEvent(message=AssistantMessage(content="second"))],
        ]
    )
    harness = AgentHarness(
        AgentHarnessConfig(provider=provider, model="fake-model", system="You are Axis.")
    )
    observed: list[AgentEvent] = []
    unsubscribe = harness.subscribe(observed.append)

    first_events = asyncio.run(collect_events(harness.prompt("hello")))

    assert observed == first_events

    unsubscribe()
    asyncio.run(collect_events(harness.continue_()))

    assert observed == first_events


def test_cancel_stops_active_run_and_overlapping_prompt_is_rejected() -> None:
    class BlockingProvider:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        def stream_response(self, **kwargs: object) -> AsyncIterator[ProviderEvent]:
            signal = kwargs["signal"]

            async def iterator() -> AsyncIterator[ProviderEvent]:
                self.started.set()
                yield ProviderResponseStartEvent(model="blocking")
                while not signal.is_cancelled():  # type: ignore[union-attr]
                    await asyncio.sleep(0)

            return iterator()

    async def scenario() -> tuple[list[AgentEvent], AgentHarness]:
        provider = BlockingProvider()
        harness = AgentHarness(
            AgentHarnessConfig(provider=provider, model="blocking", system="You are Axis.")
        )
        task = asyncio.create_task(collect_events(harness.prompt("first")))
        await provider.started.wait()

        assert harness.is_running is True
        with pytest.raises(RuntimeError, match="already running"):
            harness.prompt("second")

        harness.cancel()
        return await task, harness

    events, harness = asyncio.run(scenario())

    errors = [event for event in events if isinstance(event, ErrorEvent)]
    assert len(errors) == 1
    assert errors[0].message == "Agent run cancelled"
    assert errors[0].recoverable is True
    assert harness.messages == (UserMessage(content="first"),)
    assert harness.is_running is False


def test_queue_management_returns_immutable_snapshots() -> None:
    harness = AgentHarness(
        AgentHarnessConfig(provider=FakeProvider([]), model="fake-model", system="You are Axis.")
    )

    steering_update = harness.steer("adjust direction")
    harness.follow_up("first follow-up")
    latest_update = harness.follow_up("latest follow-up")

    assert steering_update.steering == ("adjust direction",)
    assert latest_update.follow_up == ("first follow-up", "latest follow-up")
    assert harness.pending_message_count == 3
    assert harness.has_queued_messages() is True

    popped = harness.pop_latest_follow_up()
    cleared = harness.clear_queues()

    assert popped == UserMessage(content="latest follow-up")
    assert cleared.steering == (UserMessage(content="adjust direction"),)
    assert cleared.follow_up == (UserMessage(content="first follow-up"),)
    assert harness.queued_messages.count == 0
    assert harness.has_queued_messages() is False


def test_harness_drains_follow_ups_one_at_a_time_by_default() -> None:
    assistants = [
        AssistantMessage(content="first"),
        AssistantMessage(content="second"),
        AssistantMessage(content="third"),
    ]
    provider = FakeProvider(
        [[ProviderResponseEndEvent(message=assistant)] for assistant in assistants]
    )
    harness = AgentHarness(
        AgentHarnessConfig(provider=provider, model="fake-model", system="You are Axis.")
    )
    harness.follow_up("follow-up one")
    harness.follow_up("follow-up two")

    events = asyncio.run(collect_events(harness.prompt("initial")))

    assert harness.messages == (
        UserMessage(content="initial"),
        assistants[0],
        UserMessage(content="follow-up one"),
        assistants[1],
        UserMessage(content="follow-up two"),
        assistants[2],
    )
    assert len(provider.calls) == 3
    assert [event.type for event in events].count("queue_update") == 2
    assert harness.pending_message_count == 0


def test_harness_can_drain_all_queued_messages_together() -> None:
    first = AssistantMessage(content="first")
    final = AssistantMessage(content="final")
    provider = FakeProvider(
        [
            [ProviderResponseEndEvent(message=first)],
            [ProviderResponseEndEvent(message=final)],
        ]
    )
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=provider,
            model="fake-model",
            system="You are Axis.",
            queue_mode="all",
        )
    )
    harness.follow_up("follow-up one")
    harness.follow_up("follow-up two")

    events = asyncio.run(collect_events(harness.prompt("initial")))

    assert harness.messages == (
        UserMessage(content="initial"),
        first,
        UserMessage(content="follow-up one"),
        UserMessage(content="follow-up two"),
        final,
    )
    assert len(provider.calls) == 2
    assert [event.type for event in events].count("queue_update") == 1


def test_prompt_repairs_interrupted_tool_calls_before_appending_new_user_message() -> None:
    first_call = ToolCall(id="call-1", name="read")
    interrupted_call = ToolCall(id="call-2", name="write")
    assistant_with_tools = AssistantMessage(tool_calls=[first_call, interrupted_call])
    existing_result = ToolResultMessage(
        tool_call_id="call-1",
        name="read",
        content="done",
    )
    provider = FakeProvider(
        [[ProviderResponseEndEvent(message=AssistantMessage(content="Recovered"))]]
    )
    harness = AgentHarness(
        AgentHarnessConfig(provider=provider, model="fake-model", system="You are Axis."),
        messages=[
            UserMessage(content="Modify files"),
            assistant_with_tools,
            existing_result,
        ],
    )

    asyncio.run(collect_events(harness.prompt("Continue safely")))

    repaired = ToolResultMessage(
        tool_call_id="call-2",
        name="write",
        content="Tool call interrupted by user",
        ok=False,
        error="Tool call interrupted by user",
    )
    assert provider.calls[0][2] == [
        UserMessage(content="Modify files"),
        assistant_with_tools,
        existing_result,
        repaired,
        UserMessage(content="Continue safely"),
    ]
    assert harness.messages[:5] == tuple(provider.calls[0][2])
