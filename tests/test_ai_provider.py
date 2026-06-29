"""Tests for Axis's provider boundary and deterministic fake."""

import asyncio
from collections.abc import AsyncIterator, Mapping

import pytest
from pydantic import ValidationError

from axis_agent import AgentTool, AgentToolResult, AssistantMessage, ToolCall, UserMessage
from axis_agent.types import JSONValue
from axis_ai import (
    FakeProvider,
    ProviderErrorEvent,
    ProviderEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderRetryEvent,
    ProviderTextDeltaEvent,
    ProviderThinkingDeltaEvent,
    ProviderToolCallEvent,
)


async def collect_events(stream: AsyncIterator[ProviderEvent]) -> list[ProviderEvent]:
    return [event async for event in stream]


def test_provider_events_form_a_stable_vocabulary() -> None:
    tool_call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    assistant = AssistantMessage(content="Done", tool_calls=[tool_call])
    events: list[ProviderEvent] = [
        ProviderResponseStartEvent(model="fake-model"),
        ProviderRetryEvent(
            attempt=1,
            max_attempts=3,
            delay_seconds=0.25,
            message="temporary failure",
        ),
        ProviderTextDeltaEvent(delta="Done"),
        ProviderThinkingDeltaEvent(delta="inspect"),
        ProviderToolCallEvent(tool_call=tool_call),
        ProviderResponseEndEvent(message=assistant, finish_reason="tool_calls"),
        ProviderErrorEvent(message="provider failed", data={"status": 500}),
    ]

    assert [event.type for event in events] == [
        "response_start",
        "retry",
        "text_delta",
        "thinking_delta",
        "tool_call",
        "response_end",
        "error",
    ]


def test_provider_events_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ProviderTextDeltaEvent(delta="hello", unexpected=True)  # type: ignore[call-arg]


def test_fake_provider_replays_one_script_per_call_and_records_snapshots() -> None:
    async def executor(
        arguments: Mapping[str, JSONValue],
        signal: object | None = None,
    ) -> AgentToolResult:
        del arguments, signal
        return AgentToolResult(tool_call_id="call-1", name="read", ok=True, content="")

    first_end = ProviderResponseEndEvent(message=AssistantMessage(content="first"))
    second_end = ProviderResponseEndEvent(message=AssistantMessage(content="second"))
    provider = FakeProvider([[first_end], [second_end]])
    messages = [UserMessage(content="hello")]
    tool = AgentTool(
        name="read",
        description="Read a file.",
        input_schema={"type": "object"},
        executor=executor,
    )
    tools = [tool]

    first = asyncio.run(
        collect_events(
            provider.stream_response(
                model="fake-model",
                system="You are Axis.",
                messages=messages,
                tools=tools,
            )
        )
    )
    messages.append(UserMessage(content="later mutation"))
    tools.clear()
    second = asyncio.run(
        collect_events(
            provider.stream_response(
                model="fake-model",
                system="You are Axis.",
                messages=messages,
                tools=tools,
            )
        )
    )

    assert first == [first_end]
    assert second == [second_end]
    assert provider.calls[0][0:2] == ("fake-model", "You are Axis.")
    assert provider.calls[0][2] == [UserMessage(content="hello")]
    assert provider.calls[0][3] == [tool]
    assert provider.calls[1][2] == messages
    assert provider.calls[1][3] == []


def test_fake_provider_returns_an_empty_stream_after_scripts_are_consumed() -> None:
    provider = FakeProvider([])

    events = asyncio.run(
        collect_events(
            provider.stream_response(
                model="fake-model",
                system="",
                messages=[],
                tools=[],
            )
        )
    )

    assert events == []


def test_fake_provider_observes_cancellation_between_events() -> None:
    class CancellationSignal:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

        def is_cancelled(self) -> bool:
            return self.cancelled

    signal = CancellationSignal()
    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="fake-model"),
                ProviderTextDeltaEvent(delta="should not arrive"),
            ]
        ]
    )

    async def consume_and_cancel() -> list[ProviderEvent]:
        observed: list[ProviderEvent] = []
        async for event in provider.stream_response(
            model="fake-model",
            system="",
            messages=[],
            tools=[],
            signal=signal,
        ):
            observed.append(event)
            signal.cancel()
        return observed

    events = asyncio.run(consume_and_cancel())

    assert [event.type for event in events] == ["response_start"]
