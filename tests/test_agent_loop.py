"""Tests for Axis's provider/tool agent loop."""

import asyncio
from collections.abc import AsyncIterator, Mapping

from axis_agent import (
    AgentEvent,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    ErrorEvent,
    QueueUpdateEvent,
    RetryEvent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolResultMessage,
    UserMessage,
)
from axis_agent.loop import run_agent_loop
from axis_agent.types import JSONValue
from axis_ai import (
    FakeProvider,
    ProviderErrorEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderRetryEvent,
    ProviderTextDeltaEvent,
    ProviderThinkingDeltaEvent,
)


async def collect_events(stream: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [event async for event in stream]


def test_text_loop_streams_events_and_appends_only_complete_message() -> None:
    messages = [UserMessage(content="Say hello")]
    assistant = AssistantMessage(content="Hello")
    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="fake-model"),
                ProviderTextDeltaEvent(delta="Hel"),
                ProviderTextDeltaEvent(delta="lo"),
                ProviderResponseEndEvent(message=assistant, finish_reason="stop"),
            ]
        ]
    )

    events = asyncio.run(
        collect_events(
            run_agent_loop(
                provider=provider,
                model="fake-model",
                system="You are Axis.",
                messages=messages,
                tools=[],
            )
        )
    )

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_delta",
        "message_delta",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    assert messages == [UserMessage(content="Say hello"), assistant]
    assert provider.calls[0][2] == [UserMessage(content="Say hello")]


def test_text_loop_forwards_thinking_and_retry_without_recording_them() -> None:
    messages = [UserMessage(content="Think briefly")]
    assistant = AssistantMessage(content="Done")
    provider = FakeProvider(
        [
            [
                ProviderRetryEvent(
                    attempt=1,
                    max_attempts=3,
                    delay_seconds=0.25,
                    message="temporary failure",
                    data={"status": 503},
                ),
                ProviderResponseStartEvent(model="fake-model"),
                ProviderThinkingDeltaEvent(delta="inspect"),
                ProviderResponseEndEvent(message=assistant),
            ]
        ]
    )

    events = asyncio.run(
        collect_events(
            run_agent_loop(
                provider=provider,
                model="fake-model",
                system="You are Axis.",
                messages=messages,
                tools=[],
            )
        )
    )

    retry = next(event for event in events if isinstance(event, RetryEvent))
    thinking = next(event for event in events if isinstance(event, ThinkingDeltaEvent))
    assert retry.data == {"status": 503}
    assert thinking.delta == "inspect"
    assert messages == [UserMessage(content="Think briefly"), assistant]


def test_text_loop_surfaces_provider_error_without_duplicate_missing_message_error() -> None:
    messages = [UserMessage(content="Hello")]
    provider = FakeProvider([[ProviderErrorEvent(message="provider failed", data={"status": 500})]])

    events = asyncio.run(
        collect_events(
            run_agent_loop(
                provider=provider,
                model="fake-model",
                system="You are Axis.",
                messages=messages,
                tools=[],
            )
        )
    )

    errors = [event for event in events if isinstance(event, ErrorEvent)]
    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "error",
        "turn_end",
        "agent_end",
    ]
    assert len(errors) == 1
    assert errors[0].message == "provider failed"
    assert errors[0].data == {"status": 500}
    assert messages == [UserMessage(content="Hello")]


def test_text_loop_reports_stream_that_never_produces_complete_message() -> None:
    messages = [UserMessage(content="Hello")]
    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="fake-model"),
                ProviderTextDeltaEvent(delta="unfinished"),
            ]
        ]
    )

    events = asyncio.run(
        collect_events(
            run_agent_loop(
                provider=provider,
                model="fake-model",
                system="You are Axis.",
                messages=messages,
                tools=[],
            )
        )
    )

    error = next(event for event in events if isinstance(event, ErrorEvent))
    assert error.message == "Provider stream ended without an assistant message"
    assert messages == [UserMessage(content="Hello")]


def test_loop_executes_tool_and_continues_until_assistant_stops() -> None:
    async def executor(
        arguments: Mapping[str, JSONValue],
        signal: object | None = None,
    ) -> AgentToolResult:
        del signal
        return AgentToolResult(
            tool_call_id="executor-returned-the-wrong-id",
            name="read",
            ok=True,
            content=f"contents of {arguments['path']}",
            data={"path": arguments["path"]},
            details={"source": "fake"},
        )

    tool = AgentTool(
        name="read",
        description="Read a file.",
        input_schema={"type": "object"},
        executor=executor,
    )
    tool_call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    first_assistant = AssistantMessage(content="I will read it.", tool_calls=[tool_call])
    final_assistant = AssistantMessage(content="README.md contains project documentation.")
    messages = [UserMessage(content="Read README.md")]
    provider = FakeProvider(
        [
            [ProviderResponseEndEvent(message=first_assistant, finish_reason="tool_calls")],
            [
                ProviderResponseStartEvent(model="fake-model"),
                ProviderTextDeltaEvent(delta=final_assistant.content),
                ProviderResponseEndEvent(message=final_assistant, finish_reason="stop"),
            ],
        ]
    )

    events = asyncio.run(
        collect_events(
            run_agent_loop(
                provider=provider,
                model="fake-model",
                system="You are Axis.",
                messages=messages,
                tools=[tool],
            )
        )
    )

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "turn_end",
        "turn_start",
        "message_start",
        "message_delta",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    assert messages == [
        UserMessage(content="Read README.md"),
        first_assistant,
        ToolResultMessage(
            tool_call_id="call-1",
            name="read",
            content="contents of README.md",
            data={"path": "README.md"},
            details={"source": "fake"},
        ),
        final_assistant,
    ]
    tool_end = next(event for event in events if isinstance(event, ToolExecutionEndEvent))
    assert tool_end.result.tool_call_id == "call-1"
    assert provider.calls[1][2] == messages[:3]


def test_loop_executes_multiple_tool_calls_in_order_before_next_provider_turn() -> None:
    execution_order: list[str] = []

    def make_tool(name: str) -> AgentTool:
        async def executor(
            arguments: Mapping[str, JSONValue],
            signal: object | None = None,
        ) -> AgentToolResult:
            del arguments, signal
            execution_order.append(name)
            return AgentToolResult(tool_call_id="ignored", name=name, ok=True, content=name)

        return AgentTool(
            name=name,
            description=f"Run {name}.",
            input_schema={"type": "object"},
            executor=executor,
        )

    first_call = ToolCall(id="call-1", name="first")
    second_call = ToolCall(id="call-2", name="second")
    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(tool_calls=[first_call, second_call])
                )
            ],
            [ProviderResponseEndEvent(message=AssistantMessage(content="Done"))],
        ]
    )
    messages = [UserMessage(content="Run both")]

    asyncio.run(
        collect_events(
            run_agent_loop(
                provider=provider,
                model="fake-model",
                system="You are Axis.",
                messages=messages,
                tools=[make_tool("first"), make_tool("second")],
            )
        )
    )

    assert execution_order == ["first", "second"]
    assert [
        message.tool_call_id for message in messages if isinstance(message, ToolResultMessage)
    ] == [
        "call-1",
        "call-2",
    ]
    assert len(provider.calls) == 2


def test_loop_records_unknown_tool_as_failed_result_and_continues() -> None:
    tool_call = ToolCall(id="call-1", name="missing")
    provider = FakeProvider(
        [
            [ProviderResponseEndEvent(message=AssistantMessage(tool_calls=[tool_call]))],
            [ProviderResponseEndEvent(message=AssistantMessage(content="Recovered"))],
        ]
    )
    messages = [UserMessage(content="Use a missing tool")]

    asyncio.run(
        collect_events(
            run_agent_loop(
                provider=provider,
                model="fake-model",
                system="You are Axis.",
                messages=messages,
                tools=[],
            )
        )
    )

    result = next(message for message in messages if isinstance(message, ToolResultMessage))
    assert result.ok is False
    assert result.tool_call_id == "call-1"
    assert result.error == "Unknown tool: missing"
    assert len(provider.calls) == 2


def test_loop_isolates_tool_exception_as_failed_result_and_continues() -> None:
    async def broken_executor(
        arguments: Mapping[str, JSONValue],
        signal: object | None = None,
    ) -> AgentToolResult:
        del arguments, signal
        raise RuntimeError("tool exploded")

    tool = AgentTool(
        name="broken",
        description="Fail predictably.",
        input_schema={"type": "object"},
        executor=broken_executor,
    )
    tool_call = ToolCall(id="call-1", name="broken")
    provider = FakeProvider(
        [
            [ProviderResponseEndEvent(message=AssistantMessage(tool_calls=[tool_call]))],
            [ProviderResponseEndEvent(message=AssistantMessage(content="Recovered"))],
        ]
    )
    messages = [UserMessage(content="Run broken tool")]

    asyncio.run(
        collect_events(
            run_agent_loop(
                provider=provider,
                model="fake-model",
                system="You are Axis.",
                messages=messages,
                tools=[tool],
            )
        )
    )

    result = next(message for message in messages if isinstance(message, ToolResultMessage))
    assert result.ok is False
    assert result.error == "tool exploded"
    assert result.content == "tool exploded"
    assert len(provider.calls) == 2


def test_loop_rejects_invalid_max_turns_before_calling_provider() -> None:
    provider = FakeProvider(
        [[ProviderResponseEndEvent(message=AssistantMessage(content="unused"))]]
    )

    events = asyncio.run(
        collect_events(
            run_agent_loop(
                provider=provider,
                model="fake-model",
                system="You are Axis.",
                messages=[],
                tools=[],
                max_turns=0,
            )
        )
    )

    error = next(event for event in events if isinstance(event, ErrorEvent))
    assert [event.type for event in events] == ["agent_start", "error", "agent_end"]
    assert error.recoverable is False
    assert error.message == "max_turns must be at least 1"
    assert provider.calls == []


def test_loop_stops_after_configured_max_turns() -> None:
    async def executor(
        arguments: Mapping[str, JSONValue],
        signal: object | None = None,
    ) -> AgentToolResult:
        del arguments, signal
        return AgentToolResult(tool_call_id="ignored", name="again", ok=True, content="ok")

    tool = AgentTool(
        name="again",
        description="Request another turn.",
        input_schema={"type": "object"},
        executor=executor,
    )
    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(tool_calls=[ToolCall(id="call-1", name="again")])
                )
            ],
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(tool_calls=[ToolCall(id="call-2", name="again")])
                )
            ],
        ]
    )

    events = asyncio.run(
        collect_events(
            run_agent_loop(
                provider=provider,
                model="fake-model",
                system="You are Axis.",
                messages=[UserMessage(content="Keep going")],
                tools=[tool],
                max_turns=2,
            )
        )
    )

    error = next(event for event in events if isinstance(event, ErrorEvent))
    assert error.recoverable is True
    assert error.message == "Agent loop stopped after reaching max_turns=2"
    assert len(provider.calls) == 2


def test_loop_cancels_unstarted_tool_calls_and_preserves_protocol_results() -> None:
    class CancellationSignal:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

        def is_cancelled(self) -> bool:
            return self.cancelled

    signal = CancellationSignal()
    execution_order: list[str] = []

    async def first_executor(
        arguments: Mapping[str, JSONValue],
        signal: object | None = None,
    ) -> AgentToolResult:
        del arguments, signal
        execution_order.append("first")
        cancellation_signal.cancel()
        return AgentToolResult(tool_call_id="call-1", name="first", ok=True, content="done")

    async def second_executor(
        arguments: Mapping[str, JSONValue],
        signal: object | None = None,
    ) -> AgentToolResult:
        del arguments, signal
        execution_order.append("second")
        return AgentToolResult(tool_call_id="call-2", name="second", ok=True, content="done")

    cancellation_signal = signal
    tools = [
        AgentTool("first", "Run first.", {"type": "object"}, first_executor),
        AgentTool("second", "Run second.", {"type": "object"}, second_executor),
    ]
    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(
                        tool_calls=[
                            ToolCall(id="call-1", name="first"),
                            ToolCall(id="call-2", name="second"),
                        ]
                    )
                )
            ]
        ]
    )
    messages = [UserMessage(content="Run both")]

    events = asyncio.run(
        collect_events(
            run_agent_loop(
                provider=provider,
                model="fake-model",
                system="You are Axis.",
                messages=messages,
                tools=tools,
                signal=signal,
            )
        )
    )

    results = [message for message in messages if isinstance(message, ToolResultMessage)]
    errors = [event for event in events if isinstance(event, ErrorEvent)]
    assert execution_order == ["first"]
    assert [(result.tool_call_id, result.ok) for result in results] == [
        ("call-1", True),
        ("call-2", False),
    ]
    assert results[1].error == "Tool call cancelled"
    assert len(errors) == 1
    assert errors[0].message == "Agent run cancelled"
    assert len(provider.calls) == 1


def test_loop_injects_steering_after_tool_batch() -> None:
    async def executor(
        arguments: Mapping[str, JSONValue],
        signal: object | None = None,
    ) -> AgentToolResult:
        del arguments, signal
        return AgentToolResult(tool_call_id="call-1", name="read", ok=True, content="contents")

    tool = AgentTool("read", "Read a file.", {"type": "object"}, executor)
    tool_call = ToolCall(id="call-1", name="read")
    first_assistant = AssistantMessage(tool_calls=[tool_call])
    final_assistant = AssistantMessage(content="Adjusted answer")
    provider = FakeProvider(
        [
            [ProviderResponseEndEvent(message=first_assistant)],
            [ProviderResponseEndEvent(message=final_assistant)],
        ]
    )
    messages = [UserMessage(content="Read")]
    steering_batches: list[tuple[UserMessage, ...]] = [(UserMessage(content="Focus on tests"),), ()]

    def get_steering() -> tuple[UserMessage, ...]:
        return steering_batches.pop(0) if steering_batches else ()

    events = asyncio.run(
        collect_events(
            run_agent_loop(
                provider=provider,
                model="fake-model",
                system="You are Axis.",
                messages=messages,
                tools=[tool],
                get_steering_messages=get_steering,
                get_queue_update=QueueUpdateEvent,
            )
        )
    )

    assert messages == [
        UserMessage(content="Read"),
        first_assistant,
        ToolResultMessage(tool_call_id="call-1", name="read", content="contents"),
        UserMessage(content="Focus on tests"),
        final_assistant,
    ]
    assert provider.calls[1][2] == messages[:4]
    assert "queue_update" in [event.type for event in events]


def test_loop_injects_follow_up_only_when_run_would_stop() -> None:
    first_assistant = AssistantMessage(content="First task done")
    final_assistant = AssistantMessage(content="Follow-up done")
    provider = FakeProvider(
        [
            [ProviderResponseEndEvent(message=first_assistant)],
            [ProviderResponseEndEvent(message=final_assistant)],
        ]
    )
    messages = [UserMessage(content="First task")]
    follow_up_batches: list[tuple[UserMessage, ...]] = [(UserMessage(content="Second task"),), ()]

    def get_follow_up() -> tuple[UserMessage, ...]:
        return follow_up_batches.pop(0) if follow_up_batches else ()

    events = asyncio.run(
        collect_events(
            run_agent_loop(
                provider=provider,
                model="fake-model",
                system="You are Axis.",
                messages=messages,
                tools=[],
                get_follow_up_messages=get_follow_up,
                get_queue_update=QueueUpdateEvent,
            )
        )
    )

    assert messages == [
        UserMessage(content="First task"),
        first_assistant,
        UserMessage(content="Second task"),
        final_assistant,
    ]
    assert provider.calls[1][2] == messages[:3]
    assert [event.type for event in events].count("queue_update") == 1
