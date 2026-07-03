"""Provider-neutral agent loop."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from axis_agent.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    QueueUpdateEvent,
    RetryEvent,
    ThinkingDeltaEvent,
    ToolApprovalRequestEvent,
    ToolApprovalResolvedEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from axis_agent.messages import AgentMessage, AssistantMessage, ToolResultMessage
from axis_agent.tools import (
    AgentTool,
    AgentToolResult,
    ToolApprovalDecision,
    ToolApprovalHandler,
    ToolCall,
)
from axis_agent.types import JSONValue

if TYPE_CHECKING:
    from axis_ai.provider import CancellationToken, ModelProvider


async def run_agent_loop(
    *,
    provider: ModelProvider,
    model: str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    max_turns: int | None = None,
    signal: CancellationToken | None = None,
    get_steering_messages: Callable[[], Sequence[AgentMessage]] | None = None,
    get_follow_up_messages: Callable[[], Sequence[AgentMessage]] | None = None,
    get_queue_update: Callable[[], QueueUpdateEvent] | None = None,
    request_tool_approval: ToolApprovalHandler | None = None,
) -> AsyncIterator[AgentEvent]:
    """Run provider/tool turns until the assistant stops requesting tools.

    The caller owns ``messages``. This function appends complete assistant
    messages and tool results as the run progresses; streamed deltas remain
    ephemeral observations.
    """
    # Import provider event classes at execution time. ``axis_ai.events`` uses
    # agent message/tool contracts, so importing it while ``axis_agent`` itself
    # is initializing would make ``import axis_ai`` order-dependent.
    from axis_ai.events import (
        ProviderErrorEvent,
        ProviderResponseEndEvent,
        ProviderResponseStartEvent,
        ProviderRetryEvent,
        ProviderTextDeltaEvent,
        ProviderThinkingDeltaEvent,
    )

    yield AgentStartEvent()

    if max_turns is not None and max_turns < 1:
        yield ErrorEvent(message="max_turns must be at least 1", recoverable=False)
        yield AgentEndEvent()
        return

    tool_by_name = {tool.name: tool for tool in tools}
    turn = 1

    while max_turns is None or turn <= max_turns:
        if signal is not None and signal.is_cancelled():
            yield ErrorEvent(message="Agent run cancelled", recoverable=True)
            break

        yield TurnStartEvent(turn=turn)
        assistant_message: AssistantMessage | None = None
        saw_provider_error = False

        async for provider_event in provider.stream_response(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            signal=signal,
        ):
            if isinstance(provider_event, ProviderResponseStartEvent):
                yield MessageStartEvent()
            elif isinstance(provider_event, ProviderTextDeltaEvent):
                yield MessageDeltaEvent(delta=provider_event.delta)
            elif isinstance(provider_event, ProviderThinkingDeltaEvent):
                yield ThinkingDeltaEvent(delta=provider_event.delta)
            elif isinstance(provider_event, ProviderRetryEvent):
                yield RetryEvent(
                    attempt=provider_event.attempt,
                    max_attempts=provider_event.max_attempts,
                    delay_seconds=provider_event.delay_seconds,
                    message=provider_event.message,
                    data=provider_event.data,
                )
            elif isinstance(provider_event, ProviderResponseEndEvent):
                assistant_message = provider_event.message
                messages.append(assistant_message)
                yield MessageEndEvent(message=assistant_message)
            elif isinstance(provider_event, ProviderErrorEvent):
                saw_provider_error = True
                yield ErrorEvent(
                    message=provider_event.message,
                    recoverable=False,
                    data=provider_event.data,
                )

        if assistant_message is None:
            if signal is not None and signal.is_cancelled():
                yield ErrorEvent(message="Agent run cancelled", recoverable=True)
                yield TurnEndEvent(turn=turn)
                break
            yield TurnEndEvent(turn=turn)
            if not saw_provider_error:
                yield ErrorEvent(
                    message="Provider stream ended without an assistant message",
                    recoverable=False,
                )
            break

        if not assistant_message.tool_calls:
            yield TurnEndEvent(turn=turn)
            queue_events = _drain_queued_messages(
                messages,
                get_steering_messages,
                get_queue_update,
            )
            if queue_events:
                for queue_event in queue_events:
                    yield queue_event
                turn += 1
                continue
            queue_events = _drain_queued_messages(
                messages,
                get_follow_up_messages,
                get_queue_update,
            )
            if queue_events:
                for queue_event in queue_events:
                    yield queue_event
                turn += 1
                continue
            break

        async for tool_event in _execute_tool_calls(
            assistant_message.tool_calls,
            tool_by_name,
            messages,
            signal,
            request_tool_approval,
        ):
            yield tool_event

        yield TurnEndEvent(turn=turn)
        if signal is not None and signal.is_cancelled():
            break
        for queue_event in _drain_queued_messages(
            messages,
            get_steering_messages,
            get_queue_update,
        ):
            yield queue_event
        turn += 1
    else:
        yield ErrorEvent(
            message=f"Agent loop stopped after reaching max_turns={max_turns}",
            recoverable=True,
        )

    yield AgentEndEvent()


def _drain_queued_messages(
    messages: list[AgentMessage],
    get_messages: Callable[[], Sequence[AgentMessage]] | None,
    get_queue_update: Callable[[], QueueUpdateEvent] | None,
) -> tuple[AgentEvent, ...]:
    if get_messages is None:
        return ()
    queued_messages = tuple(get_messages())
    if not queued_messages:
        return ()

    messages.extend(queued_messages)
    events: list[AgentEvent] = []
    for message in queued_messages:
        events.append(MessageStartEvent(message_role=message.role))
        events.append(MessageEndEvent(message=message))
    if get_queue_update is not None:
        events.append(get_queue_update())
    return tuple(events)


def _auto_approved(tool: AgentTool, tool_call: ToolCall) -> bool:
    """Return True when *tool_call* is classified as safe enough to skip approval.

    Failures inside the classifier are treated as *not approved* so the
    call safely falls through to the normal approval path.
    """
    if tool.auto_approve_if is None:
        return False
    try:
        return bool(tool.auto_approve_if(tool_call.arguments))
    except Exception:
        return False


async def _execute_tool_calls(
    tool_calls: list[ToolCall],
    tool_by_name: Mapping[str, AgentTool],
    messages: list[AgentMessage],
    signal: CancellationToken | None,
    request_tool_approval: ToolApprovalHandler | None,
) -> AsyncIterator[AgentEvent]:
    for index, tool_call in enumerate(tool_calls):
        if signal is not None and signal.is_cancelled():
            for cancelled_tool_call in tool_calls[index:]:
                result = _cancelled_tool_result(cancelled_tool_call)
                messages.append(_tool_result_message(result))
                yield ToolExecutionEndEvent(result=result)
            yield ErrorEvent(message="Agent run cancelled", recoverable=True)
            return

        tool = tool_by_name.get(tool_call.name)
        if tool is None:
            yield ToolExecutionStartEvent(tool_call=tool_call)
            result = _unknown_tool_result(tool_call)
        else:
            if tool.requires_approval:
                if _auto_approved(tool, tool_call):
                    pass  # auto-approved: command is read-only
                else:
                    yield ToolApprovalRequestEvent(tool_call=tool_call)
                    decision, reason = await _resolve_tool_approval(
                        tool,
                        tool_call,
                        signal,
                        request_tool_approval,
                    )
                    yield ToolApprovalResolvedEvent(
                        tool_call_id=tool_call.id,
                        decision=decision,
                        reason=reason,
                    )
                    if decision == "deny":
                        result = _denied_tool_result(tool_call, reason)
                        messages.append(_tool_result_message(result))
                        yield ToolExecutionEndEvent(result=result)
                        continue
            yield ToolExecutionStartEvent(tool_call=tool_call)
            result = await _execute_tool(tool, tool_call, signal)

        messages.append(_tool_result_message(result))
        yield ToolExecutionEndEvent(result=result)

    if signal is not None and signal.is_cancelled():
        yield ErrorEvent(message="Agent run cancelled", recoverable=True)


async def _execute_tool(
    tool: AgentTool,
    tool_call: ToolCall,
    signal: CancellationToken | None,
) -> AgentToolResult:
    try:
        result = await tool.execute(tool_call.arguments, signal=signal)
    except Exception as exc:  # noqa: BLE001 - tools are an isolation boundary
        return AgentToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            ok=False,
            content=str(exc),
            error=str(exc),
        )

    if result.tool_call_id != tool_call.id:
        return result.model_copy(update={"tool_call_id": tool_call.id})
    return result


async def _resolve_tool_approval(
    tool: AgentTool,
    tool_call: ToolCall,
    signal: CancellationToken | None,
    request_tool_approval: ToolApprovalHandler | None,
) -> tuple[ToolApprovalDecision, str | None]:
    if signal is not None and signal.is_cancelled():
        return "deny", "Tool approval cancelled"
    if request_tool_approval is None:
        return "deny", "No tool approval handler is configured"
    try:
        decision = await request_tool_approval(tool, tool_call, signal)
    except Exception as exc:  # noqa: BLE001 - approval must fail closed
        return "deny", f"Tool approval failed: {exc}"
    if signal is not None and signal.is_cancelled():
        return "deny", "Tool approval cancelled"
    if decision not in {"allow_once", "allow_session", "deny"}:
        return "deny", f"Invalid tool approval decision: {decision}"
    if decision == "deny":
        return decision, "Tool call denied by user"
    return decision, None


def _unknown_tool_result(tool_call: ToolCall) -> AgentToolResult:
    message = f"Unknown tool: {tool_call.name}"
    return AgentToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        ok=False,
        content=message,
        error=message,
    )


def _cancelled_tool_result(tool_call: ToolCall) -> AgentToolResult:
    message = "Tool call cancelled"
    return AgentToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        ok=False,
        content=message,
        error=message,
    )


def _denied_tool_result(tool_call: ToolCall, reason: str | None) -> AgentToolResult:
    message = reason or "Tool call denied by user"
    return AgentToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        ok=False,
        content=message,
        error=message,
    )


def _tool_result_message(result: AgentToolResult) -> ToolResultMessage:
    data: dict[str, JSONValue] | None = result.data
    content = result.content
    if not result.ok and result.error and result.error not in content:
        content = f"{content}\n\nError: {result.error}"
    if data is not None and not content:
        content = str(data)

    return ToolResultMessage(
        tool_call_id=result.tool_call_id,
        name=result.name,
        content=content,
        ok=result.ok,
        data=result.data,
        details=result.details,
        error=result.error,
    )
