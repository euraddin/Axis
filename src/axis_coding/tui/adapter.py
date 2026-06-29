"""Reduce portable agent events into framework-independent TUI state."""

from axis_agent import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AgentToolResult,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    QueueUpdateEvent,
    RetryEvent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from axis_coding.tui.state import (
    TuiMessageItem,
    TuiNoticeItem,
    TuiState,
    TuiToolItem,
    format_tool_call_summary,
)

_CANCELLATION_MESSAGE = "Agent run cancelled"


class TuiEventAdapter:
    """Apply AgentEvents to one mutable display state in event order."""

    def __init__(self, state: TuiState | None = None) -> None:
        self.state = state if state is not None else TuiState()

    def apply(self, event: AgentEvent) -> TuiState:
        """Apply one event and return the same state for convenient inspection."""
        if isinstance(event, AgentStartEvent):
            self.state.running = True
            self.state.cancelled = False
            self.state.current_turn = None
            self.state.error = None
        elif isinstance(event, AgentEndEvent):
            self._flush_live_buffers()
            self.state.running = False
            self.state.current_turn = None
        elif isinstance(event, TurnStartEvent):
            self.state.current_turn = event.turn
        elif isinstance(event, TurnEndEvent):
            if self.state.current_turn == event.turn:
                self.state.current_turn = None
        elif isinstance(event, MessageStartEvent):
            if event.message_role == "assistant":
                self._flush_live_buffers()
        elif isinstance(event, ThinkingDeltaEvent):
            self._flush_assistant_buffer()
            self.state.thinking_buffer += event.delta
        elif isinstance(event, MessageDeltaEvent):
            self._flush_thinking_buffer()
            self.state.assistant_buffer += event.delta
        elif isinstance(event, MessageEndEvent):
            self._apply_message_end(event)
        elif isinstance(event, ToolExecutionStartEvent):
            self._flush_live_buffers()
            self.state.items.append(
                TuiToolItem(
                    tool_call=event.tool_call,
                    summary=format_tool_call_summary(event.tool_call),
                )
            )
        elif isinstance(event, ToolExecutionUpdateEvent):
            tool = self._find_tool(event.tool_call_id)
            if tool is None:
                self.state.items.append(TuiNoticeItem(level="status", text=event.message))
            else:
                tool.updates.append(event.message)
        elif isinstance(event, ToolExecutionEndEvent):
            self._finish_tool(event.result)
        elif isinstance(event, RetryEvent):
            self.state.items.append(
                TuiNoticeItem(
                    level="retry",
                    text=event.message,
                    attempt=event.attempt,
                    max_attempts=event.max_attempts,
                )
            )
        elif isinstance(event, QueueUpdateEvent):
            self.state.queued_steering = event.steering
            self.state.queued_follow_up = event.follow_up
        elif isinstance(event, ErrorEvent):
            self._apply_error(event)
        return self.state

    def _apply_message_end(self, event: MessageEndEvent) -> None:
        message = event.message
        if message.role == "user":
            self._flush_live_buffers()
            self.state.items.append(TuiMessageItem(role="user", text=message.content))
            return
        if message.role == "tool":
            return

        self._flush_thinking_buffer()
        text = message.content or self.state.assistant_buffer
        self.state.assistant_buffer = ""
        if text:
            self.state.items.append(TuiMessageItem(role="assistant", text=text))

    def _finish_tool(self, result: AgentToolResult) -> None:
        tool = self._find_tool(result.tool_call_id)
        if tool is None:
            call = ToolCall(id=result.tool_call_id, name=result.name, arguments={})
            tool = TuiToolItem(tool_call=call, summary=format_tool_call_summary(call))
            self.state.items.append(tool)
        tool.result = result
        if result.error == "Tool call cancelled" or result.content == "Tool call cancelled":
            tool.status = "cancelled"
        else:
            tool.status = "succeeded" if result.ok else "failed"

    def _apply_error(self, event: ErrorEvent) -> None:
        self._flush_live_buffers()
        if event.recoverable and event.message == _CANCELLATION_MESSAGE:
            self.state.cancelled = True
            self.state.items.append(TuiNoticeItem(level="status", text="Agent run cancelled."))
            return

        self.state.error = event.message
        self.state.items.append(
            TuiNoticeItem(
                level="error",
                text=event.message,
                recoverable=event.recoverable,
            )
        )
        if not event.recoverable:
            self.state.running = False

    def _find_tool(self, tool_call_id: str) -> TuiToolItem | None:
        return next(
            (
                item
                for item in reversed(self.state.items)
                if isinstance(item, TuiToolItem) and item.tool_call.id == tool_call_id
            ),
            None,
        )

    def _flush_live_buffers(self) -> None:
        self._flush_thinking_buffer()
        self._flush_assistant_buffer()

    def _flush_thinking_buffer(self) -> None:
        if not self.state.thinking_buffer:
            return
        self.state.items.append(TuiMessageItem(role="thinking", text=self.state.thinking_buffer))
        self.state.thinking_buffer = ""

    def _flush_assistant_buffer(self) -> None:
        if not self.state.assistant_buffer:
            return
        self.state.items.append(TuiMessageItem(role="assistant", text=self.state.assistant_buffer))
        self.state.assistant_buffer = ""
