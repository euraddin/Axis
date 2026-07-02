"""Reduce portable AgentEvents into Axis transcript state."""

from axis_agent import (
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
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from axis_coding.tui.state import TuiState

_CANCELLATION_MESSAGE = "Agent run cancelled"


class TuiEventAdapter:
    """Apply events in order without depending on Textual widgets."""

    def __init__(self, state: TuiState | None = None) -> None:
        self.state = state if state is not None else TuiState()

    def apply(self, event: AgentEvent) -> TuiState:
        """Apply one event and return the owned mutable state."""
        if isinstance(event, AgentStartEvent):
            self.state.running = True
            self.state.cancelled = False
            self.state.current_turn = None
            self.state.error = None
        elif isinstance(event, AgentEndEvent):
            self._flush_assistant_buffer()
            self.state.running = False
            self.state.current_turn = None
        elif isinstance(event, TurnStartEvent):
            self.state.current_turn = event.turn
        elif isinstance(event, TurnEndEvent):
            if self.state.current_turn == event.turn:
                self.state.current_turn = None
        elif isinstance(event, MessageStartEvent):
            if event.message_role == "assistant":
                self.state.assistant_buffer = ""
        elif isinstance(event, ThinkingDeltaEvent):
            self.state.add_thinking_delta(event.delta)
        elif isinstance(event, MessageDeltaEvent):
            self.state.assistant_buffer += event.delta
        elif isinstance(event, MessageEndEvent):
            self._apply_message_end(event)
        elif isinstance(event, ToolApprovalRequestEvent):
            self._flush_assistant_buffer()
            self.state.ensure_tool_call(event.tool_call)
        elif isinstance(event, ToolApprovalResolvedEvent):
            pass
        elif isinstance(event, ToolExecutionStartEvent):
            self._flush_assistant_buffer()
            self.state.ensure_tool_call(event.tool_call)
        elif isinstance(event, ToolExecutionUpdateEvent):
            self.state.add_item("tool", f"… {event.message}")
        elif isinstance(event, ToolExecutionEndEvent):
            self.state.record_tool_result(event.result)
        elif isinstance(event, RetryEvent):
            self.state.add_item("status", f"… {event.message}")
        elif isinstance(event, QueueUpdateEvent):
            self.state.update_queue(steering=event.steering, follow_up=event.follow_up)
        elif isinstance(event, ErrorEvent):
            self._apply_error(event)
        return self.state

    def _apply_message_end(self, event: MessageEndEvent) -> None:
        message = event.message
        if message.role == "user":
            self.state.add_user_message(message.content)
            return
        if message.role == "tool":
            return
        text = message.content or self.state.assistant_buffer
        if text:
            self.state.add_item("assistant", text)
        self.state.assistant_buffer = ""

    def _apply_error(self, event: ErrorEvent) -> None:
        self._flush_assistant_buffer()
        if event.recoverable and event.message == _CANCELLATION_MESSAGE:
            self.state.cancelled = True
            self.state.add_item("status", "Agent run cancelled.")
            return
        self.state.error = event.message
        self.state.add_item("error", f"Error: {event.message}")
        if not event.recoverable:
            self.state.running = False

    def _flush_assistant_buffer(self) -> None:
        if not self.state.assistant_buffer:
            return
        self.state.add_item("assistant", self.state.assistant_buffer)
        self.state.assistant_buffer = ""
