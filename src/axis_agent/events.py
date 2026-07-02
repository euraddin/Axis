"""Events emitted by Axis's portable agent layer."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from axis_agent.messages import AgentMessage
from axis_agent.tools import AgentToolResult, ToolApprovalDecision, ToolCall
from axis_agent.types import JSONValue


class AgentStartEvent(BaseModel):
    """The complete agent run has started."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(BaseModel):
    """The complete agent run has ended."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["agent_end"] = "agent_end"


class TurnStartEvent(BaseModel):
    """One provider/tool turn has started."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["turn_start"] = "turn_start"
    turn: int


class TurnEndEvent(BaseModel):
    """One provider/tool turn has ended."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["turn_end"] = "turn_end"
    turn: int


class RetryEvent(BaseModel):
    """A transient provider failure will be retried."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["retry"] = "retry"
    attempt: int
    max_attempts: int
    delay_seconds: float
    message: str
    data: dict[str, JSONValue] | None = None


class QueueUpdateEvent(BaseModel):
    """A snapshot of queued steering and follow-up prompts."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["queue_update"] = "queue_update"
    steering: tuple[str, ...] = ()
    follow_up: tuple[str, ...] = ()


class MessageStartEvent(BaseModel):
    """A streamed transcript message has started."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["message_start"] = "message_start"
    message_role: Literal["user", "assistant", "tool"] = "assistant"


class MessageDeltaEvent(BaseModel):
    """A visible assistant-text fragment has arrived."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["message_delta"] = "message_delta"
    delta: str


class ThinkingDeltaEvent(BaseModel):
    """A reasoning/thinking fragment has arrived for display."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str


class MessageEndEvent(BaseModel):
    """A complete transcript message is available."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["message_end"] = "message_end"
    message: AgentMessage


class ToolExecutionStartEvent(BaseModel):
    """Execution of a concrete tool call has started."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call: ToolCall


class ToolApprovalRequestEvent(BaseModel):
    """A protected tool call is waiting for an external decision."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_approval_request"] = "tool_approval_request"
    tool_call: ToolCall


class ToolApprovalResolvedEvent(BaseModel):
    """The external approval boundary resolved a protected tool call."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_approval_resolved"] = "tool_approval_resolved"
    tool_call_id: str
    decision: ToolApprovalDecision
    reason: str | None = None


class ToolExecutionUpdateEvent(BaseModel):
    """A running tool has emitted an intermediate update."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    message: str
    data: dict[str, JSONValue] | None = None


class ToolExecutionEndEvent(BaseModel):
    """Execution of a concrete tool call has ended."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_execution_end"] = "tool_execution_end"
    result: AgentToolResult


class ErrorEvent(BaseModel):
    """The agent encountered an observable error."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["error"] = "error"
    message: str
    recoverable: bool = False
    data: dict[str, JSONValue] | None = None


type AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | RetryEvent
    | QueueUpdateEvent
    | MessageStartEvent
    | MessageDeltaEvent
    | ThinkingDeltaEvent
    | MessageEndEvent
    | ToolApprovalRequestEvent
    | ToolApprovalResolvedEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
    | ErrorEvent
)
