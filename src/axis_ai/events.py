"""Provider-neutral streaming events emitted by model adapters."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from axis_agent.messages import AssistantMessage
from axis_agent.tools import ToolCall
from axis_agent.types import JSONValue


class ProviderResponseStartEvent(BaseModel):
    """The provider has started one model response."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["response_start"] = "response_start"
    model: str


class ProviderRetryEvent(BaseModel):
    """A transient provider failure will be retried."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["retry"] = "retry"
    attempt: int
    max_attempts: int
    delay_seconds: float
    message: str
    data: dict[str, JSONValue] | None = None


class ProviderTextDeltaEvent(BaseModel):
    """A visible text fragment from the provider."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text_delta"] = "text_delta"
    delta: str


class ProviderThinkingDeltaEvent(BaseModel):
    """A reasoning/thinking fragment from the provider."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str


class ProviderToolCallEvent(BaseModel):
    """A complete tool call assembled by the provider adapter."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_call"] = "tool_call"
    tool_call: ToolCall


class ProviderResponseEndEvent(BaseModel):
    """The provider has completed one model response."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["response_end"] = "response_end"
    message: AssistantMessage
    finish_reason: str | None = None


class ProviderErrorEvent(BaseModel):
    """A provider failure that can be surfaced by the agent layer."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["error"] = "error"
    message: str
    data: dict[str, JSONValue] | None = None


type ProviderEvent = (
    ProviderResponseStartEvent
    | ProviderRetryEvent
    | ProviderTextDeltaEvent
    | ProviderThinkingDeltaEvent
    | ProviderToolCallEvent
    | ProviderResponseEndEvent
    | ProviderErrorEvent
)
