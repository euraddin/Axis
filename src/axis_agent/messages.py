"""Provider-neutral transcript message models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from axis_agent.tools import ToolCall
from axis_agent.types import JSONValue


class UserMessage(BaseModel):
    """A message authored by the user."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user"] = "user"
    content: str


class AssistantMessage(BaseModel):
    """A complete assistant message with optional tool calls."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant"] = "assistant"
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    provider_data: dict[str, JSONValue] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )


class ToolResultMessage(BaseModel):
    """A transcript message containing a previous tool call's result."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["tool"] = "tool"
    tool_call_id: str
    name: str
    content: str
    ok: bool = True
    data: dict[str, JSONValue] | None = None
    details: dict[str, JSONValue] | None = None
    error: str | None = None


type AgentMessage = UserMessage | AssistantMessage | ToolResultMessage
