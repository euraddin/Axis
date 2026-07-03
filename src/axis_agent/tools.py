"""Provider-neutral tool definitions and execution results."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from axis_agent.types import JSONValue


class ToolCancellationToken(Protocol):
    """Minimal cancellation interface accepted by tools."""

    def is_cancelled(self) -> bool:
        """Return whether tool execution should stop."""
        ...


class ToolExecutor(Protocol):
    """Async callable used to execute a tool."""

    def __call__(
        self,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> Awaitable[AgentToolResult]:
        """Execute a tool with JSON-like arguments."""
        ...


type ToolApprovalDecision = Literal["allow_once", "allow_session", "deny"]


class ToolApprovalHandler(Protocol):
    """Async decision boundary invoked before a protected tool executes."""

    def __call__(
        self,
        tool: AgentTool,
        tool_call: ToolCall,
        signal: ToolCancellationToken | None = None,
    ) -> Awaitable[ToolApprovalDecision]:
        """Return the user's decision for one concrete tool call."""
        ...


class ToolCall(BaseModel):
    """A model request to execute a named tool."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, JSONValue] = Field(default_factory=dict)


class AgentToolResult(BaseModel):
    """Structured output returned by tool execution."""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    name: str
    ok: bool
    content: str
    data: dict[str, JSONValue] | None = None
    details: dict[str, JSONValue] | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentTool:
    """A tool definition that can be injected into an agent loop."""

    name: str
    description: str
    input_schema: Mapping[str, JSONValue]
    executor: ToolExecutor
    requires_approval: bool = False
    auto_approve_if: Callable[[Mapping[str, JSONValue]], bool] | None = field(
        default=None, compare=False, hash=False
    )
    prompt_snippet: str | None = None
    prompt_guidelines: tuple[str, ...] = ()

    async def execute(
        self,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        """Execute the tool without knowing its concrete implementation."""
        return await self.executor(arguments, signal=signal)
