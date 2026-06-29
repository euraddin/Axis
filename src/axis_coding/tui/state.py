"""Structured display state independent of Textual widgets."""

from dataclasses import dataclass, field
from typing import Literal

from axis_agent import AgentToolResult, ToolCall
from axis_agent.types import JSONValue

type TuiMessageRole = Literal["user", "assistant", "thinking"]
type TuiToolStatus = Literal["running", "succeeded", "failed", "cancelled"]
type TuiNoticeLevel = Literal["status", "retry", "error"]


@dataclass(slots=True)
class TuiMessageItem:
    """One committed user, assistant, or reasoning block."""

    role: TuiMessageRole
    text: str
    kind: Literal["message"] = field(default="message", init=False)


@dataclass(slots=True)
class TuiToolItem:
    """One tool call with its evolving lifecycle and structured result."""

    tool_call: ToolCall
    summary: str
    status: TuiToolStatus = "running"
    updates: list[str] = field(default_factory=list)
    result: AgentToolResult | None = None
    kind: Literal["tool"] = field(default="tool", init=False)


@dataclass(slots=True)
class TuiNoticeItem:
    """A retry, status, or error notice in transcript order."""

    level: TuiNoticeLevel
    text: str
    attempt: int | None = None
    max_attempts: int | None = None
    recoverable: bool = True
    kind: Literal["notice"] = field(default="notice", init=False)


type TuiItem = TuiMessageItem | TuiToolItem | TuiNoticeItem


@dataclass(slots=True)
class TuiState:
    """Mutable accumulated state consumed by a future Textual view."""

    items: list[TuiItem] = field(default_factory=list)
    assistant_buffer: str = ""
    thinking_buffer: str = ""
    running: bool = False
    cancelled: bool = False
    current_turn: int | None = None
    error: str | None = None
    queued_steering: tuple[str, ...] = ()
    queued_follow_up: tuple[str, ...] = ()

    @property
    def queued_message_count(self) -> int:
        """Return the number of messages waiting in Harness queues."""
        return len(self.queued_steering) + len(self.queued_follow_up)

    @property
    def active_tool_count(self) -> int:
        """Return the number of tool calls that have not completed."""
        return sum(
            isinstance(item, TuiToolItem) and item.status == "running" for item in self.items
        )


def format_tool_call_summary(tool_call: ToolCall) -> str:
    """Return a concise human-readable summary without losing structured arguments."""
    arguments = tool_call.arguments
    if tool_call.name == "read":
        path = _string_argument(arguments, "path")
        if path is not None:
            return f"read {path}{_read_range(arguments)}"
    elif tool_call.name in {"write", "edit"}:
        path = _string_argument(arguments, "path")
        if path is not None:
            return f"{tool_call.name} {path}"
    elif tool_call.name == "bash":
        command = _string_argument(arguments, "command")
        if command is not None:
            timeout = _number_argument(arguments, "timeout")
            suffix = f" (timeout {timeout:g}s)" if timeout is not None else ""
            return f"$ {command}{suffix}"

    return f"{tool_call.name} {tool_call.arguments}" if tool_call.arguments else tool_call.name


def _read_range(arguments: dict[str, JSONValue]) -> str:
    offset = _integer_argument(arguments, "offset")
    limit = _integer_argument(arguments, "limit")
    if offset is None and limit is None:
        return ""
    start = max(1, offset or 1)
    if limit is None:
        return f":{start}-"
    return f":{start}-{start + max(1, limit) - 1}"


def _string_argument(arguments: dict[str, JSONValue], name: str) -> str | None:
    value = arguments.get(name)
    return value if isinstance(value, str) else None


def _integer_argument(arguments: dict[str, JSONValue], name: str) -> int | None:
    value = arguments.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number_argument(arguments: dict[str, JSONValue], name: str) -> float | None:
    value = arguments.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
