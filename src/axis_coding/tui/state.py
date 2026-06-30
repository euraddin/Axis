"""Framework-independent transcript state for Axis's Textual frontend."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from axis_agent import AgentMessage, AgentToolResult, ToolCall
from axis_agent.types import JSONValue
from axis_coding.skills import Skill, parse_skill_invocation

type ChatItemRole = Literal[
    "user",
    "assistant",
    "tool",
    "error",
    "status",
    "thinking",
    "skill",
    "branch_summary",
    "compaction_summary",
]

TOOL_RESULT_PREVIEW_LINES = 8
TOOL_PATCH_PREVIEW_LINES = 32
TERMINAL_COMMAND_OUTPUT_PREVIEW_LINES = 120
TOOL_RESULT_PREVIEW_CHARS = 2_000


@dataclass(slots=True)
class ChatItem:
    """One semantic block in the visible transcript."""

    role: ChatItemRole
    text: str
    tool_call_id: str | None = None
    tool_result_text: str | None = None
    always_show_tool_result: bool = False


@dataclass(slots=True)
class TuiState:
    """Mutable display state reduced from durable messages and live events."""

    items: list[ChatItem] = field(default_factory=list)
    assistant_buffer: str = ""
    running: bool = False
    cancelled: bool = False
    current_turn: int | None = None
    error: str | None = None
    show_tool_results: bool = False
    show_thinking: bool = False
    queued_steering: tuple[str, ...] = ()
    queued_follow_up: tuple[str, ...] = ()
    skills: tuple[Skill, ...] = ()

    def add_item(
        self,
        role: ChatItemRole,
        text: str,
        *,
        tool_call_id: str | None = None,
        tool_result_text: str | None = None,
        always_show_tool_result: bool = False,
    ) -> ChatItem:
        """Append one semantic transcript item and return it."""
        item = ChatItem(
            role=role,
            text=text,
            tool_call_id=tool_call_id,
            tool_result_text=tool_result_text,
            always_show_tool_result=always_show_tool_result,
        )
        self.items.append(item)
        return item

    def add_user_message(self, content: str) -> None:
        """Add user text while compacting expanded skills and future summaries."""
        if branch_summary := _parse_branch_summary_message(content):
            self.add_item(
                "branch_summary",
                "Branch summary (Ctrl+O to expand)",
                tool_result_text=branch_summary,
            )
            return
        if compaction_summary := _parse_compaction_summary_message(content):
            self.add_item(
                "compaction_summary",
                "Compaction summary (Ctrl+O to expand)",
                tool_result_text=compaction_summary,
            )
            return

        invocation = parse_skill_invocation(content)
        if invocation is None:
            self.add_item("user", content)
            return
        self.add_item("skill", f"Using skill: {invocation.name}")
        if invocation.additional_instructions:
            self.add_item("user", invocation.additional_instructions)

    def add_thinking_delta(self, delta: str) -> None:
        """Merge adjacent reasoning fragments into one transcript block."""
        if self.items and self.items[-1].role == "thinking":
            self.items[-1].text += delta
            return
        self.add_item("thinking", delta)

    def add_tool_call(self, tool_call: ToolCall) -> None:
        """Append one collapsed invocation, recognizing reads of known skills."""
        if skill_name := self._read_skill_name(tool_call):
            self.add_item(
                "skill",
                f"Loading skill: {skill_name}",
                tool_call_id=tool_call.id,
            )
            return
        self.add_item(
            "tool",
            format_tool_call_block(tool_call),
            tool_call_id=tool_call.id,
        )

    def record_tool_result(self, result: AgentToolResult) -> None:
        """Attach a result by call id or preserve it as an orphaned tool item."""
        result_text = format_tool_result_block(
            name=result.name,
            ok=result.ok,
            content=result.content,
            data=result.data,
        )
        for item in reversed(self.items):
            if item.role in {"tool", "skill"} and item.tool_call_id == result.tool_call_id:
                item.tool_result_text = result_text
                return
        self.add_item(
            "tool",
            format_tool_result_summary(name=result.name, ok=result.ok),
            tool_call_id=result.tool_call_id,
            tool_result_text=result_text,
        )

    def update_queue(self, *, steering: tuple[str, ...], follow_up: tuple[str, ...]) -> None:
        """Replace queued-message snapshots."""
        self.queued_steering = steering
        self.queued_follow_up = follow_up

    def toggle_tool_results(self) -> bool:
        """Toggle expanded tool results and return the new value."""
        self.show_tool_results = not self.show_tool_results
        return self.show_tool_results

    def toggle_thinking(self) -> bool:
        """Toggle reasoning-token visibility and return the new value."""
        self.show_thinking = not self.show_thinking
        return self.show_thinking

    def clear(self) -> None:
        """Clear display state without mutating the durable session."""
        self.items.clear()
        self.assistant_buffer = ""
        self.error = None

    def set_skills(self, skills: Iterable[Skill]) -> None:
        """Replace skill metadata used only for transcript presentation."""
        self.skills = tuple(skills)

    def load_messages(self, messages: Iterable[AgentMessage]) -> None:
        """Build display state directly from an authoritative transcript."""
        for message in messages:
            if message.role == "user":
                self.add_user_message(message.content)
            elif message.role == "assistant":
                if message.content:
                    self.add_item("assistant", message.content)
                for tool_call in message.tool_calls:
                    self.add_tool_call(tool_call)
            else:
                self.record_tool_result(
                    AgentToolResult(
                        tool_call_id=message.tool_call_id,
                        name=message.name,
                        ok=message.ok,
                        content=message.content,
                        data=message.data,
                        details=message.details,
                        error=message.error,
                    )
                )

    @property
    def queued_message_count(self) -> int:
        """Return all waiting steering and follow-up messages."""
        return len(self.queued_steering) + len(self.queued_follow_up)

    @property
    def active_tool_count(self) -> int:
        """Return tool calls that do not yet have an attached result."""
        return sum(
            item.role in {"tool", "skill"}
            and item.tool_call_id is not None
            and item.tool_result_text is None
            for item in self.items
        )

    def _read_skill_name(self, tool_call: ToolCall) -> str | None:
        if tool_call.name != "read":
            return None
        path = _string_argument(tool_call.arguments, "path")
        if path is None:
            return None
        read_path = _normalized_path(path)
        for skill in self.skills:
            if _normalized_path(skill.path) == read_path:
                return skill.name
        return None


def format_tool_call_block(tool_call: ToolCall) -> str:
    """Format one collapsed coding-tool invocation."""
    invocation = format_tool_call_invocation(tool_call)
    return invocation if tool_call.name == "bash" else f"→ {invocation}"


def format_tool_call_invocation(tool_call: ToolCall) -> str:
    """Format known coding tools tersely and preserve unknown arguments."""
    arguments = tool_call.arguments
    if tool_call.name == "read":
        if path := _string_argument(arguments, "path"):
            return f"read {path}{_read_range(arguments)}"
    elif tool_call.name in {"write", "edit"}:
        if path := _string_argument(arguments, "path"):
            return f"{tool_call.name} {path}"
    elif tool_call.name == "bash" and (command := _string_argument(arguments, "command")):
        timeout = _number_argument(arguments, "timeout")
        suffix = f" (timeout {timeout:g}s)" if timeout is not None else ""
        return f"$ {command}{suffix}"
    return f"{tool_call.name} {arguments}" if arguments else tool_call.name


def format_tool_result_summary(*, name: str, ok: bool) -> str:
    """Format a result that has no visible originating call."""
    return f"{'✓' if ok else '✗'} {name}"


def format_tool_result_block(
    *,
    name: str,
    ok: bool,
    content: str,
    data: dict[str, JSONValue] | None = None,
) -> str:
    """Format and bound one expandable tool result."""
    lines = [format_tool_result_summary(name=name, ok=ok)]
    if content:
        lines.append(_preview_text(content, max_lines=TOOL_RESULT_PREVIEW_LINES))
    if patch := _result_patch(name=name, ok=ok, data=data):
        lines.extend(["", "Patch:", _preview_text(patch, max_lines=TOOL_PATCH_PREVIEW_LINES)])
    return "\n".join(lines)


def format_terminal_command_result_block(
    *,
    ok: bool,
    added_to_context: bool,
    output: str,
) -> str:
    """Format an input-bar shell result that remains visible when collapsed."""
    status = "✓" if ok else "✗"
    context = "added to context" if added_to_context else "not added to context"
    lines = [f"{status} bash · {context}"]
    if output:
        lines.append(_preview_text(output, max_lines=TERMINAL_COMMAND_OUTPUT_PREVIEW_LINES))
    return "\n".join(lines)


def visible_chat_text(item: ChatItem, *, show_tool_results: bool) -> str:
    """Return the exact plain text represented by one visible item."""
    if item.role == "branch_summary":
        if show_tool_results and item.tool_result_text:
            return f"**Branch Summary**\n\n{item.tool_result_text}"
        return item.text
    if item.role == "compaction_summary":
        if show_tool_results and item.tool_result_text:
            return f"**Compaction Summary**\n\n{item.tool_result_text}"
        return item.text
    if item.role not in {"tool", "skill"} or not show_tool_results or not item.tool_result_text:
        return item.text
    return f"{item.text}\n\n{item.tool_result_text}"


def _preview_text(text: str, *, max_lines: int) -> str:
    lines = text.splitlines()
    if not lines:
        return text[:TOOL_RESULT_PREVIEW_CHARS]
    preview_lines = lines[:max_lines]
    preview = "\n".join(preview_lines)
    hidden_lines = len(lines) - len(preview_lines)
    truncated_by_chars = len(preview) > TOOL_RESULT_PREVIEW_CHARS
    if truncated_by_chars:
        preview = preview[:TOOL_RESULT_PREVIEW_CHARS].rstrip()
    if hidden_lines or truncated_by_chars:
        details: list[str] = []
        if hidden_lines:
            suffix = "s" if hidden_lines != 1 else ""
            details.append(f"{hidden_lines} more line{suffix}")
        if truncated_by_chars:
            details.append("additional text")
        preview = f"{preview}\n\n[Preview only: {', '.join(details)} hidden from the TUI.]"
    return preview


def _result_patch(
    *,
    name: str,
    ok: bool,
    data: dict[str, JSONValue] | None,
) -> str | None:
    if name != "edit" or not ok or data is None:
        return None
    patch = data.get("patch")
    return patch if isinstance(patch, str) and patch.strip() else None


def _parse_branch_summary_message(content: str) -> str | None:
    prefix = (
        "The following is a summary of a branch that this conversation came back from:\n<summary>\n"
    )
    suffix = "\n</summary>"
    if content.startswith(prefix) and content.endswith(suffix):
        return content.removeprefix(prefix).removesuffix(suffix)
    return None


def _parse_compaction_summary_message(content: str) -> str | None:
    prefix = "Previous conversation summary:\n"
    return content.removeprefix(prefix) if content.startswith(prefix) else None


def _read_range(arguments: dict[str, JSONValue]) -> str:
    offset = _integer_argument(arguments, "offset")
    limit = _integer_argument(arguments, "limit")
    if offset is None and limit is None:
        return ""
    start = max(1, offset or 1)
    return f":{start}-" if limit is None else f":{start}-{start + max(1, limit) - 1}"


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


def _normalized_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


format_tool_call_summary = format_tool_call_invocation
