"""Tool-approval policies and human-readable coding-tool previews."""

from __future__ import annotations

import json
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, TextIO

from axis_agent import (
    AgentTool,
    ToolApprovalDecision,
    ToolApprovalHandler,
    ToolCall,
)
from axis_agent.tools import ToolCancellationToken
from axis_agent.types import JSONValue

PREVIEW_TEXT_LIMIT = 2_000


class ToolApprovalPolicy(StrEnum):
    """Top-level policy used by non-interactive frontends."""

    ASK = "ask"
    DENY = "deny"
    ALLOW = "allow"


@dataclass(frozen=True, slots=True)
class ToolApprovalPreview:
    """A bounded, UI-neutral description of one proposed tool call."""

    title: str
    summary: str
    details: tuple[str, ...] = ()

    def render_plain(self) -> str:
        lines = [self.summary]
        lines.extend(self.details)
        return "\n".join(lines)


class ToolApprovalResolver(Protocol):
    """Frontend-specific prompt used by the session-scoped controller."""

    def __call__(
        self,
        tool: AgentTool,
        tool_call: ToolCall,
        signal: ToolCancellationToken | None = None,
    ) -> Awaitable[ToolApprovalDecision]: ...


@dataclass(slots=True)
class SessionToolApprovalController:
    """Remember explicit per-tool approvals for one in-memory UI session."""

    resolver: ToolApprovalResolver
    allowed_tools: set[str] = field(default_factory=set)

    async def __call__(
        self,
        tool: AgentTool,
        tool_call: ToolCall,
        signal: ToolCancellationToken | None = None,
    ) -> ToolApprovalDecision:
        if signal is not None and signal.is_cancelled():
            return "deny"
        if tool.name in self.allowed_tools:
            return "allow_session"
        decision = await self.resolver(tool, tool_call, signal)
        if decision == "allow_session":
            self.allowed_tools.add(tool.name)
        return decision


class PolicyToolApprovalHandler:
    """Apply allow/deny/TTY-ask behavior for print mode."""

    def __init__(
        self,
        policy: ToolApprovalPolicy,
        *,
        cwd: Path,
        stdin: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        self.policy = policy
        self.cwd = cwd
        self.stdin = sys.stdin if stdin is None else stdin
        self.stderr = sys.stderr if stderr is None else stderr
        self._controller = SessionToolApprovalController(self._resolve)

    async def __call__(
        self,
        tool: AgentTool,
        tool_call: ToolCall,
        signal: ToolCancellationToken | None = None,
    ) -> ToolApprovalDecision:
        return await self._controller(tool, tool_call, signal)

    async def _resolve(
        self,
        tool: AgentTool,
        tool_call: ToolCall,
        signal: ToolCancellationToken | None = None,
    ) -> ToolApprovalDecision:
        del tool
        if signal is not None and signal.is_cancelled():
            return "deny"
        if self.policy is ToolApprovalPolicy.ALLOW:
            return "allow_once"
        if self.policy is ToolApprovalPolicy.DENY or not _is_interactive(self.stdin):
            return "deny"

        preview = build_tool_approval_preview(tool_call, cwd=self.cwd)
        self.stderr.write(f"\n{preview.title}\n{preview.render_plain()}\n")
        self.stderr.write("Allow? [y] once / [a] this tool for session / [d] deny: ")
        self.stderr.flush()
        answer = self.stdin.readline().strip().lower()
        if answer in {"y", "yes"}:
            return "allow_once"
        if answer in {"a", "always", "session"}:
            return "allow_session"
        return "deny"


def build_tool_approval_preview(tool_call: ToolCall, *, cwd: Path) -> ToolApprovalPreview:
    """Describe exact known-tool arguments without executing the tool."""
    arguments = tool_call.arguments
    if tool_call.name == "read":
        path = _display_path(arguments, cwd)
        offset = arguments.get("offset")
        limit = arguments.get("limit")
        range_text = "all requested content"
        if offset is not None or limit is not None:
            range_text = f"offset={offset!r}, limit={limit!r}"
        return ToolApprovalPreview(
            title="Allow read?",
            summary=f"Read file: {path}",
            details=(f"Range: {range_text}",),
        )
    if tool_call.name == "write":
        path = _display_path(arguments, cwd)
        content = arguments.get("content")
        text = content if isinstance(content, str) else repr(content)
        action = "overwrite" if Path(path).exists() else "create"
        return ToolApprovalPreview(
            title="Allow write?",
            summary=f"{action.capitalize()} file: {path}",
            details=(
                f"Content: {len(text)} characters",
                _bounded_text(text),
            ),
        )
    if tool_call.name == "edit":
        path = _display_path(arguments, cwd)
        edits = _edit_items(arguments)
        details = [f"Replacements: {len(edits)}"]
        for index, (old_text, new_text) in enumerate(edits, start=1):
            details.append(
                f"Edit {index}:\n- {_bounded_text(old_text)}\n+ {_bounded_text(new_text)}"
            )
        return ToolApprovalPreview(
            title="Allow edit?",
            summary=f"Edit file: {path}",
            details=tuple(details),
        )
    if tool_call.name == "bash":
        command = arguments.get("command")
        command_text = command if isinstance(command, str) else repr(command)
        timeout = arguments.get("timeout")
        timeout_text = "none" if timeout is None else str(timeout)
        return ToolApprovalPreview(
            title="Allow bash?",
            summary=f"Working directory: {cwd}",
            details=(
                f"Timeout: {timeout_text}",
                f"Command:\n{_bounded_text(command_text)}",
            ),
        )
    return ToolApprovalPreview(
        title=f"Allow {tool_call.name}?",
        summary=f"Tool: {tool_call.name}",
        details=(
            _bounded_text(json.dumps(arguments, ensure_ascii=False, sort_keys=True, indent=2)),
        ),
    )


def _display_path(arguments: Mapping[str, JSONValue], cwd: Path) -> str:
    value = arguments.get("path")
    if not isinstance(value, str):
        return repr(value)
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else cwd / path)


def _edit_items(arguments: Mapping[str, JSONValue]) -> tuple[tuple[str, str], ...]:
    value = arguments.get("edits")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = None
    items: list[tuple[str, str]] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            old_text = item.get("oldText")
            new_text = item.get("newText")
            if isinstance(old_text, str) and isinstance(new_text, str):
                items.append((old_text, new_text))
    old_text = arguments.get("oldText")
    new_text = arguments.get("newText")
    if isinstance(old_text, str) and isinstance(new_text, str):
        items.append((old_text, new_text))
    return tuple(items)


def _bounded_text(text: str) -> str:
    if len(text) <= PREVIEW_TEXT_LIMIT:
        return text
    omitted = len(text) - PREVIEW_TEXT_LIMIT
    return f"{text[:PREVIEW_TEXT_LIMIT]}\n… [{omitted} characters omitted]"


def _is_interactive(stream: TextIO) -> bool:
    isatty: Callable[[], bool] | None = getattr(stream, "isatty", None)
    return bool(isatty is not None and isatty())


def approval_handler_for_policy(
    policy: ToolApprovalPolicy,
    *,
    cwd: Path,
    stdin: TextIO | None = None,
    stderr: TextIO | None = None,
) -> ToolApprovalHandler:
    """Create the print-mode handler selected by the CLI policy."""
    return PolicyToolApprovalHandler(
        policy,
        cwd=cwd,
        stdin=stdin,
        stderr=stderr,
    )
