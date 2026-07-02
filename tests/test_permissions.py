"""Tests for coding-tool approval policies and bounded previews."""

import asyncio
from io import StringIO
from pathlib import Path

from axis_agent import AgentTool, AgentToolResult, ToolCall
from axis_coding.permissions import (
    PolicyToolApprovalHandler,
    ToolApprovalPolicy,
    build_tool_approval_preview,
)


class TtyInput(StringIO):
    def isatty(self) -> bool:
        return True


async def _unused_executor(arguments: object, signal: object | None = None) -> AgentToolResult:
    del arguments, signal
    return AgentToolResult(tool_call_id="", name="read", ok=True, content="unused")


def _tool(name: str) -> AgentTool:
    return AgentTool(name, name, {"type": "object"}, _unused_executor)  # type: ignore[arg-type]


def test_tool_approval_previews_show_exact_high_risk_arguments(tmp_path: Path) -> None:
    write_path = tmp_path / "new.py"
    previews = [
        build_tool_approval_preview(
            ToolCall(id="read-1", name="read", arguments={"path": "README.md", "offset": 5}),
            cwd=tmp_path,
        ),
        build_tool_approval_preview(
            ToolCall(
                id="write-1",
                name="write",
                arguments={"path": str(write_path), "content": "print('hello')\n"},
            ),
            cwd=tmp_path,
        ),
        build_tool_approval_preview(
            ToolCall(
                id="edit-1",
                name="edit",
                arguments={
                    "path": "app.py",
                    "edits": [{"oldText": "before", "newText": "after"}],
                },
            ),
            cwd=tmp_path,
        ),
        build_tool_approval_preview(
            ToolCall(
                id="bash-1",
                name="bash",
                arguments={"command": "rm -rf build", "timeout": 10},
            ),
            cwd=tmp_path,
        ),
    ]

    rendered = "\n".join(preview.render_plain() for preview in previews)
    assert str(tmp_path / "README.md") in rendered
    assert "offset=5" in rendered
    assert "print('hello')" in rendered
    assert "before" in rendered and "after" in rendered
    assert "rm -rf build" in rendered
    assert "Timeout: 10" in rendered


def test_print_policy_allow_and_deny_are_non_interactive(tmp_path: Path) -> None:
    call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})

    allow = PolicyToolApprovalHandler(ToolApprovalPolicy.ALLOW, cwd=tmp_path)
    deny = PolicyToolApprovalHandler(ToolApprovalPolicy.DENY, cwd=tmp_path)

    assert asyncio.run(allow(_tool("read"), call)) == "allow_once"
    assert asyncio.run(deny(_tool("read"), call)) == "deny"


def test_print_ask_policy_fails_closed_without_tty(tmp_path: Path) -> None:
    handler = PolicyToolApprovalHandler(
        ToolApprovalPolicy.ASK,
        cwd=tmp_path,
        stdin=StringIO("y\n"),
        stderr=StringIO(),
    )

    decision = asyncio.run(
        handler(
            _tool("bash"),
            ToolCall(id="call-1", name="bash", arguments={"command": "echo unsafe"}),
        )
    )

    assert decision == "deny"


def test_print_ask_policy_can_allow_tool_for_current_session(tmp_path: Path) -> None:
    stdin = TtyInput("a\n")
    stderr = StringIO()
    handler = PolicyToolApprovalHandler(
        ToolApprovalPolicy.ASK,
        cwd=tmp_path,
        stdin=stdin,
        stderr=stderr,
    )
    tool = _tool("read")

    first = asyncio.run(
        handler(tool, ToolCall(id="call-1", name="read", arguments={"path": "a.py"}))
    )
    second = asyncio.run(
        handler(tool, ToolCall(id="call-2", name="read", arguments={"path": "b.py"}))
    )

    assert first == "allow_session"
    assert second == "allow_session"
    assert stderr.getvalue().count("Allow read?") == 1
