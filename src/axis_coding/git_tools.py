"""Git workflow tools for Axis coding sessions."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from axis_agent.tools import (
    AgentTool,
    AgentToolResult,
    ToolCancellationToken,
)
from axis_agent.types import JSONValue

DEFAULT_LOG_COUNT = 10
DEFAULT_LOG_MAX_COUNT = 50
DEFAULT_GIT_TIMEOUT_SECONDS = 30.0
_GIT = "git"

# ---------------------------------------------------------------------------
# git subprocess helper
# ---------------------------------------------------------------------------


class GitToolError(ValueError):
    """A git tool received invalid arguments or git reported a fatal error."""


async def _git(
    *args: str,
    cwd: Path,
    timeout: float = DEFAULT_GIT_TIMEOUT_SECONDS,
    cancellation_signal: ToolCancellationToken | None = None,
) -> AgentToolResult:
    """Run ``git`` via subprocess and return a structured result."""
    command = " ".join((_GIT, *args))

    if cancellation_signal is not None and cancellation_signal.is_cancelled():
        return AgentToolResult(
            tool_call_id="",
            name="git",
            ok=False,
            content="Git command cancelled.",
            error="Cancelled",
            data={"command": command},
        )

    try:
        process = await asyncio.create_subprocess_exec(
            _GIT,
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**__import__("os").environ, "GIT_PAGER": "cat", "PAGER": "cat"},
        )
    except OSError as exc:
        return AgentToolResult(
            tool_call_id="",
            name="git",
            ok=False,
            content=f"Failed to start git: {exc}",
            error=str(exc),
            data={"command": command},
        )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        return AgentToolResult(
            tool_call_id="",
            name="git",
            ok=False,
            content=f"Git command timed out after {timeout} seconds.",
            error="Timeout",
            data={"command": command, "timed_out": True},
        )

    stdout_text = stdout_bytes.decode(errors="replace").strip()
    stderr_text = stderr_bytes.decode(errors="replace").strip()
    exit_code = process.returncode or 0

    if exit_code != 0:
        message = stderr_text or stdout_text or f"git exited with code {exit_code}"
        return AgentToolResult(
            tool_call_id="",
            name="git",
            ok=False,
            content=message,
            error=message,
            data={"command": command, "exit_code": exit_code},
        )

    return AgentToolResult(
        tool_call_id="",
        name="git",
        ok=True,
        content=stdout_text or "(no output)",
        data={"command": command, "exit_code": 0},
    )


def _truthy_arg(arguments: Mapping[str, JSONValue], name: str) -> bool:
    """Return True when *arguments[name]* is truthy."""
    value = arguments.get(name)
    return bool(value)


def _str_arg(arguments: Mapping[str, JSONValue], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise GitToolError(f"{name} must be a string")
    return value


def _optional_str_arg(arguments: Mapping[str, JSONValue], name: str) -> str | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise GitToolError(f"{name} must be a string")
    stripped = value.strip()
    return stripped if stripped else None


def _optional_int_arg(
    arguments: Mapping[str, JSONValue], name: str, *, min_value: int | None = None
) -> int | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise GitToolError(f"{name} must be an integer")
    if min_value is not None and value < min_value:
        raise GitToolError(f"{name} must be at least {min_value}")
    return value


# ---------------------------------------------------------------------------
# git_status
# ---------------------------------------------------------------------------


def _parse_porcelain_status(output: str) -> dict[str, list[str]]:
    """Parse ``git status --porcelain=v1`` into categorised file lists.

    Returns a dict with keys ``staged``, ``unstaged``, ``untracked``,
    ``conflict``, and ``ignored``.
    """
    categories: dict[str, list[str]] = {
        "staged": [],
        "unstaged": [],
        "untracked": [],
        "conflict": [],
        "ignored": [],
    }
    if not output:
        return categories

    for line in output.split("\n"):
        if len(line) < 3:
            continue
        xy = line[:2]
        filename = line[3:].strip()
        # Resolve double-quoted filenames (core.quotePath defaults to true).
        if filename.startswith('"') and filename.endswith('"'):
            filename = _unescape_git_quoted_path(filename)

        x, y = xy[0], xy[1]

        if x in {"D", "A", "M", "R", "C", "U"}:
            categories["staged"].append(filename)

        if y in {"M", "D"}:
            categories["unstaged"].append(filename)
        elif y == "?":
            categories["untracked"].append(filename)
        elif y == "!":
            categories["ignored"].append(filename)

        if x == "U" or y == "U" or xy in {"AA", "DD", "AU", "UA", "DU", "UD"}:
            categories["conflict"].append(filename)

    return categories


def _unescape_git_quoted_path(quoted: str) -> str:
    """Convert a C-style git-quoted path (``"a\\b"``) back to the real path."""
    inner = quoted[1:-1]
    result: list[str] = []
    octal_bytes: list[int] = []

    def _flush_octals() -> None:
        if octal_bytes:
            result.append(bytes(octal_bytes).decode("utf-8", errors="replace"))
            octal_bytes.clear()

    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch == "\\" and i + 1 < len(inner):
            next_ch = inner[i + 1]
            if next_ch == "\\":
                _flush_octals()
                result.append("\\")
                i += 2
                continue
            if next_ch == "n":
                _flush_octals()
                result.append("\n")
                i += 2
                continue
            if next_ch == "t":
                _flush_octals()
                result.append("\t")
                i += 2
                continue
            # Octal sequence: \ooo — accumulate as UTF-8 bytes.
            if next_ch.isdigit():
                octal = ""
                j = i + 1
                while j < len(inner) and inner[j].isdigit() and j - (i + 1) < 3:
                    octal += inner[j]
                    j += 1
                if octal:
                    octal_bytes.append(int(octal, 8))
                    i = j
                    continue
            _flush_octals()
            result.append(ch)
            i += 1
        else:
            _flush_octals()
            result.append(ch)
            i += 1
    _flush_octals()
    return "".join(result)


async def _execute_git_status(
    arguments: Mapping[str, JSONValue],
    *,
    cwd: Path,
) -> AgentToolResult:
    del arguments
    # 1. Get the current branch.
    branch_result = await _git("branch", "--show-current", cwd=cwd)
    branch = (
        branch_result.content.strip()
        if branch_result.ok and branch_result.content != "(no output)"
        else "HEAD (detached)"
    )

    # 2. Get porcelain status.
    status_result = await _git("status", "--porcelain=v1", "--branch", cwd=cwd)
    if not status_result.ok:
        return AgentToolResult(
            tool_call_id="",
            name="git_status",
            ok=False,
            content=f"git status failed: {status_result.content}",
            error=status_result.error,
            data={"branch": branch},
        )

    porcelain = status_result.content
    parsed = _parse_porcelain_status(porcelain)

    lines = ["# Git Status", "", f"**Branch:** {branch}", ""]
    for category, label in (
        ("staged", "Staged for commit"),
        ("unstaged", "Modified but not staged"),
        ("untracked", "Untracked files"),
        ("conflict", "Conflicts"),
        ("ignored", "Ignored files"),
    ):
        files = parsed[category]
        if not files:
            continue
        lines.append(f"## {label} ({len(files)})")
        for f in files:
            lines.append(f"- `{f}`")
        lines.append("")

    if not any(parsed[cat] for cat in parsed):
        lines.append("Working tree clean. Nothing to commit.")

    summary_parts = [
        f"{len(parsed['staged'])} staged",
        f"{len(parsed['unstaged'])} modified",
        f"{len(parsed['untracked'])} untracked",
    ]
    conflict_count = len(parsed["conflict"])
    if conflict_count:
        summary_parts.append(f"{conflict_count} conflicted")

    staged: list[JSONValue] = list(parsed["staged"])
    unstaged: list[JSONValue] = list(parsed["unstaged"])
    untracked: list[JSONValue] = list(parsed["untracked"])
    conflict: list[JSONValue] = list(parsed["conflict"])
    data: dict[str, JSONValue] = {
        "branch": branch,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
        "conflict": conflict,
        "clean": not any(parsed[cat] for cat in parsed),
        "summary": ", ".join(summary_parts),
    }
    return AgentToolResult(
        tool_call_id="",
        name="git_status",
        ok=True,
        content="\n".join(lines),
        data=data,
    )


@dataclass(frozen=True, slots=True)
class GitStatusToolDefinition:
    """Definition for the ``git_status`` tool."""

    name: str = "git_status"
    description: str = (
        "Show the working tree status in a structured format. Returns the current "
        "branch and categorised file lists: staged, unstaged, untracked, and conflicts."
    )
    prompt_snippet: str = "Show working tree status with categorised file lists"
    prompt_guidelines: tuple[str, ...] = (
        "Use git_status to inspect the state of the working tree before committing.",
        "Prefer git_status over shelling out to git status for structured output.",
    )
    input_schema: Mapping[str, JSONValue] = field(default_factory=lambda: {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    })
    requires_approval: bool = False

    def to_agent_tool(self, *, cwd: Path) -> AgentTool:
        async def execute(
            arguments: Mapping[str, JSONValue],
            signal: ToolCancellationToken | None = None,
        ) -> AgentToolResult:
            del signal
            return await _execute_git_status(arguments, cwd=cwd)

        return AgentTool(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            executor=execute,
            requires_approval=self.requires_approval,
            prompt_snippet=self.prompt_snippet,
            prompt_guidelines=self.prompt_guidelines,
        )


def create_git_status_tool(*, cwd: Path) -> AgentTool:
    """Create the ``git_status`` tool."""
    return GitStatusToolDefinition().to_agent_tool(cwd=cwd)


# ---------------------------------------------------------------------------
# git_diff
# ---------------------------------------------------------------------------


async def _execute_git_diff(
    arguments: Mapping[str, JSONValue],
    *,
    cwd: Path,
) -> AgentToolResult:
    staged = _truthy_arg(arguments, "staged")
    path = _optional_str_arg(arguments, "path")

    args: list[str] = ["diff", "--no-color"]
    if staged:
        args.append("--staged")
    if path is not None:
        args.append("--")
        args.append(path)

    result = await _git(*args, cwd=cwd)
    if not result.ok:
        result.name = "git_diff"
        return result

    # Add a stats summary line.
    stats_result = await _git(
        "diff", "--stat", *(["--staged"] if staged else []), cwd=cwd
    )
    stats_text = ""
    if stats_result.ok and stats_result.content != "(no output)":
        stats_text = f"\n\n## Stats\n\n```text\n{stats_result.content}\n```"

    output = result.content
    has_diff = output and output != "(no output)"
    result_data: dict[str, JSONValue] = dict(result.data) if result.data else {}
    result_data["staged"] = staged
    if path is not None:
        result_data["path"] = path

    return AgentToolResult(
        tool_call_id="",
        name="git_diff",
        ok=True,
        content=(
            f"# Git Diff{' (staged)' if staged else ''}\n\n"
            + (f"```diff\n{output}\n```" if has_diff else "(no changes)")
            + stats_text
        ),
        data=result_data,
    )


@dataclass(frozen=True, slots=True)
class GitDiffToolDefinition:
    """Definition for the ``git_diff`` tool."""

    name: str = "git_diff"
    description: str = (
        "Show changes between the working tree, the index, and HEAD. "
        "By default shows unstaged changes; set staged=true for staged changes. "
        "Optionally limit to a specific file or directory with the path parameter."
    )
    prompt_snippet: str = "Show working-tree and staged diffs"
    prompt_guidelines: tuple[str, ...] = (
        "Use git_diff to review changes before committing them.",
        "Check git_diff with staged=true to see what will be committed.",
    )
    input_schema: Mapping[str, JSONValue] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "staged": {
                "type": "boolean",
                "description": "Show staged changes instead of working-tree changes.",
            },
            "path": {
                "type": "string",
                "description": "Limit the diff to a specific file or directory.",
            },
        },
        "additionalProperties": False,
    })
    requires_approval: bool = False

    def to_agent_tool(self, *, cwd: Path) -> AgentTool:
        async def execute(
            arguments: Mapping[str, JSONValue],
            signal: ToolCancellationToken | None = None,
        ) -> AgentToolResult:
            del signal
            return await _execute_git_diff(arguments, cwd=cwd)

        return AgentTool(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            executor=execute,
            requires_approval=self.requires_approval,
            prompt_snippet=self.prompt_snippet,
            prompt_guidelines=self.prompt_guidelines,
        )


def create_git_diff_tool(*, cwd: Path) -> AgentTool:
    """Create the ``git_diff`` tool."""
    return GitDiffToolDefinition().to_agent_tool(cwd=cwd)


# ---------------------------------------------------------------------------
# git_log
# ---------------------------------------------------------------------------


async def _execute_git_log(
    arguments: Mapping[str, JSONValue],
    *,
    cwd: Path,
) -> AgentToolResult:
    max_count = _optional_int_arg(
        arguments, "max_count", min_value=1
    ) or DEFAULT_LOG_COUNT
    max_count = min(max_count, DEFAULT_LOG_MAX_COUNT)
    path = _optional_str_arg(arguments, "path")

    args: list[str] = [
        "log",
        f"--max-count={max_count}",
        "--oneline",
        "--decorate",
        "--no-color",
    ]
    if path is not None:
        args.extend(["--", path])

    result = await _git(*args, cwd=cwd)
    if not result.ok:
        result.name = "git_log"
        return result

    output = result.content
    entries = output.split("\n") if output and output != "(no output)" else []
    shown = len(entries)
    result_data: dict[str, JSONValue] = {
        **(dict(result.data) if result.data else {}),
        "max_count": max_count,
        "shown": shown,
    }
    if path is not None:
        result_data["path"] = path

    return AgentToolResult(
        tool_call_id="",
        name="git_log",
        ok=True,
        content=(
            f"# Git Log ({shown} commits)\n\n```text\n{output}\n```"
            if output
            else "# Git Log\n\nNo commits found."
        ),
        data=result_data,
    )


@dataclass(frozen=True, slots=True)
class GitLogToolDefinition:
    """Definition for the ``git_log`` tool."""

    name: str = "git_log"
    description: str = (
        "Show the commit history in a compact one-line format. "
        f"Returns up to {DEFAULT_LOG_MAX_COUNT} commits (default {DEFAULT_LOG_COUNT}). "
        "Optionally filter by file path."
    )
    prompt_snippet: str = "Show recent commit history"
    prompt_guidelines: tuple[str, ...] = (
        "Use git_log to understand recent changes and project direction.",
        "Use git_log with a path to see the history of a specific file.",
    )
    input_schema: Mapping[str, JSONValue] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "max_count": {
                "type": "integer",
                "description": (
                    f"Maximum number of commits (default {DEFAULT_LOG_COUNT}, "
                    f"max {DEFAULT_LOG_MAX_COUNT})."
                ),
            },
            "path": {
                "type": "string",
                "description": "Limit to commits affecting this file or directory.",
            },
        },
        "additionalProperties": False,
    })
    requires_approval: bool = False

    def to_agent_tool(self, *, cwd: Path) -> AgentTool:
        async def execute(
            arguments: Mapping[str, JSONValue],
            signal: ToolCancellationToken | None = None,
        ) -> AgentToolResult:
            del signal
            try:
                return await _execute_git_log(arguments, cwd=cwd)
            except GitToolError as exc:
                return AgentToolResult(
                    tool_call_id="",
                    name="git_log",
                    ok=False,
                    content=str(exc),
                    error=str(exc),
                )

        return AgentTool(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            executor=execute,
            requires_approval=self.requires_approval,
            prompt_snippet=self.prompt_snippet,
            prompt_guidelines=self.prompt_guidelines,
        )


def create_git_log_tool(*, cwd: Path) -> AgentTool:
    """Create the ``git_log`` tool."""
    return GitLogToolDefinition().to_agent_tool(cwd=cwd)


# ---------------------------------------------------------------------------
# git_commit
# ---------------------------------------------------------------------------


async def _execute_git_commit(
    arguments: Mapping[str, JSONValue],
    *,
    cwd: Path,
) -> AgentToolResult:
    message = _str_arg(arguments, "message")
    if not message.strip():
        return AgentToolResult(
            tool_call_id="",
            name="git_commit",
            ok=False,
            content="Commit message cannot be empty.",
            error="Empty commit message",
        )

    # Verify there are staged changes before committing.
    status_result = await _git("diff", "--staged", "--quiet", cwd=cwd)
    if status_result.ok and status_result.data and status_result.data.get("exit_code") == 0:
        return AgentToolResult(
            tool_call_id="",
            name="git_commit",
            ok=False,
            content="Nothing to commit. Stage changes first (use bash: git add <files>).",
            error="No staged changes",
            data={"message": message.strip()},
        )

    result = await _git("commit", "-m", message.strip(), cwd=cwd)
    if not result.ok:
        result.name = "git_commit"
        return result

    # Get the new commit hash.
    hash_result = await _git("rev-parse", "--short", "HEAD", cwd=cwd)
    commit_hash = (
        hash_result.content.strip()
        if hash_result.ok and hash_result.content != "(no output)"
        else None
    )

    result_data: dict[str, JSONValue] = {
        "message": message.strip(),
        "commit": commit_hash,
    }

    return AgentToolResult(
        tool_call_id="",
        name="git_commit",
        ok=True,
        content=(
            f"# Commit Created\n\n"
            f"**Commit:** `{commit_hash or 'unknown'}`\n"
            f"**Message:** {message.strip()}\n\n"
            + f"```text\n{result.content}\n```"
        ),
        data=result_data,
    )


@dataclass(frozen=True, slots=True)
class GitCommitToolDefinition:
    """Definition for the ``git_commit`` tool."""

    name: str = "git_commit"
    description: str = (
        "Create a new commit with the currently staged changes. "
        "Stage files first with `git add` via the bash tool. "
        "Run git_diff with staged=true to review what will be committed."
    )
    prompt_snippet: str = "Commit staged changes with a message"
    prompt_guidelines: tuple[str, ...] = (
        "Commit messages should be concise, descriptive, and use imperative mood "
        "(e.g. 'Add feature X' not 'Added feature X').",
        "Stage changes with bash: git add before using git_commit.",
        "Review staged changes with git_diff(staged=true) before committing.",
        "Make small, focused commits — one logical change per commit.",
    )
    input_schema: Mapping[str, JSONValue] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Commit message in imperative mood.",
            },
        },
        "required": ["message"],
        "additionalProperties": False,
    })
    requires_approval: bool = True

    def to_agent_tool(self, *, cwd: Path) -> AgentTool:
        async def execute(
            arguments: Mapping[str, JSONValue],
            signal: ToolCancellationToken | None = None,
        ) -> AgentToolResult:
            del signal
            return await _execute_git_commit(arguments, cwd=cwd)

        return AgentTool(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            executor=execute,
            requires_approval=self.requires_approval,
            prompt_snippet=self.prompt_snippet,
            prompt_guidelines=self.prompt_guidelines,
        )


def create_git_commit_tool(*, cwd: Path) -> AgentTool:
    """Create the ``git_commit`` tool."""
    return GitCommitToolDefinition().to_agent_tool(cwd=cwd)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def create_git_tools(*, cwd: Path) -> list[AgentTool]:
    """Create Axis's default git workflow tools in stable order.

    Returns ``[git_status, git_diff, git_log, git_commit]``.
    """
    return [
        create_git_status_tool(cwd=cwd),
        create_git_diff_tool(cwd=cwd),
        create_git_log_tool(cwd=cwd),
        create_git_commit_tool(cwd=cwd),
    ]
