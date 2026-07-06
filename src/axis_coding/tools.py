"""Built-in local coding tools for Axis sessions."""

import asyncio
import base64
import difflib
import json
import os
import signal
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any

from axis_agent.tools import AgentTool, AgentToolResult, ToolCancellationToken, ToolExecutor
from axis_agent.types import JSONValue

DEFAULT_MAX_OUTPUT_BYTES = 50 * 1024
DEFAULT_MAX_OUTPUT_LINES = 2_000
SUPPORTED_IMAGE_MIME_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
UTF8_BOM = "\ufeff"

_file_locks: dict[Path, asyncio.Lock] = {}


class ToolInputError(ValueError):
    """Raised when a coding tool receives invalid structured arguments."""


@dataclass(frozen=True, slots=True)
class TruncationResult:
    """Structured metadata describing head-truncated output."""

    content: str
    truncated: bool
    truncated_by: str | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int

    def to_json(self) -> dict[str, JSONValue]:
        """Return JSON-compatible truncation metadata."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A coding tool plus schema and system-prompt metadata."""

    name: str
    description: str
    prompt_snippet: str
    prompt_guidelines: tuple[str, ...]
    input_schema: Mapping[str, JSONValue]
    executor: ToolExecutor
    requires_approval: bool
    auto_approve_if: Callable[[Mapping[str, JSONValue]], bool] | None = field(
        default=None, compare=False, hash=False
    )

    def to_agent_tool(self) -> AgentTool:
        """Narrow this application definition to the portable core contract."""
        return AgentTool(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            executor=self.executor,
            requires_approval=self.requires_approval,
            auto_approve_if=self.auto_approve_if,
            prompt_snippet=self.prompt_snippet,
            prompt_guidelines=self.prompt_guidelines,
        )


def create_coding_tools(
    *,
    cwd: str | Path | None = None,
    include_web_tools: bool = True,
    include_git_tools: bool = True,
    include_lint_tool: bool = True,
) -> list[AgentTool]:
    """Create Axis's default tools in stable order.

    Order: read, write, edit, bash, git_status, git_diff, git_log, git_commit,
    lint, web_fetch, web_search.
    """
    root = Path.cwd() if cwd is None else Path(cwd)
    tools: list[AgentTool] = [
        create_read_tool(cwd=root),
        create_write_tool(cwd=root),
        create_edit_tool(cwd=root),
        create_bash_tool(cwd=root),
    ]
    if include_git_tools:
        from axis_coding.git_tools import create_git_tools

        tools.extend(create_git_tools(cwd=root))
    if include_lint_tool:
        from axis_coding.lint_tools import create_lint_tool

        tools.append(create_lint_tool(cwd=root))
    if include_web_tools:
        from axis_coding.web_tools import create_web_tools

        tools.extend(create_web_tools())
    return tools


def create_read_tool_definition(*, cwd: str | Path | None = None) -> ToolDefinition:
    """Create the local ``read`` tool bound to a working directory."""
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        del signal
        raw_path = _str_arg(arguments, "path")
        path = _path_arg(arguments, "path", cwd=root)
        offset = _optional_int_arg(arguments, "offset")
        limit = _optional_int_arg(arguments, "limit")

        if offset is not None and offset < 0:
            raise ToolInputError("offset must be at least 0")
        if limit is not None and limit < 1:
            raise ToolInputError("limit must be at least 1")
        if not path.exists():
            raise ToolInputError(f"File not found: {path}")
        if path.is_dir():
            raise ToolInputError(f"Path is a directory: {path}")

        mime_type = SUPPORTED_IMAGE_MIME_TYPES.get(path.suffix.lower())
        if mime_type is not None:
            image_bytes = path.read_bytes()
            return AgentToolResult(
                tool_call_id="",
                name="read",
                ok=True,
                content=f"Read image file [{mime_type}]",
                data={
                    "path": str(path),
                    "mime_type": mime_type,
                    "bytes": len(image_bytes),
                    "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                },
            )

        text = path.read_text(encoding="utf-8")
        all_lines = text.split("\n")
        start_line = 0 if offset is None or offset == 0 else offset - 1
        if start_line >= len(all_lines):
            raise ToolInputError(
                f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)"
            )

        user_limited_lines: int | None = None
        if limit is not None:
            end_line = min(start_line + limit, len(all_lines))
            selected = "\n".join(all_lines[start_line:end_line])
            user_limited_lines = end_line - start_line
        else:
            selected = "\n".join(all_lines[start_line:])

        truncation = truncate_head(selected)
        start_display = start_line + 1
        metadata: dict[str, JSONValue] = {
            "path": str(path),
            "truncation": truncation.to_json(),
        }

        if truncation.first_line_exceeds_limit:
            first_line_bytes = len(all_lines[start_line].encode())
            output = (
                f"[Line {start_display} is {format_size(first_line_bytes)}, exceeds "
                f"{format_size(DEFAULT_MAX_OUTPUT_BYTES)} limit. Use bash: sed -n "
                f"'{start_display}p' {raw_path} | head -c {DEFAULT_MAX_OUTPUT_BYTES}]"
            )
        elif truncation.truncated:
            end_display = start_display + truncation.output_lines - 1
            next_offset = end_display + 1
            output = truncation.content
            if truncation.truncated_by == "lines":
                output += (
                    f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output += (
                    f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)} "
                    f"({format_size(DEFAULT_MAX_OUTPUT_BYTES)} limit). "
                    f"Use offset={next_offset} to continue.]"
                )
        elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
            remaining = len(all_lines) - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output = (
                f"{truncation.content}\n\n[{remaining} more lines in file. "
                f"Use offset={next_offset} to continue.]"
            )
        else:
            output = truncation.content

        return AgentToolResult(
            tool_call_id="",
            name="read",
            ok=True,
            content=output,
            data=metadata,
        )

    return ToolDefinition(
        name="read",
        description=(
            "Read a text file or supported image. Text output is truncated to "
            f"{DEFAULT_MAX_OUTPUT_LINES} lines or {DEFAULT_MAX_OUTPUT_BYTES // 1024}KB. "
            "Use offset and limit to read large files incrementally."
        ),
        prompt_snippet="Read file contents",
        prompt_guidelines=("Use read to examine files instead of shelling out to cat or sed.",),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
                "offset": {
                    "type": "integer",
                    "description": "1-based line number; 0 also means the beginning",
                },
                "limit": {"type": "integer", "description": "Maximum number of lines"},
            },
            "required": ["path"],
        },
        executor=execute,
        requires_approval=False,
    )


def create_read_tool(*, cwd: str | Path | None = None) -> AgentTool:
    """Create the provider-neutral local ``read`` tool."""
    return create_read_tool_definition(cwd=cwd).to_agent_tool()


def create_write_tool_definition(*, cwd: str | Path | None = None) -> ToolDefinition:
    """Create the local ``write`` tool bound to a working directory."""
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        del signal
        path = _path_arg(arguments, "path", cwd=root)
        content = _str_arg(arguments, "content")

        async with _file_lock(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        return AgentToolResult(
            tool_call_id="",
            name="write",
            ok=True,
            content=f"Successfully wrote to {path}.",
            data={"path": str(path), "characters": len(content)},
        )

    return ToolDefinition(
        name="write",
        description=(
            "Write complete UTF-8 content to a file. Creates parent directories and "
            "overwrites any existing file."
        ),
        prompt_snippet="Create or completely rewrite files",
        prompt_guidelines=("Use write only for new files or intentional complete rewrites.",),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Complete file content"},
            },
            "required": ["path", "content"],
        },
        executor=execute,
        requires_approval=True,
    )


def create_write_tool(*, cwd: str | Path | None = None) -> AgentTool:
    """Create the provider-neutral local ``write`` tool."""
    return create_write_tool_definition(cwd=cwd).to_agent_tool()


def create_edit_tool_definition(*, cwd: str | Path | None = None) -> ToolDefinition:
    """Create the local atomic exact-replacement ``edit`` tool."""
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        del signal
        prepared = _prepare_edit_arguments(arguments)
        path = _path_arg(prepared, "path", cwd=root)
        edits = _edits_arg(prepared)

        if not path.exists():
            raise ToolInputError(f"Could not edit file: {path}. File not found.")
        if path.is_dir():
            raise ToolInputError(f"Could not edit file: {path}. Path is a directory.")

        async with _file_lock(path):
            raw_content = path.read_bytes().decode("utf-8")
            bom, content = _strip_bom(raw_content)
            original_ending = detect_line_ending(content)
            normalized = normalize_to_lf(content)
            base_content, new_content = apply_edits_to_normalized_content(
                normalized,
                edits,
                str(path),
            )
            final_content = bom + restore_line_endings(new_content, original_ending)
            path.write_bytes(final_content.encode("utf-8"))

        diff_text, first_changed_line = generate_diff_string(base_content, new_content)
        patch = generate_unified_patch(str(path), base_content, new_content)
        return AgentToolResult(
            tool_call_id="",
            name="edit",
            ok=True,
            content=f"Successfully replaced {len(edits)} block(s) in {path}.",
            data={
                "path": str(path),
                "edits": len(edits),
                "diff": diff_text,
                "patch": patch,
                "first_changed_line": first_changed_line,
            },
        )

    return ToolDefinition(
        name="edit",
        description=(
            "Edit one UTF-8 file with exact replacements. Every oldText must identify one "
            "unique, non-overlapping region of the original file. All edits are validated "
            "before the file is written."
        ),
        prompt_snippet="Make precise file edits with exact text replacement",
        prompt_guidelines=(
            "Use edit for precise changes; every oldText must match exactly once.",
            "Combine disjoint changes to one file into a single edit call.",
            "All oldText values are matched against the original file, not intermediate output.",
            "Keep oldText small while including enough context to make it unique.",
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit"},
                "edits": {
                    "type": "array",
                    "description": "One or more exact replacements",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {"type": "string"},
                            "newText": {"type": "string"},
                        },
                        "required": ["oldText", "newText"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["path", "edits"],
            "additionalProperties": False,
        },
        executor=execute,
        requires_approval=True,
    )


def create_edit_tool(*, cwd: str | Path | None = None) -> AgentTool:
    """Create the provider-neutral local ``edit`` tool."""
    return create_edit_tool_definition(cwd=cwd).to_agent_tool()


# ---------------------------------------------------------------------------
# Bash command safety classifier for auto-approval
# ---------------------------------------------------------------------------

_READ_ONLY_COMMANDS: frozenset[str] = frozenset({
    # File inspection
    "ls", "dir", "cat", "head", "tail", "more", "less", "zcat", "zless",
    "nl", "od", "hexdump", "xxd", "strings",
    # Search / discovery
    "grep", "egrep", "fgrep", "rg", "ag", "ack",
    "find", "locate", "which", "whereis", "whence", "type",
    # Metadata / counting
    "wc", "stat", "file", "du", "df",
    # Output-only
    "echo", "printf", "pwd", "whoami", "who", "id", "groups",
    "date", "cal", "uptime", "hostname", "uname", "arch",
    "env", "printenv", "locale",
    # Process inspection
    "ps", "pgrep", "pidof", "pstree", "top", "htop",
    # Text processing (read-only)
    "sort", "uniq", "cut", "paste", "join", "tr",
    "expand", "unexpand",
    # Comparison
    "diff", "cmp", "comm", "sdiff",
    # Help / documentation
    "man", "info", "whatis", "apropos", "help",
    # Path utilities
    "tree", "realpath", "readlink", "dirname", "basename",
})

_READ_ONLY_GIT_SUBCOMMANDS: frozenset[str] = frozenset({
    "log", "show", "diff", "status", "blame",
    "rev-parse", "rev-list", "ls-files", "ls-tree", "describe",
    "branch", "tag", "remote", "stash", "config",
    "shortlog", "whatchanged", "reflog",
})

_READ_ONLY_DOCKER_SUBCOMMANDS: frozenset[str] = frozenset({
    "ps", "images", "inspect", "logs", "stats",
    "version", "info", "history", "top",
})

_READ_ONLY_KUBECTL_SUBCOMMANDS: frozenset[str] = frozenset({
    "get", "describe", "logs", "explain", "top",
    "api-resources", "api-versions", "cluster-info",
    "config view", "version", "auth can-i",
})

_DESTRUCTIVE_PATTERNS: tuple[str, ...] = (
    ">", ">>", "| tee ", "|tee ",
)

_MULTI_COMMAND_GIT = frozenset({
    "branch", "tag", "remote", "stash", "config",
})

_MULTI_READ_ONLY_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "pip": frozenset({"list", "show", "freeze", "config", "cache list"}),
    "pip3": frozenset({"list", "show", "freeze", "config", "cache list"}),
    "npm": frozenset({"list", "view", "outdated", "ls", "info", "config list"}),
    "yarn": frozenset({"list", "info", "config", "why"}),
    "pnpm": frozenset({"list", "view", "outdated"}),
    "cargo": frozenset({"check", "doc", "tree", "metadata", "readme"}),
    "go": frozenset({"version", "env", "doc", "list", "mod why", "mod graph"}),
    "python": frozenset({"-V", "--version", "-c"}),
    "python3": frozenset({"-V", "--version", "-c"}),
    "node": frozenset({"-v", "--version", "-e", "-p"}),
    "rustc": frozenset({"-V", "--version"}),
    "docker": _READ_ONLY_DOCKER_SUBCOMMANDS,
    "kubectl": _READ_ONLY_KUBECTL_SUBCOMMANDS,
}


def _bash_command_is_read_only(arguments: Mapping[str, JSONValue]) -> bool:
    """Return True when *arguments* describe a read-only shell invocation.

    The classifier is a user-facing convenience, not a security boundary:
    it errs on the side of requiring approval for unrecognised commands.
    """
    raw = arguments.get("command")
    if not isinstance(raw, str) or not raw.strip():
        return False
    command = raw.strip()

    # Output redirection or tee → may write to the filesystem.
    for pattern in _DESTRUCTIVE_PATTERNS:
        if pattern in command:
            return False

    # Break on the first shell metacharacter to isolate the command word.
    main = _first_command_word(command)
    if not main:
        return False

    # Always-read-only commands.
    if main in _READ_ONLY_COMMANDS:
        return True

    # Dual-use commands with subcommand awareness.
    subcommand = _first_subcommand(command, main)

    if main == "git":
        if subcommand is None:
            return True  # plain "git" is read-only
        if subcommand in _READ_ONLY_GIT_SUBCOMMANDS:
            if subcommand in _MULTI_COMMAND_GIT:
                subsub = _first_subcommand(command, subcommand)
                if subcommand == "branch":
                    return subsub in {None, "-r", "-a", "-l", "--list", "--remote", "--all"}
                if subcommand in {"tag", "stash"}:
                    return subsub in {None, "-l", "--list"} or (
                        subcommand == "stash" and subsub == "list"
                    )
                if subcommand == "remote":
                    return subsub in {None, "-v", "--verbose", "show"}
                if subcommand == "config":
                    return subsub in {None, "--list", "--get", "--get-regexp", "-l",
                                      "--global", "--local", "--system"}
            return True
        return False

    if main in _MULTI_READ_ONLY_SUBCOMMANDS:
        allowed = _MULTI_READ_ONLY_SUBCOMMANDS[main]
        if not allowed:
            return False
        if subcommand is None:
            # Commands like "cargo" with no subcommand are safe.
            return True
        # Allow flag-style subcommands (-V, --version, -c for python, -e for node).
        for sc in allowed:
            if subcommand == sc or subcommand.startswith(f"{sc} "):
                return True
            if sc.startswith("-") and subcommand.startswith(sc):
                return True
        return False

    return False


def _first_command_word(command: str) -> str | None:
    """Return the first shell word in *command*, skipping env assignments."""
    tokens = _shell_split(command)
    if not tokens:
        return None
    word = tokens[0]
    # Skip leading VAR=value assignments.
    while "=" in word and word.partition("=")[0].isidentifier():
        tokens.pop(0)
        word = tokens[0] if tokens else ""
    if not word:
        return None
    # Skip common no-op wrappers only when followed by another word.
    while word in {"command", "exec", "builtin", "env", "nice", "nohup", "time"}:
        if len(tokens) <= 1:
            break
        tokens.pop(0)
        word = tokens[0] if tokens else ""
    return word or None


def _first_subcommand(command: str, main: str) -> str | None:
    """Return the first argument after *main* that isn't a flag."""
    parts = command.split()
    try:
        idx = parts.index(main)
    except ValueError:
        return None
    for token in parts[idx + 1 :]:
        if not token.startswith("-"):
            return token
    return None


def _shell_split(command: str) -> list[str]:
    """Split *command* on unquoted whitespace (approximate, fast)."""
    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(command):
        ch = command[i]
        if quote:
            if ch == quote:
                quote = None
            else:
                current.append(ch)
        elif ch in {"'", '"'}:
            quote = ch
        elif ch in {" ", "\t"}:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(ch)
        i += 1
    if current:
        tokens.append("".join(current))
    return tokens


def _sandbox_docker_available() -> bool:
    """Return True when Docker appears to be installed and reachable."""
    import shutil

    return shutil.which("docker") is not None


def _wrap_sandbox_command(command: str, cwd: Path, image: str) -> str:
    """Wrap *command* so it runs inside a disposable Docker container.

    The project directory is mounted read-only at the same path.  If Docker
    is unavailable the original command is returned unchanged so execution
    continues un-sandboxed.
    """
    if not _sandbox_docker_available():
        return command

    cwd_str = str(cwd).replace("\\", "/")
    # fmt: off
    wrapped = (
        f"docker run --rm "
        f"--network none "
        f"--volume {cwd_str}:{cwd_str}:ro "
        f"--workdir {cwd_str} "
        f"--memory 512m --cpus 1 "
        f"{image} "
        f"sh -c {_escape_shell(command)}"
    )
    # fmt: on
    return wrapped


def _escape_shell(command: str) -> str:
    """Single-quote a shell command for use inside ``sh -c``."""
    escaped = command.replace("'", "'\"'\"'")
    return f"'{escaped}'"


def create_bash_tool_definition(
    *,
    cwd: str | Path | None = None,
    sandbox_image: str = "python:3.14-slim",
) -> ToolDefinition:
    """Create the local asynchronous ``bash`` tool.

    When *sandbox_image* is set and the caller passes ``sandbox=true``, the
    command is executed inside a Docker container with the working directory
    mounted read-only.  This is a best-effort safety layer, **not** a
    hardened security boundary.
    """
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        command = _str_arg(arguments, "command")
        timeout = _optional_float_arg(arguments, "timeout")
        use_sandbox = bool(arguments.get("sandbox"))
        if timeout is not None and timeout <= 0:
            raise ToolInputError("timeout must be greater than 0")
        if signal is not None and signal.is_cancelled():
            raise ToolInputError("Command cancelled")

        if use_sandbox:
            command = _wrap_sandbox_command(command, root, sandbox_image)

        start = monotonic()
        if os.name == "posix":
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        else:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

        output_bytes, timed_out, cancelled = await _communicate_with_cancellation(
            process,
            timeout=timeout,
            cancellation_signal=signal,
        )
        output = output_bytes.decode(errors="replace")
        truncation = truncate_tail(output)
        full_output_path: str | None = None
        output_text = truncation.content or "(no output)"

        if truncation.truncated:
            full_output_path = _write_temp_output(output)
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                output_text += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of line "
                    f"{end_line}. Full output: {full_output_path}]"
                )
            elif truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. "
                    f"Full output: {full_output_path}]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_OUTPUT_BYTES)} limit). "
                    f"Full output: {full_output_path}]"
                )

        exit_code = process.returncode
        status: str | None = None
        if timed_out:
            status = (
                f"Command timed out after {timeout:g} seconds" if timeout else "Command timed out"
            )
        elif cancelled:
            status = "Command cancelled"
        elif exit_code not in (0, None):
            status = f"Command exited with code {exit_code}"
        if status is not None:
            output_text = append_status_block(output_text, status)

        ok = exit_code == 0 and not timed_out and not cancelled
        return AgentToolResult(
            tool_call_id="",
            name="bash",
            ok=ok,
            content=output_text,
            error=None if ok else status,
            data={
                "command": command,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "cancelled": cancelled,
                "duration_seconds": round(monotonic() - start, 3),
                "truncation": truncation.to_json(),
                "full_output_path": full_output_path,
            },
        )

    return ToolDefinition(
        name="bash",
        description=(
            "Execute a shell command in the current working directory. stdout and stderr are "
            f"combined; output keeps the last {DEFAULT_MAX_OUTPUT_LINES} lines or "
            f"{DEFAULT_MAX_OUTPUT_BYTES // 1024}KB. Set sandbox=true to run inside a "
            "disposable Docker container with the working directory mounted read-only "
            "(requires Docker). An optional timeout has no default value."
        ),
        prompt_snippet="Execute shell commands, optionally sandboxed with Docker",
        prompt_guidelines=(
            "Use sandbox=true when running untrusted or potentially destructive commands.",
            "Docker must be installed and running for sandbox mode; it falls back to "
            "un-sandboxed execution when unavailable.",
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {
                    "type": "number",
                    "description": "Optional positive timeout in seconds",
                },
                "sandbox": {
                    "type": "boolean",
                    "description": (
                        "Run the command inside a disposable Docker container with "
                        "the working directory mounted read-only."
                    ),
                },
            },
            "required": ["command"],
        },
        executor=execute,
        requires_approval=True,
        auto_approve_if=_bash_command_is_read_only,
    )


def create_bash_tool(
    *,
    cwd: str | Path | None = None,
    sandbox_image: str = "python:3.14-slim",
) -> AgentTool:
    """Create the provider-neutral local ``bash`` tool."""
    return create_bash_tool_definition(cwd=cwd, sandbox_image=sandbox_image).to_agent_tool()


def truncate_head(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> TruncationResult:
    """Keep the beginning of content within line and UTF-8 byte limits."""
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)
    total_bytes = len(content.encode())
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    first_line_bytes = len(lines[0].encode()) if lines else 0
    if first_line_bytes > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            last_line_partial=False,
            first_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by = "lines"
    for index, line in enumerate(lines[:max_lines]):
        line_bytes = len(line.encode()) + (1 if index > 0 else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines.append(line)
        output_bytes += line_bytes

    output = "\n".join(output_lines)
    return TruncationResult(
        content=output,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines),
        output_bytes=len(output.encode()),
        last_line_partial=False,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_tail(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> TruncationResult:
    """Keep the end of content within line and UTF-8 byte limits."""
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)
    total_bytes = len(content.encode())
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by = "lines"
    last_line_partial = False
    for line in reversed(lines):
        line_bytes = len(line.encode()) + (1 if output_lines else 0)
        if len(output_lines) >= max_lines:
            truncated_by = "lines"
            break
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not output_lines:
                clipped = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines.insert(0, clipped)
                output_bytes = len(clipped.encode())
                last_line_partial = True
            break
        output_lines.insert(0, line)
        output_bytes += line_bytes

    output = "\n".join(output_lines)
    return TruncationResult(
        content=output,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines),
        output_bytes=len(output.encode()),
        last_line_partial=last_line_partial,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def format_size(size: int) -> str:
    """Format a byte count for human-facing continuation hints."""
    if size < 1024:
        return f"{size}B"
    if size % 1024 == 0:
        return f"{size // 1024}KB"
    return f"{size / 1024:.1f}KB"


def append_status_block(text: str, status: str) -> str:
    """Append command status after a blank line when output exists."""
    return f"{text}\n\n{status}" if text else status


def _split_lines_for_counting(content: str) -> list[str]:
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    encoded = text.encode()
    if len(encoded) <= max_bytes:
        return text
    return encoded[-max_bytes:].decode(errors="ignore")


def _str_arg(arguments: Mapping[str, JSONValue], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ToolInputError(f"{name} must be a string")
    return value


def _path_arg(arguments: Mapping[str, JSONValue], name: str, *, cwd: Path) -> Path:
    path = Path(_str_arg(arguments, name)).expanduser()
    return path if path.is_absolute() else cwd / path


def _optional_int_arg(arguments: Mapping[str, JSONValue], name: str) -> int | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolInputError(f"{name} must be an integer")
    return value


def _optional_float_arg(arguments: Mapping[str, JSONValue], name: str) -> float | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ToolInputError(f"{name} must be a number")
    return float(value)


def _prepare_edit_arguments(arguments: Mapping[str, JSONValue]) -> Mapping[str, JSONValue]:
    prepared = dict(arguments)
    edits_value = prepared.get("edits")
    if isinstance(edits_value, str):
        try:
            parsed = json.loads(edits_value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            prepared["edits"] = parsed

    old_text = prepared.get("oldText")
    new_text = prepared.get("newText")
    if isinstance(old_text, str) and isinstance(new_text, str):
        edits = prepared.get("edits")
        edit_list = edits if isinstance(edits, list) else []
        prepared["edits"] = [*edit_list, {"oldText": old_text, "newText": new_text}]
        prepared.pop("oldText", None)
        prepared.pop("newText", None)
    return prepared


def _edits_arg(arguments: Mapping[str, JSONValue]) -> list[dict[str, str]]:
    value = arguments.get("edits")
    if not isinstance(value, list) or not value:
        raise ToolInputError("edits must contain at least one replacement")

    edits: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ToolInputError(f"edits[{index}] must be an object")
        old_text = item.get("oldText")
        new_text = item.get("newText")
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ToolInputError(
                f"edits[{index}].oldText and edits[{index}].newText must be strings"
            )
        edits.append({"oldText": old_text, "newText": new_text})
    return edits


def detect_line_ending(content: str) -> str:
    """Return the first newline style found, defaulting to LF."""
    crlf_index = content.find("\r\n")
    lf_index = content.find("\n")
    if lf_index == -1 or crlf_index == -1:
        return "\n"
    return "\r\n" if crlf_index < lf_index else "\n"


def normalize_to_lf(text: str) -> str:
    """Normalize CRLF and bare CR to LF for exact matching."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    """Restore normalized text to the original dominant newline style."""
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def apply_edits_to_normalized_content(
    normalized_content: str,
    edits: list[dict[str, str]],
    path: str,
) -> tuple[str, str]:
    """Validate all exact replacements, then apply them from right to left."""
    normalized_edits = [
        {
            "oldText": normalize_to_lf(edit["oldText"]),
            "newText": normalize_to_lf(edit["newText"]),
        }
        for edit in edits
    ]
    for index, edit in enumerate(normalized_edits):
        if not edit["oldText"]:
            raise ToolInputError(_empty_old_text_error(path, index, len(normalized_edits)))

    matches: list[tuple[int, int, str]] = []
    for index, edit in enumerate(normalized_edits):
        old_text = edit["oldText"]
        occurrences = _count_occurrences(normalized_content, old_text)
        if occurrences == 0:
            raise ToolInputError(_not_found_error(path, index, len(normalized_edits)))
        if occurrences > 1:
            raise ToolInputError(_duplicate_error(path, index, len(normalized_edits), occurrences))
        start = normalized_content.index(old_text)
        matches.append((start, start + len(old_text), edit["newText"]))

    _validate_non_overlapping(matches)
    new_content = normalized_content
    for start, end, new_text in sorted(matches, reverse=True):
        new_content = f"{new_content[:start]}{new_text}{new_content[end:]}"
    if new_content == normalized_content:
        raise ToolInputError(_no_change_error(path, len(normalized_edits)))
    return normalized_content, new_content


def generate_diff_string(old: str, new: str) -> tuple[str, int | None]:
    """Return an ndiff view and the first changed line number."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    diff_lines = list(difflib.ndiff(old_lines, new_lines))
    first_changed_line: int | None = None
    new_line_number = 0
    for line in diff_lines:
        if line.startswith("  "):
            new_line_number += 1
        elif line.startswith("+"):
            new_line_number += 1
            if first_changed_line is None:
                first_changed_line = new_line_number
        elif line.startswith("-") and first_changed_line is None:
            first_changed_line = max(new_line_number + 1, 1)
    return "\n".join(diff_lines), first_changed_line


def generate_unified_patch(path: str, old: str, new: str) -> str:
    """Return a standard unified diff for the replacement."""
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
        )
    )


def _validate_non_overlapping(spans: list[tuple[int, int, str]]) -> None:
    previous_end = -1
    for start, end, _new_text in sorted(spans):
        if start < previous_end:
            raise ToolInputError("Edits must not overlap")
        previous_end = end


def _count_occurrences(content: str, text: str) -> int:
    count = 0
    start = 0
    while True:
        index = content.find(text, start)
        if index == -1:
            return count
        count += 1
        start = index + len(text)


def _strip_bom(content: str) -> tuple[str, str]:
    return (UTF8_BOM, content[1:]) if content.startswith(UTF8_BOM) else ("", content)


def _not_found_error(path: str, edit_index: int, total_edits: int) -> str:
    if total_edits == 1:
        return f"Could not find the exact text in {path}"
    return f"Could not find edits[{edit_index}] in {path}"


def _duplicate_error(path: str, edit_index: int, total_edits: int, occurrences: int) -> str:
    if total_edits == 1:
        return f"Found {occurrences} occurrences of the text in {path}; oldText must be unique"
    return (
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}; "
        "each oldText must be unique"
    )


def _empty_old_text_error(path: str, edit_index: int, total_edits: int) -> str:
    if total_edits == 1:
        return f"oldText must not be empty in {path}"
    return f"edits[{edit_index}].oldText must not be empty in {path}"


def _no_change_error(path: str, total_edits: int) -> str:
    noun = "replacement" if total_edits == 1 else "replacements"
    return f"No changes made to {path}; the {noun} produced identical content"


async def _communicate_with_cancellation(
    process: asyncio.subprocess.Process,
    *,
    timeout: float | None,
    cancellation_signal: ToolCancellationToken | None,
) -> tuple[bytes, bool, bool]:
    communicate = asyncio.create_task(process.communicate())
    cancel_watch: asyncio.Task[None] | None = None
    try:
        wait_for: set[asyncio.Task[Any]] = {communicate}
        if cancellation_signal is not None:
            cancel_watch = asyncio.create_task(_wait_for_cancel(cancellation_signal))
            wait_for.add(cancel_watch)

        done, _pending = await asyncio.wait(
            wait_for,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if communicate in done:
            output_bytes, _stderr = communicate.result()
            return output_bytes, False, False

        cancelled = cancel_watch is not None and cancel_watch in done
        _kill_process_tree(process)
        try:
            output_bytes, _stderr = await communicate
        except asyncio.CancelledError:
            output_bytes = b""
        return output_bytes, not cancelled, cancelled
    except asyncio.CancelledError:
        _kill_process_tree(process)
        if not communicate.done():
            communicate.cancel()
        raise
    finally:
        if cancel_watch is not None:
            cancel_watch.cancel()


async def _wait_for_cancel(cancellation_signal: ToolCancellationToken) -> None:
    while not cancellation_signal.is_cancelled():
        await asyncio.sleep(0.05)


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        try:
            process.kill()
        except ProcessLookupError:
            return


def _write_temp_output(output: str) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="axis-bash-",
        suffix=".log",
        delete=False,
    ) as handle:
        handle.write(output)
        return handle.name


class _FileLockContext:
    def __init__(self, path: Path) -> None:
        self._path = path.resolve()
        self._lock: asyncio.Lock | None = None

    async def __aenter__(self) -> None:
        lock = _file_locks.setdefault(self._path, asyncio.Lock())
        self._lock = lock
        await lock.acquire()

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        if self._lock is not None:
            self._lock.release()


def _file_lock(path: Path) -> _FileLockContext:
    return _FileLockContext(path)
