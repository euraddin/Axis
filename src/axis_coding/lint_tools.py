"""Code-review lint tool for Axis coding sessions.

Runs the project's configured linter and returns structured results
that the agent can inspect and act on.
"""

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

DEFAULT_LINT_TIMEOUT_SECONDS = 60.0
MAX_LINT_OUTPUT_BYTES = 100_000


class LintToolError(ValueError):
    """A lint tool received invalid arguments or the linter failed."""


def _str_arg(arguments: Mapping[str, JSONValue], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise LintToolError(f"{name} must be a string")
    return value


def _optional_str_arg(arguments: Mapping[str, JSONValue], name: str) -> str | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise LintToolError(f"{name} must be a string")
    stripped = value.strip()
    return stripped if stripped else None


def _detect_python_linter(cwd: Path) -> tuple[str, list[str]]:
    """Inspect project configuration and return ``(command, base_args)``.

    Checks for common Python lint tools in order of preference.
    """
    candidates = [
        (["ruff", "check"], _has_pyproject_tool(cwd, "ruff")),
        # ruff is preferred; mypy is type-only, not lint.
        (["pylint"], _has_config(cwd, ".pylintrc")),
    ]
    for args, exists in candidates:
        if exists:
            return (args[0], args[1:])
    # Fallback: just use ruff if installed.
    return ("ruff", ["check"])


def _has_pyproject_tool(cwd: Path, tool_name: str) -> bool:
    pyproject = cwd / "pyproject.toml"
    if not pyproject.exists():
        return False
    try:
        text = pyproject.read_text()
        # Simple heuristic: the tool name appears as a TOML section or key.
        return f"[tool.{tool_name}]" in text or f"[tool.{tool_name}." in text
    except OSError:
        return False


def _has_config(cwd: Path, filename: str) -> bool:
    return (cwd / filename).exists()


async def _run_linter(
    *,
    cwd: Path,
    command: str,
    args: list[str],
    timeout: float = DEFAULT_LINT_TIMEOUT_SECONDS,
) -> tuple[str, int | None, bool]:
    """Run a linter subprocess and return (output, exit_code, timed_out)."""
    try:
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        return (
            (f"Linter not found: {command}. Install it or configure a different linter."),
            None,
            False,
        )

    try:
        stdout_bytes, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        return f"Linter timed out after {timeout} seconds.", None, True

    output = stdout_bytes.decode(errors="replace").strip()
    return output, process.returncode, False


def _parse_ruff_output(output: str) -> dict[str, JSONValue]:
    """Extract summary info from ruff's text output."""
    lines = output.split("\n")
    errors: list[JSONValue] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # ruff lines look like: "src/file.py:10:5: F841 ..."
        if ":" in stripped:
            parts = stripped.split(":", 3)
            if len(parts) >= 4:
                errors.append(
                    {
                        "file": parts[0],
                        "line": parts[1],
                        "col": parts[2],
                        "message": parts[3].strip(),
                    }
                )

    fixed_hint = ""
    if "fixable with" in output.lower() or "auto-fix" in output.lower():
        fixed_hint = (
            "Run `ruff check --fix .` (or the equivalent for your linter) "
            "to auto-fix the flagged issues."
        )

    return {
        "total_errors": len(errors),
        "errors": errors[:200],  # Cap at 200 individual errors.
        "fix_hint": fixed_hint,
    }


# ---------------------------------------------------------------------------
# lint tool
# ---------------------------------------------------------------------------


async def _execute_lint(
    arguments: Mapping[str, JSONValue],
    *,
    cwd: Path,
) -> AgentToolResult:
    del arguments  # No user arguments — auto-detects from project config.
    command, base_args = _detect_python_linter(cwd)
    output, exit_code, timed_out = await _run_linter(cwd=cwd, command=command, args=base_args)

    ok = exit_code == 0 and not timed_out
    parsed = _parse_ruff_output(output) if "ruff" in command else {}

    lines = [
        f"# Lint Results: {command}",
        "",
        f"**Exit code:** {exit_code}",
        f"**Issues found:** {parsed.get('total_errors', 'unknown')}",
    ]
    if parsed.get("fix_hint"):
        lines.append(f"\n{parsed['fix_hint']}")
    if output:
        truncated = (
            output[:MAX_LINT_OUTPUT_BYTES] if len(output) > MAX_LINT_OUTPUT_BYTES else output
        )
        lines.append(f"\n```text\n{truncated}\n```")
        if len(output) > MAX_LINT_OUTPUT_BYTES:
            lines.append("\n[Output truncated]")

    return AgentToolResult(
        tool_call_id="",
        name="lint",
        ok=ok,
        content="\n".join(lines),
        data={
            "command": command,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "total_errors": parsed.get("total_errors"),
        },
    )


@dataclass(frozen=True, slots=True)
class LintToolDefinition:
    """Definition for the ``lint`` tool."""

    name: str = "lint"
    description: str = (
        "Run the project's configured linter (auto-detected from pyproject.toml "
        "or config files) and return structured results with file paths, line "
        "numbers, and error messages. Supports Python (ruff, pylint)."
    )
    prompt_snippet: str = "Run the project linter and return structured results"
    prompt_guidelines: tuple[str, ...] = (
        "Run lint before committing changes to catch style and correctness issues.",
        "After lint reports issues, fix them and re-run lint to verify.",
        "Prefer the lint tool over shelling out to ruff/pylint directly for structured output.",
    )
    input_schema: Mapping[str, JSONValue] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    )
    requires_approval: bool = False

    def to_agent_tool(self, *, cwd: Path) -> AgentTool:
        async def execute(
            arguments: Mapping[str, JSONValue],
            signal: ToolCancellationToken | None = None,
        ) -> AgentToolResult:
            del signal
            return await _execute_lint(arguments, cwd=cwd)

        return AgentTool(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            executor=execute,
            requires_approval=self.requires_approval,
            prompt_snippet=self.prompt_snippet,
            prompt_guidelines=self.prompt_guidelines,
        )


def create_lint_tool(*, cwd: Path) -> AgentTool:
    """Create the ``lint`` tool bound to *cwd*."""
    return LintToolDefinition().to_agent_tool(cwd=cwd)
