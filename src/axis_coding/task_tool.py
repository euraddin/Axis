"""Sub-agent delegation tool for Axis coding sessions.

The ``task`` tool spawns an independent sub-agent with a restricted
read-only tool set. The sub-agent has its own context window and
returns a structured result — keeping the parent context lean while
still delegating investigation, search, and analysis work.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from axis_agent.events import ErrorEvent
from axis_agent.harness import AgentHarness, AgentHarnessConfig
from axis_agent.messages import AssistantMessage
from axis_agent.tools import (
    AgentTool,
    AgentToolResult,
    ToolCancellationToken,
)
from axis_agent.types import JSONValue
from axis_ai.provider import CancellationToken, ModelProvider

DEFAULT_TASK_MAX_TURNS = 12
DEFAULT_TASK_CONTEXT_TOKENS = 80_000  # rough cap for sub-agent exploration

_SUBAGENT_SYSTEM = """\
You are a focused sub-agent delegating an isolated task for a parent coding \
agent. Your response will be returned to the parent as a structured result.

Rules:
- Use your tools to investigate thoroughly, then synthesize a clear answer.
- When the task requires reading files, searching the web, or examining git \
history, do it — don't guess.
- Be concise but complete. The parent agent needs actionable information, not \
narration.
- Do NOT ask clarifying questions. Interpret the task as best you can and \
deliver the best answer.
- Your tools are read-only; you cannot modify files or run shell commands.
- Return your final answer as plain text. No JSON wrapper is needed."""


class TaskToolError(ValueError):
    """Invalid task-tool arguments or sub-agent failure."""


async def _run_subagent(
    *,
    provider: ModelProvider,
    model: str,
    prompt: str,
    subagent_tools: list[AgentTool],
    max_turns: int,
    cancellation_signal: CancellationToken | None = None,
) -> AgentToolResult:
    """Run a sub-agent harness to completion and return its final text."""
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=provider,
            model=model,
            system=_SUBAGENT_SYSTEM,
            tools=subagent_tools,
            max_turns=max_turns,
        )
    )

    error: ErrorEvent | None = None
    try:
        async for event in harness.prompt(prompt):
            if cancellation_signal is not None and cancellation_signal.is_cancelled():
                harness.cancel()
            if isinstance(event, ErrorEvent) and not event.recoverable:
                error = event
    except asyncio.CancelledError:
        return AgentToolResult(
            tool_call_id="",
            name="task",
            ok=False,
            content="Sub-agent task was cancelled.",
            error="Cancelled",
        )

    if error is not None:
        return AgentToolResult(
            tool_call_id="",
            name="task",
            ok=False,
            content=f"Sub-agent failed: {error.message}",
            error=error.message,
            data=error.data,
        )

    final_text = _last_assistant_text(harness.messages)
    if not final_text:
        return AgentToolResult(
            tool_call_id="",
            name="task",
            ok=False,
            content="Sub-agent returned an empty response.",
            error="Empty response",
        )

    return AgentToolResult(
        tool_call_id="",
        name="task",
        ok=True,
        content=final_text,
        data={"max_turns": max_turns},
    )


def _last_assistant_text(
    messages: tuple[object, ...],
) -> str:
    for message in reversed(messages):
        if isinstance(message, AssistantMessage) and message.content.strip():
            return message.content.strip()
    return ""


async def _execute_task_tool(
    arguments: Mapping[str, JSONValue],
    *,
    provider: ModelProvider,
    model: str,
    cwd: Path,
    subagent_tools: list[AgentTool],
    cancellation_signal: CancellationToken | None = None,
) -> AgentToolResult:
    prompt = _str_arg(arguments, "prompt")
    if not prompt.strip():
        return AgentToolResult(
            tool_call_id="",
            name="task",
            ok=False,
            content="Task prompt cannot be empty.",
            error="Empty prompt",
        )

    max_turns = _optional_int_arg(arguments, "max_turns") or DEFAULT_TASK_MAX_TURNS
    max_turns = max(1, min(max_turns, 30))

    return await _run_subagent(
        provider=provider,
        model=model,
        prompt=prompt.strip(),
        subagent_tools=subagent_tools,
        max_turns=max_turns,
        cancellation_signal=cancellation_signal,
    )


def _str_arg(arguments: Mapping[str, JSONValue], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise TaskToolError(f"{name} must be a string")
    return value


def _optional_int_arg(arguments: Mapping[str, JSONValue], name: str) -> int | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TaskToolError(f"{name} must be an integer")
    return value


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskToolDefinition:
    """Definition for the ``task`` sub-agent delegation tool."""

    name: str = "task"
    description: str = (
        "Launch a focused sub-agent to investigate, search, or analyze a scoped "
        "question. The sub-agent has a read-only tool set (read, git_status, "
        "git_diff, git_log, web_search, web_fetch) and its own context window. "
        "Use this for research tasks, code exploration, and multi-step investigation "
        "that would bloat the parent context. The sub-agent returns a single "
        "synthesized answer."
    )
    prompt_snippet: str = "Delegate investigation to a read-only sub-agent"
    prompt_guidelines: tuple[str, ...] = (
        "Use the task tool to delegate research, codebase exploration, and web "
        "searches to a sub-agent. This keeps your context window lean.",
        "Give the sub-agent a clear, self-contained prompt with enough context "
        "to work independently.",
        "The sub-agent cannot modify files or run shell commands — use it for "
        "reading and analysis only.",
        "After the sub-agent returns, verify critical findings before acting on them.",
    )
    input_schema: Mapping[str, JSONValue] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "The task for the sub-agent to investigate. Be specific "
                        "and include relevant context (file paths, search terms, etc.)."
                    ),
                },
                "max_turns": {
                    "type": "integer",
                    "description": (
                        f"Maximum tool-calling turns (default {DEFAULT_TASK_MAX_TURNS}, max 30)."
                    ),
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        }
    )
    requires_approval: bool = False

    def to_agent_tool(
        self,
        *,
        provider: ModelProvider,
        model: str,
        cwd: Path,
        subagent_tools: list[AgentTool],
    ) -> AgentTool:
        async def execute(
            arguments: Mapping[str, JSONValue],
            signal: ToolCancellationToken | None = None,
        ) -> AgentToolResult:
            return await _execute_task_tool(
                arguments,
                provider=provider,
                model=model,
                cwd=cwd,
                subagent_tools=subagent_tools,
                cancellation_signal=signal,
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


def create_task_tool(
    *,
    provider: ModelProvider,
    model: str,
    cwd: Path,
    subagent_tools: list[AgentTool] | None = None,
) -> AgentTool:
    """Create the ``task`` sub-agent delegation tool.

    When *subagent_tools* is not provided, a default read-only set is built
    from this package's tool factories.
    """
    if subagent_tools is None:
        from axis_coding.git_tools import (
            create_git_diff_tool,
            create_git_log_tool,
            create_git_status_tool,
        )
        from axis_coding.tools import create_read_tool
        from axis_coding.web_tools import (
            create_web_fetch_tool,
            create_web_search_tool,
        )

        subagent_tools = [
            create_read_tool(cwd=cwd),
            create_git_status_tool(cwd=cwd),
            create_git_diff_tool(cwd=cwd),
            create_git_log_tool(cwd=cwd),
            create_web_search_tool(),
            create_web_fetch_tool(),
        ]

    return TaskToolDefinition().to_agent_tool(
        provider=provider,
        model=model,
        cwd=cwd,
        subagent_tools=subagent_tools,
    )
