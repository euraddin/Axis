"""Canonical deterministic system-prompt assembly for Axis."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path

from axis_agent import AgentTool
from axis_coding.context import ProjectContextFile
from axis_coding.skills import Skill

AXIS_IDENTITY = (
    "You are Axis, the user's personal coding agent and engineering collaborator. "
    "You operate directly in the user's local development environment. Understand the "
    "user's intent, inspect real code and runtime evidence, make the smallest complete "
    "change that solves the task, verify the result, and report only what is true."
)

OPERATING_PRINCIPLES = (
    "When the user asks for a change, act proactively and complete it instead of stopping "
    "at advice.",
    "Inspect relevant files and runtime evidence before editing, and preserve the user's "
    "existing changes.",
    "Make the smallest complete change that solves the task; avoid unrelated refactors.",
    "Treat tool failures as recoverable evidence: diagnose them and continue when it is safe.",
    "Verify results in proportion to risk, and never claim that an unrun check passed.",
    "Treat explanation, review, and status requests as read-only unless the user also asks "
    "for changes.",
    "Tools run with the user's local permissions and are not a sandbox. Do not discard user "
    "work or run destructive commands unless the user explicitly requests it. Never expose "
    "secrets or write credentials into the repository.",
    "Ask a question only when missing information would materially change the result.",
    "Lead with the outcome, communicate concisely, and show relevant file paths clearly.",
    "Respond in the user's language unless the user requests otherwise.",
)


@dataclass(frozen=True, slots=True)
class BuildSystemPromptOptions:
    """All variable inputs to Axis's canonical system prompt."""

    cwd: Path
    current_date: date
    tools: Sequence[AgentTool] = ()
    skills: Sequence[Skill] = ()
    context_files: Sequence[ProjectContextFile] = ()


def build_system_prompt(options: BuildSystemPromptOptions) -> str:
    """Build Axis's system prompt in one stable section order."""
    sections = [
        _section("identity", AXIS_IDENTITY),
        _section("operating_principles", _format_bullets(OPERATING_PRINCIPLES)),
        _section("available_tools", format_available_tools(options.tools)),
    ]

    guidelines = collect_tool_guidelines(options.tools)
    if guidelines:
        sections.append(_section("tool_guidelines", _format_bullets(guidelines)))

    if _has_tool(options.tools, "read"):
        skills = format_skills_for_prompt(options.skills)
        if skills:
            sections.append(skills)

    project_context = format_project_context(options.context_files)
    if project_context:
        sections.append(project_context)

    cwd = str(options.cwd).replace("\\", "/")
    sections.append(
        _section(
            "environment",
            f"Current date: {options.current_date.isoformat()}\nCurrent working directory: {cwd}",
        )
    )
    return "\n\n".join(sections)


def format_available_tools(tools: Sequence[AgentTool]) -> str:
    """Format user-facing tool summaries in configured order."""
    lines = [f"- {tool.name}: {tool.prompt_snippet}" for tool in tools if tool.prompt_snippet]
    return "\n".join(lines) if lines else "(none)"


def collect_tool_guidelines(tools: Sequence[AgentTool]) -> tuple[str, ...]:
    """Collect tool-derived guidance once in stable order."""
    names = {tool.name for tool in tools}
    guidelines: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        normalized = value.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        guidelines.append(normalized)

    if "bash" in names:
        if names.isdisjoint({"grep", "find", "ls"}):
            add("Use bash for file exploration commands such as ls, rg, and find.")
        else:
            add("Prefer dedicated grep, find, and ls tools over bash for file exploration.")

    for tool in tools:
        for guideline in tool.prompt_guidelines:
            add(guideline)
    return tuple(guidelines)


def format_skills_for_prompt(skills: Sequence[Skill]) -> str:
    """Format a progressive-disclosure skill index without skill bodies."""
    if not skills:
        return ""

    lines = [
        "Skills provide specialized instructions for matching tasks. Read the full skill file "
        "before applying one. Resolve its relative references against the directory containing "
        "the skill file.",
        "<available_skills>",
    ]
    for skill in sorted(skills, key=lambda item: (item.name.casefold(), item.name)):
        description = skill.description or "No description"
        lines.extend(
            [
                "  <skill>",
                f"    <name>{escape(skill.name)}</name>",
                f"    <description>{escape(description)}</description>",
                f"    <location>{escape(str(skill.path))}</location>",
                "  </skill>",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def format_project_context(context_files: Sequence[ProjectContextFile]) -> str:
    """Format discovered instructions in their established precedence order."""
    if not context_files:
        return ""

    lines = [
        "<project_context>",
        "Project instructions are ordered from lower to higher precedence. Later instructions "
        "take precedence when they conflict within their scope.",
    ]
    for context_file in context_files:
        path = escape(str(context_file.path), quote=True)
        lines.extend(
            [
                f'<project_instructions path="{path}">',
                context_file.content,
                "</project_instructions>",
            ]
        )
    lines.append("</project_context>")
    return "\n".join(lines)


def _format_bullets(values: Iterable[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


def _section(name: str, content: str) -> str:
    return f"<{name}>\n{content}\n</{name}>"


def _has_tool(tools: Sequence[AgentTool], name: str) -> bool:
    return any(tool.name == name for tool in tools)
