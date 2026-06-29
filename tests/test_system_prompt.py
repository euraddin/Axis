"""Tests for Axis's canonical deterministic system prompt."""

from collections.abc import Mapping
from datetime import date
from pathlib import Path

from axis_agent import AgentTool, AgentToolResult
from axis_agent.types import JSONValue
from axis_coding import (
    BuildSystemPromptOptions,
    ProjectContextFile,
    Skill,
    build_system_prompt,
    collect_tool_guidelines,
    create_coding_tools,
    format_available_tools,
    format_project_context,
    format_skills_for_prompt,
)


async def unused_executor(
    arguments: Mapping[str, JSONValue],
    signal: object | None = None,
) -> AgentToolResult:
    del arguments, signal
    return AgentToolResult(
        tool_call_id="",
        name="unused",
        ok=True,
        content="",
    )


def test_base_prompt_has_an_exact_stable_shape(tmp_path: Path) -> None:
    prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=tmp_path,
            current_date=date(2026, 6, 29),
        )
    )

    assert prompt == (
        "<identity>\n"
        "You are Axis, the user's personal coding agent and engineering collaborator. "
        "You operate directly in the user's local development environment. Understand the "
        "user's intent, inspect real code and runtime evidence, make the smallest complete "
        "change that solves the task, verify the result, and report only what is true.\n"
        "</identity>\n\n"
        "<operating_principles>\n"
        "- When the user asks for a change, act proactively and complete it instead of "
        "stopping at advice.\n"
        "- Inspect relevant files and runtime evidence before editing, and preserve the "
        "user's existing changes.\n"
        "- Make the smallest complete change that solves the task; avoid unrelated refactors.\n"
        "- Treat tool failures as recoverable evidence: diagnose them and continue when it "
        "is safe.\n"
        "- Verify results in proportion to risk, and never claim that an unrun check passed.\n"
        "- Treat explanation, review, and status requests as read-only unless the user also "
        "asks for changes.\n"
        "- Tools run with the user's local permissions and are not a sandbox. Do not discard "
        "user work or run destructive commands unless the user explicitly requests it. Never "
        "expose secrets or write credentials into the repository.\n"
        "- Ask a question only when missing information would materially change the result.\n"
        "- Lead with the outcome, communicate concisely, and show relevant file paths clearly.\n"
        "- Respond in the user's language unless the user requests otherwise.\n"
        "</operating_principles>\n\n"
        "<available_tools>\n"
        "(none)\n"
        "</available_tools>\n\n"
        "<environment>\n"
        "Current date: 2026-06-29\n"
        f"Current working directory: {tmp_path}\n"
        "</environment>"
    )


def test_all_dynamic_sections_use_the_declared_order(tmp_path: Path) -> None:
    skill = Skill(
        name="review",
        path=tmp_path / "skills" / "review" / "SKILL.md",
        content="SECRET SKILL BODY",
        description="Review code safely",
    )
    context = ProjectContextFile(
        path=tmp_path / "AGENTS.md",
        content="Use project rules.",
    )
    options = BuildSystemPromptOptions(
        cwd=tmp_path,
        tools=create_coding_tools(cwd=tmp_path),
        skills=[skill],
        context_files=[context],
        current_date=date(2026, 6, 29),
    )

    first = build_system_prompt(options)
    second = build_system_prompt(options)

    assert first == second
    section_positions = [
        first.index(f"<{name}>")
        for name in (
            "identity",
            "operating_principles",
            "available_tools",
            "tool_guidelines",
            "available_skills",
            "project_context",
            "environment",
        )
    ]
    assert section_positions == sorted(section_positions)
    assert "SECRET SKILL BODY" not in first
    assert f"<location>{skill.path}</location>" in first
    assert "Use project rules." in first


def test_available_tools_hide_tools_without_prompt_snippets() -> None:
    hidden = AgentTool(
        name="hidden",
        description="Provider-visible only",
        input_schema={"type": "object"},
        executor=unused_executor,
    )

    assert format_available_tools([hidden]) == "(none)"


def test_tool_guidelines_are_derived_and_deduplicated(tmp_path: Path) -> None:
    tools = create_coding_tools(cwd=tmp_path)
    duplicate = AgentTool(
        name="duplicate",
        description="Duplicate guidance",
        input_schema={"type": "object"},
        executor=unused_executor,
        prompt_guidelines=(tools[0].prompt_guidelines[0],),
    )

    guidelines = collect_tool_guidelines([*tools, duplicate])

    assert guidelines[0] == "Use bash for file exploration commands such as ls, rg, and find."
    assert guidelines.count(tools[0].prompt_guidelines[0]) == 1


def test_skills_are_escaped_and_omitted_without_read_tool(tmp_path: Path) -> None:
    skill = Skill(
        name="review&check",
        path=tmp_path / "skills" / "review&check.md",
        content="body must not appear",
        description="Review <code>",
    )

    formatted = format_skills_for_prompt([skill])
    prompt_without_read = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=tmp_path,
            skills=[skill],
            current_date=date(2026, 6, 29),
        )
    )

    assert "<name>review&amp;check</name>" in formatted
    assert "<description>Review &lt;code&gt;</description>" in formatted
    assert "body must not appear" not in formatted
    assert "<available_skills>" not in prompt_without_read


def test_project_context_preserves_order_and_escapes_path_attributes(
    tmp_path: Path,
) -> None:
    first = ProjectContextFile(path=tmp_path / 'a&"b.md', content="general")
    second = ProjectContextFile(path=tmp_path / "nested.md", content="specific")

    formatted = format_project_context([first, second])

    assert formatted.index("general") < formatted.index("specific")
    assert 'path="' in formatted
    assert "a&amp;&quot;b.md" in formatted
    assert "Later instructions take precedence" in formatted
