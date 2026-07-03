"""Build bounded, privacy-conscious context for voice polishing."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from axis_agent import AgentMessage, AssistantMessage, ToolResultMessage, UserMessage

EDITOR_SIDE_LIMIT = 1_000
SUMMARY_LIMIT = 2_000
RECENT_DIALOGUE_LIMIT = 4_000
TOTAL_CONTEXT_LIMIT = 8_000
RECENT_MESSAGE_COUNT = 6
TERM_LIMIT = 80

_BRANCH_PREFIX = (
    "The following is a summary of a branch that this conversation came back from:\n<summary>\n"
)
_SUMMARY_PREFIX = "Previous conversation summary:\n"
_SECRET_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|access[_-]?token|authorization|bearer))\s*[:=]\s*\S+"
)
_TERM_PATTERN = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]{2,}|(?:\.?\.?/)?[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+)"
)


@dataclass(frozen=True, slots=True)
class VoiceContextSnapshot:
    """Immutable context sent only to the one-shot voice polisher."""

    editor_before: str = ""
    editor_after: str = ""
    session_summary: str = ""
    recent_dialogue: str = ""
    coding_metadata: str = ""
    recent_terms: tuple[str, ...] = ()
    tool_activity: str = ""

    @property
    def editor_context(self) -> str:
        return f"Before cursor:\n{self.editor_before}\nAfter cursor:\n{self.editor_after}".strip()

    @property
    def session_memory(self) -> str:
        sections = [value for value in (self.session_summary, self.recent_dialogue) if value]
        return "\n\n".join(sections)

    @property
    def coding_context(self) -> str:
        terms = ", ".join(self.recent_terms)
        sections = [self.coding_metadata, self.tool_activity]
        if terms:
            sections.append(f"Recent technical terms: {terms}")
        return "\n".join(value for value in sections if value)

    @property
    def character_count(self) -> int:
        return sum(
            len(value)
            for value in (
                self.editor_before,
                self.editor_after,
                self.session_summary,
                self.recent_dialogue,
                self.coding_metadata,
                self.tool_activity,
                " ".join(self.recent_terms),
            )
        )


def build_voice_context_snapshot(
    *,
    messages: Sequence[AgentMessage],
    editor_text: str,
    cursor: int,
    cwd: Path,
    session_title: str | None = None,
    skill_names: Sequence[str] = (),
    git_branch: str | None = None,
) -> VoiceContextSnapshot:
    """Build a deterministic current-session-only context snapshot."""
    bounded_cursor = min(max(cursor, 0), len(editor_text))
    before = _sanitize(editor_text[max(0, bounded_cursor - EDITOR_SIDE_LIMIT) : bounded_cursor])
    after = _sanitize(editor_text[bounded_cursor : bounded_cursor + EDITOR_SIDE_LIMIT])

    summaries: list[str] = []
    conversational: list[AgentMessage] = []
    for message in messages:
        if isinstance(message, UserMessage) and (summary := _summary_text(message.content)):
            summaries.append(summary)
        elif isinstance(message, (UserMessage, AssistantMessage)):
            conversational.append(message)

    summary_text = _tail_join(summaries, SUMMARY_LIMIT)
    recent = conversational[-RECENT_MESSAGE_COUNT:]
    dialogue_lines = [
        f"{message.role}: {_sanitize(message.content)}"
        for message in recent
        if message.content.strip()
    ]
    recent_dialogue = _tail_join(dialogue_lines, RECENT_DIALOGUE_LIMIT)

    branch = git_branch if git_branch is not None else current_git_branch(cwd)
    metadata_lines = [f"cwd: {cwd}", f"git branch: {branch or '(none)'}"]
    if session_title:
        metadata_lines.append(f"session title: {_sanitize(session_title)}")
    if skill_names:
        metadata_lines.append(
            "loaded skills: " + ", ".join(_sanitize(name) for name in skill_names if name.strip())
        )
    metadata = "\n".join(metadata_lines)[:1_000]

    tool_activity = _tool_activity(messages)
    term_source = "\n".join((before, after, summary_text, recent_dialogue, metadata, tool_activity))
    terms = _extract_terms(term_source)

    snapshot = VoiceContextSnapshot(
        editor_before=before,
        editor_after=after,
        session_summary=summary_text,
        recent_dialogue=recent_dialogue,
        coding_metadata=metadata,
        recent_terms=terms,
        tool_activity=tool_activity,
    )
    return _fit_total_budget(snapshot)


def current_git_branch(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), "branch", "--show-current"],
            capture_output=True,
            check=False,
            text=True,
            timeout=1,
        )
    except OSError, subprocess.SubprocessError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _summary_text(content: str) -> str | None:
    if content.startswith(_SUMMARY_PREFIX):
        return content.removeprefix(_SUMMARY_PREFIX).strip()
    if content.startswith(_BRANCH_PREFIX) and content.endswith("\n</summary>"):
        return content[len(_BRANCH_PREFIX) : -len("\n</summary>")].strip()
    return None


def _tool_activity(messages: Sequence[AgentMessage]) -> str:
    lines: list[str] = []
    calls: dict[str, tuple[str, str | None]] = {}
    for message in messages:
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                target: str | None = None
                if call.name in {"read", "write", "edit"}:
                    value = call.arguments.get("path")
                    if isinstance(value, str):
                        target = _sanitize(value)[:240]
                calls[call.id] = (call.name, target)
        elif isinstance(message, ToolResultMessage):
            name, target = calls.get(message.tool_call_id, (message.name, None))
            status = "ok" if message.ok else "failed"
            suffix = f" target={target}" if target else ""
            lines.append(f"{name}: {status}{suffix}")
    return "\n".join(lines[-6:])


def _extract_terms(text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _TERM_PATTERN.finditer(text):
        term = match.group(0).strip(".,:;()[]{}<>\"'")
        key = term.casefold()
        if not term or key in seen or _looks_secret(term):
            continue
        seen.add(key)
        ordered.append(term[:120])
    return tuple(ordered[-TERM_LIMIT:])


def _sanitize(text: str) -> str:
    return _SECRET_PATTERN.sub(r"\1=[REDACTED]", text).strip()


def _looks_secret(term: str) -> bool:
    return len(term) >= 28 and term.isalnum()


def _tail_join(values: Sequence[str], limit: int) -> str:
    joined = "\n\n".join(value for value in values if value)
    return joined[-limit:]


def _fit_total_budget(snapshot: VoiceContextSnapshot) -> VoiceContextSnapshot:
    overflow = snapshot.character_count - TOTAL_CONTEXT_LIMIT
    if overflow <= 0:
        return snapshot
    dialogue = snapshot.recent_dialogue
    trim = min(overflow, len(dialogue))
    dialogue = dialogue[trim:]
    overflow -= trim
    summary = snapshot.session_summary
    if overflow > 0:
        trim = min(overflow, len(summary))
        summary = summary[trim:]
        overflow -= trim
    tools = snapshot.tool_activity
    if overflow > 0:
        trim = min(overflow, len(tools))
        tools = tools[trim:]
        overflow -= trim
    terms = list(snapshot.recent_terms)
    while overflow > 0 and terms:
        overflow -= len(terms.pop(0)) + 1
    metadata = snapshot.coding_metadata
    if overflow > 0:
        metadata = metadata[min(overflow, len(metadata)) :]
    return VoiceContextSnapshot(
        editor_before=snapshot.editor_before,
        editor_after=snapshot.editor_after,
        session_summary=summary,
        recent_dialogue=dialogue,
        coding_metadata=metadata,
        recent_terms=tuple(terms),
        tool_activity=tools,
    )
