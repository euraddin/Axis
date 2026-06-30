"""Discover and explicitly expand local Markdown skills."""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path

from axis_coding.resources import (
    AxisResourcePaths,
    ResourceDiagnostic,
    ResourceError,
    derive_markdown_description,
    parse_markdown_resource,
    resource_name_key,
    valid_resource_name,
)

_SKILL_COMMAND_RE = re.compile(r"^/skill:([^\s]*)(?:\s+([\s\S]*))?$")


@dataclass(frozen=True, slots=True)
class Skill:
    """One named Markdown skill and its absolute source location."""

    name: str
    path: Path
    content: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class SkillInvocation:
    """One expanded skill prompt recovered from the durable transcript."""

    name: str
    location: str
    content: str
    additional_instructions: str | None = None


def load_skills(paths: AxisResourcePaths | None = None) -> tuple[Skill, ...]:
    """Load skills, raising when a resource directory contains invalid entries."""
    resource_paths = paths if paths is not None else AxisResourcePaths()
    skills_by_name: dict[str, Skill] = {}
    for skills_dir in resource_paths.skills_dirs:
        skills, diagnostics = _discover_skills_dir(skills_dir)
        if diagnostics:
            raise ResourceError(diagnostics[0].format())
        for skill in skills:
            skills_by_name[resource_name_key(skill.name)] = skill
    return _sorted_skills(skills_by_name.values())


def load_skills_with_diagnostics(
    paths: AxisResourcePaths | None = None,
) -> tuple[tuple[Skill, ...], tuple[ResourceDiagnostic, ...]]:
    """Load usable skills while preserving errors and precedence decisions."""
    resource_paths = paths if paths is not None else AxisResourcePaths()
    skills_by_name: dict[str, Skill] = {}
    diagnostics: list[ResourceDiagnostic] = []
    for skills_dir in resource_paths.skills_dirs:
        skills, directory_diagnostics = _discover_skills_dir(skills_dir)
        diagnostics.extend(directory_diagnostics)
        for skill in skills:
            key = resource_name_key(skill.name)
            previous = skills_by_name.get(key)
            if previous is not None:
                diagnostics.append(
                    ResourceDiagnostic(
                        kind="skill",
                        name=skill.name,
                        path=skill.path,
                        message=f"overrides lower-precedence resource at {previous.path}",
                    )
                )
            skills_by_name[key] = skill
    return _sorted_skills(skills_by_name.values()), tuple(diagnostics)


def expand_skill_command(text: str, skills: Sequence[Skill]) -> str | None:
    """Expand `/skill:name [instructions]`, or return None for normal prompts."""
    stripped = text.strip()
    if not stripped.startswith("/skill:"):
        return None

    match = _SKILL_COMMAND_RE.fullmatch(stripped)
    if match is None:
        return None
    name, instructions = match.groups()
    if not name:
        raise ResourceError("Skill command must include a skill name")

    skill = {resource_name_key(item.name): item for item in skills}.get(resource_name_key(name))
    if skill is None:
        raise ResourceError(f"Unknown skill: {name}")
    return format_skill_invocation(skill, instructions)


def format_skill_invocation(
    skill: Skill,
    additional_instructions: str | None = None,
) -> str:
    """Embed a skill with an explicit base directory for relative references."""
    name = escape(skill.name, quote=True)
    location = escape(str(skill.path), quote=True)
    block = (
        f'<skill name="{name}" location="{location}">\n'
        f"References are relative to {skill.path.parent}.\n\n"
        f"{skill.content.strip()}\n"
        "</skill>"
    )
    if additional_instructions and additional_instructions.strip():
        return f"{block}\n\n{additional_instructions.strip()}"
    return block


def parse_skill_invocation(text: str) -> SkillInvocation | None:
    """Parse the exact expanded skill shape produced by Axis."""
    match = re.fullmatch(
        r'<skill name="([^"]+)" location="([^"]+)">\n([\s\S]*?)\n</skill>'
        r"(?:\n\n([\s\S]+))?",
        text,
    )
    if match is None:
        return None
    name, location, content, additional_instructions = match.groups()
    return SkillInvocation(
        name=name,
        location=location,
        content=content,
        additional_instructions=additional_instructions,
    )


def _discover_skills_dir(
    skills_dir: Path,
) -> tuple[tuple[Skill, ...], tuple[ResourceDiagnostic, ...]]:
    directory = skills_dir.expanduser().resolve()
    if not directory.exists() or not directory.is_dir():
        return (), ()

    try:
        entries = sorted(directory.iterdir(), key=lambda path: (path.name.casefold(), path.name))
    except OSError as exc:
        return (), (
            ResourceDiagnostic(
                kind="skill",
                path=directory,
                message=f"could not list skill directory: {exc}",
                severity="error",
            ),
        )

    skills: list[Skill] = []
    diagnostics: list[ResourceDiagnostic] = []
    seen: set[str] = set()
    for entry in entries:
        candidate = _skill_candidate(entry)
        if candidate is None:
            continue
        name, path = candidate
        key = resource_name_key(name)
        if not valid_resource_name(name):
            diagnostics.append(
                ResourceDiagnostic(
                    kind="skill",
                    name=name,
                    path=path,
                    message=(
                        "invalid resource name; use letters, numbers, dots, dashes or underscores"
                    ),
                    severity="error",
                )
            )
            continue
        if key in seen:
            diagnostics.append(
                ResourceDiagnostic(
                    kind="skill",
                    name=name,
                    path=path,
                    message=f"duplicate skill name ignored in {directory}",
                )
            )
            continue
        seen.add(key)
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            diagnostics.append(
                ResourceDiagnostic(
                    kind="skill",
                    name=name,
                    path=path,
                    message=f"could not read skill: {exc}",
                    severity="error",
                )
            )
            continue
        metadata, content = parse_markdown_resource(raw)
        skills.append(
            Skill(
                name=name,
                path=path,
                content=content,
                description=metadata.get("description") or derive_markdown_description(content),
            )
        )
    return tuple(skills), tuple(diagnostics)


def _skill_candidate(entry: Path) -> tuple[str, Path] | None:
    if entry.is_dir():
        path = entry / "SKILL.md"
        return (entry.name, path) if path.is_file() else None
    if not entry.is_file() or entry.suffix.casefold() != ".md":
        return None
    if entry.name.casefold() == "agents.md":
        return None
    return entry.stem, entry


def _sorted_skills(skills: Iterable[Skill]) -> tuple[Skill, ...]:
    return tuple(sorted(skills, key=lambda skill: (skill.name.casefold(), skill.name)))
