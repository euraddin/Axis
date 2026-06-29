"""Discover, render and explicitly expand Markdown prompt templates."""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
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

_VARIABLE_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")
_ARGUMENT_VARIABLES = frozenset({"arguments", "args"})
_COMMAND_RE = re.compile(r"^/([^\s]+)(?:\s+([\s\S]*))?$")


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """One named Markdown prompt template."""

    name: str
    path: Path
    content: str
    description: str | None = None


def load_prompt_templates(
    paths: AxisResourcePaths | None = None,
) -> tuple[PromptTemplate, ...]:
    """Load templates, raising for invalid entries within a resource directory."""
    resource_paths = paths if paths is not None else AxisResourcePaths()
    templates_by_name: dict[str, PromptTemplate] = {}
    for prompts_dir in resource_paths.prompts_dirs:
        templates, diagnostics = _discover_prompt_templates_dir(prompts_dir)
        if diagnostics:
            raise ResourceError(diagnostics[0].format())
        for template in templates:
            templates_by_name[resource_name_key(template.name)] = template
    return _sorted_templates(templates_by_name.values())


def load_prompt_templates_with_diagnostics(
    paths: AxisResourcePaths | None = None,
) -> tuple[tuple[PromptTemplate, ...], tuple[ResourceDiagnostic, ...]]:
    """Load usable templates while preserving errors and precedence decisions."""
    resource_paths = paths if paths is not None else AxisResourcePaths()
    templates_by_name: dict[str, PromptTemplate] = {}
    diagnostics: list[ResourceDiagnostic] = []
    for prompts_dir in resource_paths.prompts_dirs:
        templates, directory_diagnostics = _discover_prompt_templates_dir(prompts_dir)
        diagnostics.extend(directory_diagnostics)
        for template in templates:
            key = resource_name_key(template.name)
            previous = templates_by_name.get(key)
            if previous is not None:
                diagnostics.append(
                    ResourceDiagnostic(
                        kind="prompt",
                        name=template.name,
                        path=template.path,
                        message=f"overrides lower-precedence resource at {previous.path}",
                    )
                )
            templates_by_name[key] = template
    return _sorted_templates(templates_by_name.values()), tuple(diagnostics)


def render_prompt_template(
    template: PromptTemplate,
    variables: Mapping[str, str],
    *,
    missing: str | None = None,
) -> str:
    """Replace `{{ variable }}` placeholders with explicit values."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        value = variables.get(name)
        if value is not None:
            return value
        if missing is not None:
            return missing
        raise ResourceError(f"Missing prompt template variable: {name}")

    return _VARIABLE_RE.sub(replace, template.content)


def expand_prompt_template_command(
    text: str,
    templates: Sequence[PromptTemplate],
) -> str | None:
    """Expand `/name [arguments]`, or return None when no template matches."""
    stripped = text.strip()
    if not stripped.startswith("/") or stripped.startswith(("//", "/skill:")):
        return None
    match = _COMMAND_RE.fullmatch(stripped)
    if match is None:
        return None
    name, arguments = match.groups()
    args = arguments.strip() if arguments is not None else ""
    template = {resource_name_key(item.name): item for item in templates}.get(
        resource_name_key(name)
    )
    if template is None:
        return None

    rendered = render_prompt_template(
        template,
        {"arguments": args, "args": args},
        missing="",
    )
    if args and not _references_arguments(template.content):
        return f"{rendered.rstrip()}\n\n{args}"
    return rendered


def _references_arguments(content: str) -> bool:
    return any(match.group(1) in _ARGUMENT_VARIABLES for match in _VARIABLE_RE.finditer(content))


def _discover_prompt_templates_dir(
    prompts_dir: Path,
) -> tuple[tuple[PromptTemplate, ...], tuple[ResourceDiagnostic, ...]]:
    directory = prompts_dir.expanduser().resolve()
    if not directory.exists() or not directory.is_dir():
        return (), ()

    try:
        entries = sorted(directory.iterdir(), key=lambda path: (path.name.casefold(), path.name))
    except OSError as exc:
        return (), (
            ResourceDiagnostic(
                kind="prompt",
                path=directory,
                message=f"could not list prompt directory: {exc}",
                severity="error",
            ),
        )

    templates: list[PromptTemplate] = []
    diagnostics: list[ResourceDiagnostic] = []
    seen: set[str] = set()
    for path in entries:
        if not path.is_file() or path.suffix.casefold() != ".md":
            continue
        name = path.stem
        key = resource_name_key(name)
        if not valid_resource_name(name):
            diagnostics.append(
                ResourceDiagnostic(
                    kind="prompt",
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
                    kind="prompt",
                    name=name,
                    path=path,
                    message=f"duplicate prompt template name ignored in {directory}",
                )
            )
            continue
        seen.add(key)
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            diagnostics.append(
                ResourceDiagnostic(
                    kind="prompt",
                    name=name,
                    path=path,
                    message=f"could not read prompt template: {exc}",
                    severity="error",
                )
            )
            continue
        metadata, content = parse_markdown_resource(raw)
        templates.append(
            PromptTemplate(
                name=name,
                path=path,
                content=content,
                description=metadata.get("description") or derive_markdown_description(content),
            )
        )
    return tuple(templates), tuple(diagnostics)


def _sorted_templates(
    templates: Iterable[PromptTemplate],
) -> tuple[PromptTemplate, ...]:
    return tuple(sorted(templates, key=lambda template: (template.name.casefold(), template.name)))
