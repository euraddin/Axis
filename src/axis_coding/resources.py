"""Axis resource locations and shared Markdown resource primitives."""

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from axis_coding.paths import AxisPaths

PROJECT_MARKERS = (".git", "pyproject.toml", "uv.lock", "setup.py", "package.json")
_RESOURCE_NAME_RE = re.compile(r"[\w][\w.-]*", re.UNICODE)


class ResourceError(ValueError):
    """An Axis resource is invalid or cannot be explicitly expanded."""


@dataclass(frozen=True, slots=True)
class ResourceDiagnostic:
    """A resource that could not be loaded without aborting the session."""

    kind: str
    message: str
    path: Path | None = None
    name: str | None = None
    severity: str = "warning"

    def format(self) -> str:
        parts = [self.severity, self.kind]
        if self.name is not None:
            parts.append(self.name)
        label = " ".join(parts)
        return (
            f"{label}: {self.message}"
            if self.path is None
            else (f"{label}: {self.message} ({self.path})")
        )


@dataclass(frozen=True, slots=True)
class AxisResourcePaths:
    """User and project resource directories in increasing precedence."""

    paths: AxisPaths = field(default_factory=AxisPaths)
    cwd: Path | None = None

    @property
    def project_root(self) -> Path | None:
        return find_project_root(self.cwd) if self.cwd is not None else None

    @property
    def skills_dirs(self) -> tuple[Path, ...]:
        dirs = [self.paths.user_skills_dir, self.paths.user_agents_skills_dir]
        if (root := self.project_root) is not None:
            dirs.extend(
                [
                    self.paths.project_axis_dir(root) / "skills",
                    self.paths.project_agents_dir(root) / "skills",
                ]
            )
        return _dedupe_paths(dirs)

    @property
    def prompts_dirs(self) -> tuple[Path, ...]:
        dirs = [self.paths.user_prompts_dir, self.paths.user_agents_prompts_dir]
        if (root := self.project_root) is not None:
            dirs.extend(
                [
                    self.paths.project_axis_dir(root) / "prompts",
                    self.paths.project_agents_dir(root) / "prompts",
                ]
            )
        return _dedupe_paths(dirs)


def resource_paths_with_cwd(
    paths: AxisResourcePaths | None,
    cwd: Path,
) -> AxisResourcePaths:
    """Bind resource discovery to the resolved coding-session cwd."""
    if paths is None:
        return AxisResourcePaths(cwd=cwd)
    return replace(paths, cwd=cwd)


def find_project_root(cwd: Path) -> Path:
    """Return the nearest marked project root, or cwd when none exists."""
    resolved = cwd.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    return resolved


def parse_markdown_resource(text: str) -> tuple[dict[str, str], str]:
    """Parse dependency-free `key: value` frontmatter and markdown body."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n"):
        return {}, normalized

    lines = normalized.split("\n")
    try:
        closing_line = lines.index("---", 1)
    except ValueError:
        return {}, normalized

    raw_metadata = "\n".join(lines[1:closing_line])
    body = "\n".join(lines[closing_line + 1 :])

    metadata: dict[str, str] = {}
    for line in raw_metadata.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        normalized_key = key.strip()
        if normalized_key:
            metadata[normalized_key] = value.strip().strip("\"'")
    return metadata, body


def derive_markdown_description(content: str) -> str | None:
    """Return the first markdown heading or non-empty line as a description."""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
        return stripped
    return None


def valid_resource_name(name: str) -> bool:
    """Return whether a filename stem is safe and usable as a command token."""
    return _RESOURCE_NAME_RE.fullmatch(name) is not None


def resource_name_key(name: str) -> str:
    """Return the case-insensitive identity used for lookup and precedence."""
    return name.casefold()


def _dedupe_paths(paths: list[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        if expanded in seen:
            continue
        seen.add(expanded)
        result.append(expanded)
    return tuple(result)
