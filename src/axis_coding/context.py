"""Discover hierarchical AGENTS.md instructions for Axis sessions."""

from dataclasses import dataclass
from pathlib import Path

from axis_coding.resources import AxisResourcePaths, ResourceDiagnostic


@dataclass(frozen=True, slots=True)
class ProjectContextFile:
    """One instruction file and its exact UTF-8 content."""

    path: Path
    content: str


def discover_project_context(
    paths: AxisResourcePaths,
) -> tuple[ProjectContextFile, ...]:
    """Return readable instruction files in increasing precedence."""
    context_files, _diagnostics = discover_project_context_with_diagnostics(paths)
    return context_files


def discover_project_context_with_diagnostics(
    paths: AxisResourcePaths,
) -> tuple[tuple[ProjectContextFile, ...], tuple[ResourceDiagnostic, ...]]:
    """Return readable instructions plus non-fatal read diagnostics."""
    context_files: list[ProjectContextFile] = []
    diagnostics: list[ResourceDiagnostic] = []
    for path in _context_file_candidates(paths):
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            diagnostics.append(
                ResourceDiagnostic(
                    kind="context",
                    message=f"could not read context file: {exc}",
                    path=path,
                )
            )
            continue
        context_files.append(ProjectContextFile(path=path, content=content))
    return tuple(context_files), tuple(diagnostics)


def _context_file_candidates(paths: AxisResourcePaths) -> tuple[Path, ...]:
    candidates = [
        paths.paths.home / "AGENTS.md",
        paths.paths.agents_home / "AGENTS.md",
    ]
    if paths.cwd is not None:
        cwd = paths.cwd.expanduser().resolve()
        root = paths.project_root or cwd
        candidates.extend(_ancestor_agents_files(root, cwd))
        candidates.extend(
            [
                paths.paths.project_axis_dir(root) / "AGENTS.md",
                paths.paths.project_agents_dir(root) / "AGENTS.md",
            ]
        )
    return tuple(path for path in _dedupe_resolved_paths(candidates) if path.is_file())


def _ancestor_agents_files(project_root: Path, cwd: Path) -> list[Path]:
    relative = cwd.relative_to(project_root)
    paths = [project_root / "AGENTS.md"]
    current = project_root
    for part in relative.parts:
        current /= part
        paths.append(current / "AGENTS.md")
    return paths


def _dedupe_resolved_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return result
