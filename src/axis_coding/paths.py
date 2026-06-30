"""Canonical user and project filesystem locations for Axis."""

import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class AxisPaths:
    """Resolved roots for durable Axis data and shared agent resources."""

    home: Path = field(default_factory=lambda: Path.home() / ".axis")
    agents_home: Path = field(default_factory=lambda: Path.home() / ".agents")

    @property
    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    @property
    def user_skills_dir(self) -> Path:
        return self.home / "skills"

    @property
    def user_prompts_dir(self) -> Path:
        return self.home / "prompts"

    @property
    def user_agents_skills_dir(self) -> Path:
        return self.agents_home / "skills"

    @property
    def user_agents_prompts_dir(self) -> Path:
        return self.agents_home / "prompts"

    def project_axis_dir(self, project_root: Path) -> Path:
        return project_root / ".axis"

    def project_agents_dir(self, project_root: Path) -> Path:
        return project_root / ".agents"

    def project_session_dir(self, cwd: Path) -> Path:
        """Return a stable user-home session directory for one cwd."""
        resolved = cwd.expanduser().resolve()
        digest = sha256(str(resolved).encode()).hexdigest()[:8]
        slug = _slugify_path(resolved)
        return self.sessions_dir / f"{slug or 'project'}-{digest}"

    def session_path(self, cwd: Path, session_id: str) -> Path:
        """Return a session path without creating any directory."""
        if not re.fullmatch(r"[a-zA-Z0-9._-]+", session_id):
            raise ValueError(f"Invalid session id: {session_id}")
        return self.project_session_dir(cwd) / f"{session_id}.jsonl"

    def new_session_path(self, cwd: Path) -> Path:
        """Return a fresh session path without touching the filesystem."""
        return self.session_path(cwd, uuid4().hex)

    def default_session_path(self, cwd: Path) -> Path:
        """Return the stable default-session file for one working directory."""
        return self.project_session_dir(cwd) / "default.jsonl"


def _slugify_path(path: Path, *, max_length: int = 72) -> str:
    parts = [part for part in path.parts if part not in {path.anchor, ""}]
    try:
        relative_to_home = path.relative_to(Path.home())
    except ValueError:
        pass
    else:
        parts = ["home", *relative_to_home.parts]

    normalized = [
        slug
        for part in parts
        if (slug := re.sub(r"[^a-zA-Z0-9._-]+", "-", part).strip(".-_").lower())
    ]
    joined = "-".join(normalized)
    if len(joined) <= max_length:
        return joined
    return joined[-max_length:].strip("-")
