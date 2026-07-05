"""MCP server configuration — load, merge, and validate user and project settings."""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from axis_coding.resources import AxisResourcePaths, ResourceDiagnostic

MCP_CONFIG_FILENAME = "mcp.json"
_MAX_NAME_LENGTH = 64
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
_ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


class McpServerConfig(BaseModel):
    """One stdio-based MCP server definition."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class McpConfig(BaseModel):
    """Complete named MCP server configuration."""

    model_config = ConfigDict(extra="forbid")

    servers: dict[str, McpServerConfig] = Field(default_factory=dict)


def load_mcp_config(
    paths: AxisResourcePaths,
    *,
    cwd: Path | None = None,
) -> tuple[McpConfig, tuple[ResourceDiagnostic, ...]]:
    """Load user-level then project-level MCP configs, merging by server name.

    Project-level configs override user-level servers with the same name.
    Returns the merged config and any non-fatal diagnostics.
    """
    diagnostics: list[ResourceDiagnostic] = []
    merged = McpConfig()

    # User-level config
    user_config, user_diags = _load_config_file(paths.paths.home / MCP_CONFIG_FILENAME)
    diagnostics.extend(user_diags)
    merged = _merge_configs(merged, user_config)

    # Project-level config
    if cwd is not None:
        root = paths.project_root or cwd
        project_path = paths.paths.project_axis_dir(root) / MCP_CONFIG_FILENAME
        project_config, project_diags = _load_config_file(project_path)
        diagnostics.extend(project_diags)
        merged = _merge_configs(merged, project_config)

    # Validate server names
    for name in list(merged.servers):
        if not _valid_server_name(name):
            diagnostics.append(
                ResourceDiagnostic(
                    kind="mcp",
                    name=name,
                    message=(
                        f"Invalid MCP server name '{name}'; must be 1-{_MAX_NAME_LENGTH} "
                        "characters, start with alphanumeric, and contain only "
                        "alphanumeric, dot, dash, or underscore"
                    ),
                    severity="warning",
                )
            )
            merged.servers.pop(name)

    return merged, tuple(diagnostics)


def _load_config_file(path: Path) -> tuple[McpConfig, tuple[ResourceDiagnostic, ...]]:
    """Load and parse one MCP JSON configuration file."""
    if not path.is_file():
        return McpConfig(), ()

    if path.is_symlink():
        return McpConfig(), (
            ResourceDiagnostic(
                kind="mcp",
                path=path,
                message=f"MCP config symlinks are not supported: {path}",
                severity="warning",
            ),
        )

    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return McpConfig(), (
            ResourceDiagnostic(
                kind="mcp",
                path=path,
                message=f"Could not read MCP config file: {exc}",
                severity="warning",
            ),
        )

    try:
        config = McpConfig.model_validate_json(raw)
    except Exception as exc:
        return McpConfig(), (
            ResourceDiagnostic(
                kind="mcp",
                path=path,
                message=f"Invalid MCP config: {exc}",
                severity="warning",
            ),
        )

    return config, ()


def _merge_configs(base: McpConfig, overlay: McpConfig) -> McpConfig:
    """Merge two configs; overlay servers replace base servers by name."""
    if not overlay.servers:
        return base
    merged = dict(base.servers)
    merged.update(overlay.servers)
    return McpConfig(servers=merged)


def expand_env_vars(config: McpServerConfig) -> McpServerConfig:
    """Expand ``${VAR}`` patterns in environment values against ``os.environ``."""
    expanded_env: dict[str, str] = {}
    for key, value in config.env.items():
        resolved = _ENV_VAR_RE.sub(
            lambda match: os.environ.get(match.group(1), match.group(0)),
            value,
        )
        expanded_env[key] = resolved
    return config.model_copy(update={"env": expanded_env})


def _valid_server_name(name: str) -> bool:
    if not name or len(name) > _MAX_NAME_LENGTH:
        return False
    return bool(_NAME_RE.match(name))
