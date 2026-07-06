"""Tests for the MCP (Model Context Protocol) integration package."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from axis_agent.tools import AgentTool
from axis_coding.mcp.config import (
    MCP_CONFIG_FILENAME,
    McpConfig,
    McpServerConfig,
    expand_env_vars,
    load_mcp_config,
)
from axis_coding.mcp.manager import McpManager
from axis_coding.mcp.tools import (
    is_mcp_tool_name,
    mcp_tool_name,
    mcp_tool_to_agent_tool,
    parse_mcp_tool_name,
)
from axis_coding.paths import AxisPaths
from axis_coding.resources import AxisResourcePaths

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestMcpServerConfig:
    def test_minimal_config(self) -> None:
        cfg = McpServerConfig(command="npx")
        assert cfg.command == "npx"
        assert cfg.args == []
        assert cfg.env == {}
        assert cfg.enabled is True

    def test_full_config(self) -> None:
        cfg = McpServerConfig(
            command="uvx",
            args=["mcp-server-test", "--port", "8080"],
            env={"API_KEY": "test"},
            enabled=False,
        )
        assert cfg.enabled is False
        assert len(cfg.args) == 3

    def test_rejects_empty_command(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            McpServerConfig(command="")

    def test_forbids_extra_fields(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            McpServerConfig.model_validate({"command": "npx", "unknown_field": True})


class TestMcpConfig:
    def test_empty_config(self) -> None:
        cfg = McpConfig()
        assert cfg.servers == {}

    def test_parses_json(self) -> None:
        raw = json.dumps(
            {
                "servers": {
                    "filesystem": {
                        "command": "npx",
                        "args": ["-y", "@anthropic-ai/mcp-server-filesystem", "/tmp"],
                    }
                }
            }
        )
        cfg = McpConfig.model_validate_json(raw)
        assert "filesystem" in cfg.servers
        assert cfg.servers["filesystem"].command == "npx"

    def test_rejects_invalid_top_level_keys(self) -> None:
        from pydantic import ValidationError

        raw = json.dumps({"servers": {}, "unknown": 1})
        with pytest.raises(ValidationError):
            McpConfig.model_validate_json(raw)


class TestEnvVarExpansion:
    def test_expands_dollar_brace_syntax(self) -> None:
        os.environ["MCP_TEST_VAR"] = "expanded_value"
        try:
            cfg = McpServerConfig(
                command="npx",
                env={"TOKEN": "${MCP_TEST_VAR}", "PLAIN": "keep"},
            )
            result = expand_env_vars(cfg)
            assert result.env["TOKEN"] == "expanded_value"
            assert result.env["PLAIN"] == "keep"
        finally:
            del os.environ["MCP_TEST_VAR"]

    def test_keeps_unmatched_unchanged(self) -> None:
        cfg = McpServerConfig(command="npx", env={"X": "${NONEXISTENT_VAR_12345}"})
        result = expand_env_vars(cfg)
        assert result.env["X"] == "${NONEXISTENT_VAR_12345}"


class TestLoadMcpConfig:
    def test_no_config_files_returns_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = AxisResourcePaths(cwd=Path(tmp))
            config, diags = load_mcp_config(paths, cwd=Path(tmp))
            assert config.servers == {}
            assert diags == ()

    def test_loads_user_config(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir()
            mcp_json = home / MCP_CONFIG_FILENAME
            mcp_json.write_text(
                json.dumps({"servers": {"test": {"command": "echo", "args": ["hello"]}}}),
                encoding="utf-8",
            )
            paths = AxisResourcePaths(paths=AxisPaths(home=home, agents_home=home))
            config, diags = load_mcp_config(paths, cwd=Path(tmp))
            assert "test" in config.servers
            assert diags == ()

    def test_invalid_json_produces_diagnostic(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir()
            mcp_json = home / MCP_CONFIG_FILENAME
            mcp_json.write_text("not valid json", encoding="utf-8")
            paths = AxisResourcePaths(paths=AxisPaths(home=home, agents_home=home))
            config, diags = load_mcp_config(paths, cwd=Path(tmp))
            assert config.servers == {}
            assert len(diags) == 1

    def test_rejects_invalid_server_name(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir()
            mcp_json = home / MCP_CONFIG_FILENAME
            mcp_json.write_text(
                json.dumps({"servers": {"bad name!": {"command": "echo"}}}),
                encoding="utf-8",
            )
            paths = AxisResourcePaths(paths=AxisPaths(home=home, agents_home=home))
            config, diags = load_mcp_config(paths, cwd=Path(tmp))
            assert config.servers == {}
            assert len(diags) == 1
            assert "bad name" in diags[0].message


# ---------------------------------------------------------------------------
# Tool namespacing tests
# ---------------------------------------------------------------------------


class TestToolNamespacing:
    def test_mcp_tool_name(self) -> None:
        assert mcp_tool_name("filesystem", "read_file") == "mcp:filesystem:read_file"
        assert mcp_tool_name("github", "search_repos") == "mcp:github:search_repos"

    def test_parse_mcp_tool_name(self) -> None:
        assert parse_mcp_tool_name("mcp:fs:read") == ("fs", "read")
        assert parse_mcp_tool_name("mcp:a:b:c") is None
        assert parse_mcp_tool_name("read") is None
        assert parse_mcp_tool_name("not-mcp:tool") is None

    def test_is_mcp_tool_name(self) -> None:
        assert is_mcp_tool_name("mcp:fs:read") is True
        assert is_mcp_tool_name("read") is False
        assert is_mcp_tool_name("write") is False
        assert is_mcp_tool_name("bash") is False


class TestMcpToolConversion:
    def test_converts_basic_tool(self) -> None:
        from axis_coding.mcp.client import McpToolInfo

        info = McpToolInfo(
            name="read_file",
            description="Read a file from the filesystem",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )

        async def dummy_executor(args, signal=None):
            from axis_agent.tools import AgentToolResult

            return AgentToolResult(tool_call_id="", name="test", ok=True, content="")

        tool = mcp_tool_to_agent_tool("filesystem", info, dummy_executor)
        assert isinstance(tool, AgentTool)
        assert tool.name == "mcp:filesystem:read_file"
        assert "filesystem" in tool.prompt_snippet
        assert "read_file" in tool.prompt_snippet
        assert tool.requires_approval is True
        assert "type" in tool.input_schema

    def test_normalizes_minimal_schema(self) -> None:

        from axis_coding.mcp.tools import _normalize_json_schema

        result = _normalize_json_schema({})
        assert result["type"] == "object"
        assert result["properties"] == {}

    def test_preserves_existing_schema(self) -> None:
        from axis_coding.mcp.tools import _normalize_json_schema

        schema = {
            "type": "object",
            "properties": {"x": {"type": "number"}},
            "required": ["x"],
        }
        result = _normalize_json_schema(schema)
        assert result == schema


# ---------------------------------------------------------------------------
# McpManager tests
# ---------------------------------------------------------------------------


class TestMcpManager:
    def test_empty_config_produces_no_servers(self) -> None:
        config = McpConfig()
        manager = McpManager(config)
        assert manager.server_count == 0
        assert manager.tool_count == 0
        assert manager.discovered is False

    def test_connect_all_with_empty_config(self) -> None:
        async def run() -> None:
            config = McpConfig()
            manager = McpManager(config)
            await manager.connect_all()
            assert manager.server_count == 0
            assert manager.diagnostics == ()

        asyncio.run(run())

    def test_connect_all_skips_disabled_servers(self) -> None:
        async def run() -> None:
            config = McpConfig(
                servers={
                    "disabled_one": McpServerConfig(command="echo", args=["test"], enabled=False)
                }
            )
            manager = McpManager(config)
            await manager.connect_all()
            assert manager.server_count == 0
            status = manager.server_statuses[0]
            assert status.connected is False
            assert "disabled" in (status.error or "").lower()

        asyncio.run(run())

    def test_connect_all_handles_invalid_command(self) -> None:
        async def run() -> None:
            config = McpConfig(
                servers={"bad": McpServerConfig(command="/nonexistent/path/to/binary_12345")}
            )
            manager = McpManager(config)
            await manager.connect_all()
            assert manager.server_count == 0
            assert len(manager.diagnostics) >= 1

        asyncio.run(run())

    def test_server_statuses(self) -> None:
        config = McpConfig(
            servers={
                "one": McpServerConfig(command="echo", args=["hello"]),
                "two": McpServerConfig(command="echo", args=["world"], enabled=False),
            }
        )
        manager = McpManager(config)
        statuses = manager.server_statuses
        assert len(statuses) == 2
        names = {s.name for s in statuses}
        assert names == {"one", "two"}

    def test_discover_tools_caches(self) -> None:
        async def run() -> None:
            config = McpConfig()
            manager = McpManager(config)
            tools1 = await manager.discover_tools()
            tools2 = await manager.discover_tools()
            assert tools1 == tools2

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Permissions integration
# ---------------------------------------------------------------------------


class TestMcpApprovalPreview:
    def test_parse_mcp_name(self) -> None:
        from axis_coding.permissions import _parse_mcp_name

        assert _parse_mcp_name("mcp:github:search_repos") == ("github", "search_repos")
        assert _parse_mcp_name("mcp:fs:read") == ("fs", "read")
        assert _parse_mcp_name("read") is None
        assert _parse_mcp_name("mcp:too:many:parts") is None
        assert _parse_mcp_name("mcp:a:") is None
        assert _parse_mcp_name("mcp::b") is None

    def test_build_preview_for_mcp_tool(self) -> None:
        from axis_agent.tools import ToolCall
        from axis_coding.permissions import build_tool_approval_preview

        call = ToolCall(
            id="call_1",
            name="mcp:github:search_repos",
            arguments={"query": "test", "limit": 10},
        )
        preview = build_tool_approval_preview(call, cwd=Path("/tmp"))
        assert "MCP" in preview.title
        assert "github" in preview.summary
        assert "search_repos" in preview.summary
