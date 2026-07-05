"""MCP (Model Context Protocol) integration for Axis coding sessions."""

from axis_coding.mcp.client import McpClientSession, McpConnectionError, McpToolInfo, McpToolResult
from axis_coding.mcp.config import (
    MCP_CONFIG_FILENAME,
    McpConfig,
    McpServerConfig,
    expand_env_vars,
    load_mcp_config,
)
from axis_coding.mcp.manager import McpManager, McpServerStatus
from axis_coding.mcp.tools import (
    MCP_TOOL_PREFIX,
    is_mcp_tool_name,
    mcp_tool_name,
    mcp_tool_to_agent_tool,
    parse_mcp_tool_name,
)

__all__ = [
    "MCP_CONFIG_FILENAME",
    "MCP_TOOL_PREFIX",
    "McpClientSession",
    "McpConfig",
    "McpConnectionError",
    "McpManager",
    "McpServerConfig",
    "McpServerStatus",
    "McpToolInfo",
    "McpToolResult",
    "expand_env_vars",
    "is_mcp_tool_name",
    "load_mcp_config",
    "mcp_tool_name",
    "mcp_tool_to_agent_tool",
    "parse_mcp_tool_name",
]
