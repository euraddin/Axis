"""Convert discovered MCP tools into provider-neutral AgentTool definitions."""

from __future__ import annotations

from collections.abc import Mapping

from axis_agent.tools import AgentTool, AgentToolResult, ToolCancellationToken, ToolExecutor
from axis_agent.types import JSONValue
from axis_coding.mcp.client import McpClientSession, McpToolInfo

MCP_TOOL_PREFIX = "mcp"


def mcp_tool_name(server_name: str, tool_name: str) -> str:
    """Return the namespaced Axis tool name for one MCP tool."""
    return f"{MCP_TOOL_PREFIX}:{server_name}:{tool_name}"


def parse_mcp_tool_name(name: str) -> tuple[str, str] | None:
    """Extract (server_name, tool_name) from a namespaced name, or None."""
    if not name.startswith(f"{MCP_TOOL_PREFIX}:"):
        return None
    # Must have exactly two colons: "mcp:server:tool"
    if name.count(":") != 2:
        return None
    parts = name.split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


def is_mcp_tool_name(name: str) -> bool:
    """Return whether a tool name is an MCP-prefixed tool."""
    return parse_mcp_tool_name(name) is not None


def mcp_tool_to_agent_tool(
    server_name: str,
    tool: McpToolInfo,
    executor: ToolExecutor,
) -> AgentTool:
    """Convert one MCP tool into a standard AgentTool."""
    namespaced = mcp_tool_name(server_name, tool.name)
    return AgentTool(
        name=namespaced,
        description=tool.description or f"MCP tool {tool.name} from {server_name}",
        input_schema=_normalize_json_schema(tool.input_schema),
        executor=executor,
        requires_approval=True,
        prompt_snippet=f"MCP {server_name}: {tool.name}",
        prompt_guidelines=(),
    )


def create_mcp_tool_executor(
    client: McpClientSession,
    tool_name: str,
) -> ToolExecutor:
    """Create an async executor that delegates to the MCP client."""

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        del signal  # MCP SDK handles cancellation internally
        result = await client.call_tool(tool_name, dict(arguments))
        return AgentToolResult(
            tool_call_id="",
            name=tool_name,
            ok=result.ok,
            content=result.content,
            data=result.data,
            error=result.error,
        )

    return execute


def _normalize_json_schema(schema: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Ensure the JSON Schema has required top-level fields for provider compatibility."""
    normalized = dict(schema)
    if "type" not in normalized:
        normalized["type"] = "object"
    if "properties" not in normalized:
        normalized["properties"] = {}
    return normalized
