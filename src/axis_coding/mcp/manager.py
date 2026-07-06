"""Orchestrate MCP server connections, tool discovery, and tool execution."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

from axis_agent.tools import AgentTool, AgentToolResult, ToolCancellationToken
from axis_agent.types import JSONValue
from axis_coding.mcp.client import McpClientSession, McpConnectionError, McpToolInfo
from axis_coding.mcp.config import McpConfig, McpServerConfig
from axis_coding.mcp.tools import (
    create_mcp_tool_executor,
    mcp_tool_to_agent_tool,
)
from axis_coding.resources import AxisResourcePaths, ResourceDiagnostic


@dataclass(frozen=True, slots=True)
class McpServerStatus:
    """Observable status of one MCP server."""

    name: str
    connected: bool
    tool_count: int
    tool_names: tuple[str, ...]
    error: str | None = None


class McpManager:
    """Connect to configured MCP servers, discover tools, and route executions."""

    def __init__(
        self,
        config: McpConfig,
        resource_paths: AxisResourcePaths | None = None,
    ) -> None:
        self._config = config
        self._resource_paths = resource_paths
        self._clients: dict[str, McpClientSession] = {}
        self._tools: dict[str, AgentTool] = {}
        self._server_tools: dict[str, tuple[str, ...]] = {}
        self._diagnostics: list[ResourceDiagnostic] = []
        self._discovered = False

    @property
    def diagnostics(self) -> tuple[ResourceDiagnostic, ...]:
        return tuple(self._diagnostics)

    @property
    def discovered(self) -> bool:
        return self._discovered

    @property
    def server_count(self) -> int:
        return len(self._clients)

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def server_statuses(self) -> tuple[McpServerStatus, ...]:
        """Return live status for all configured servers."""
        result: list[McpServerStatus] = []
        for name, server_config in self._config.servers.items():
            client = self._clients.get(name)
            tool_names = self._server_tools.get(name, ())
            if not server_config.enabled:
                result.append(
                    McpServerStatus(
                        name=name,
                        connected=False,
                        tool_count=0,
                        tool_names=(),
                        error="Server is disabled",
                    )
                )
            elif client is not None and client.connected:
                result.append(
                    McpServerStatus(
                        name=name,
                        connected=True,
                        tool_count=len(tool_names),
                        tool_names=tool_names,
                    )
                )
            else:
                error: str | None = None
                for diag in self._diagnostics:
                    if diag.name == name:
                        error = diag.message
                        break
                result.append(
                    McpServerStatus(
                        name=name,
                        connected=False,
                        tool_count=0,
                        tool_names=(),
                        error=error or "Not connected",
                    )
                )
        return tuple(result)

    async def connect_all(self) -> None:
        """Concurrently connect to all enabled MCP servers."""
        enabled = {
            name: server_config
            for name, server_config in self._config.servers.items()
            if server_config.enabled
        }
        if not enabled:
            return

        async def connect_one(name: str, server_config: McpServerConfig) -> None:
            try:
                client = await McpClientSession.connect(name, server_config)
                self._clients[name] = client
            except McpConnectionError as exc:
                self._diagnostics.append(
                    ResourceDiagnostic(
                        kind="mcp",
                        name=name,
                        message=str(exc),
                        severity="warning",
                    )
                )

        tasks = [asyncio.create_task(connect_one(name, cfg)) for name, cfg in enabled.items()]
        if tasks:
            await asyncio.gather(*tasks)

    async def discover_tools(self) -> tuple[AgentTool, ...]:
        """Discover tools from all connected servers and cache the results."""
        if self._discovered:
            return tuple(self._tools.values())

        async def discover_one(name: str, client: McpClientSession) -> list[McpToolInfo]:
            try:
                return await client.list_tools()
            except McpConnectionError as exc:
                self._diagnostics.append(
                    ResourceDiagnostic(
                        kind="mcp",
                        name=name,
                        message=f"Tool discovery failed: {exc}",
                        severity="warning",
                    )
                )
                return []

        tasks = [
            asyncio.create_task(discover_one(name, client))
            for name, client in self._clients.items()
        ]
        if not tasks:
            self._discovered = True
            return ()

        results = await asyncio.gather(*tasks)
        for (name, _client), tools in zip(self._clients.items(), results, strict=True):
            if not tools:
                continue
            agent_tools: list[AgentTool] = []
            tool_names: list[str] = []
            for tool_info in tools:
                executor = create_mcp_tool_executor(
                    self._clients[name],
                    tool_info.name,
                )
                agent_tool = mcp_tool_to_agent_tool(name, tool_info, executor)
                agent_tools.append(agent_tool)
                tool_names.append(agent_tool.name)
            self._server_tools[name] = tuple(tool_names)
            for at in agent_tools:
                self._tools[at.name] = at

        self._discovered = True
        return tuple(self._tools.values())

    async def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        """Execute a namespaced MCP tool, returning its result."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return AgentToolResult(
                tool_call_id="",
                name=tool_name,
                ok=False,
                content="",
                error=f"Unknown MCP tool: {tool_name}",
            )
        return await tool.execute(arguments, signal=signal)

    async def disconnect_all(self) -> None:
        """Disconnect all servers concurrently."""
        tasks = [asyncio.create_task(client.disconnect()) for client in self._clients.values()]
        if tasks:
            await asyncio.gather(*tasks)
        self._clients.clear()
        self._tools.clear()
        self._server_tools.clear()
        self._discovered = False
