"""Thin lifecycle wrapper around one MCP stdio server connection."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult

from axis_coding.mcp.config import McpServerConfig, expand_env_vars

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SECONDS = 30.0
TOOL_EXECUTION_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class McpToolInfo:
    """Discovered MCP tool metadata before conversion to AgentTool."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class McpToolResult:
    """Result of one MCP tool execution, normalized for Axis."""

    content: str
    ok: bool
    error: str | None = None
    data: dict[str, Any] | None = None


class McpClientSession:
    """One connected MCP stdio server providing tools to Axis."""

    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._server_name = ""
        self._read_stream: Any = None
        self._write_stream: Any = None
        self._ctx: Any = None
        self._session_ctx: Any = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @classmethod
    async def connect(
        cls,
        server_name: str,
        config: McpServerConfig,
    ) -> McpClientSession:
        """Launch the server subprocess and complete the MCP handshake."""
        instance = cls()
        instance._server_name = server_name
        expanded = expand_env_vars(config)
        params = StdioServerParameters(
            command=expanded.command,
            args=list(expanded.args),
            env=dict(expanded.env) if expanded.env else None,
        )
        try:
            # stdio_client is an async context manager that yields (read, write) streams
            ctx = stdio_client(params)
            read_stream, write_stream = await asyncio.wait_for(
                ctx.__aenter__(),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
            instance._read_stream = read_stream
            instance._write_stream = write_stream
            instance._ctx = ctx

            session_ctx = ClientSession(read_stream, write_stream)
            instance._session = await asyncio.wait_for(
                session_ctx.__aenter__(),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
            instance._session_ctx = session_ctx

            await asyncio.wait_for(
                instance._session.initialize(),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
            instance._connected = True
            logger.info("Connected to MCP server %r", server_name)
        except TimeoutError:
            await instance._cleanup_partial()
            raise McpConnectionError(
                f"MCP server {server_name!r} connection timed out "
                f"after {CONNECT_TIMEOUT_SECONDS}s"
            ) from None
        except Exception as exc:
            await instance._cleanup_partial()
            raise McpConnectionError(
                f"Failed to connect to MCP server {server_name!r}: {exc}"
            ) from exc
        return instance

    async def list_tools(self) -> list[McpToolInfo]:
        """Discover and cache tools from the connected server."""
        if not self._connected or self._session is None:
            raise McpConnectionError(f"MCP server {self._server_name!r} is not connected")
        try:
            result = await asyncio.wait_for(
                self._session.list_tools(),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            raise McpConnectionError(
                f"MCP server {self._server_name!r} list_tools timed out"
            ) from None
        except Exception as exc:
            raise McpConnectionError(
                f"Failed to list tools from MCP server {self._server_name!r}: {exc}"
            ) from exc

        tools: list[McpToolInfo] = []
        for tool in result.tools:
            tools.append(
                McpToolInfo(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                )
            )
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        """Execute one tool on the connected server."""
        if not self._connected or self._session is None:
            return McpToolResult(
                content="",
                ok=False,
                error=f"MCP server {self._server_name!r} is not connected",
            )
        try:
            result: CallToolResult = await asyncio.wait_for(
                self._session.call_tool(name, arguments),
                timeout=TOOL_EXECUTION_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            return McpToolResult(
                content="",
                ok=False,
                error=f"MCP tool {name!r} timed out after {TOOL_EXECUTION_TIMEOUT_SECONDS}s",
            )
        except Exception as exc:
            return McpToolResult(
                content="",
                ok=False,
                error=f"MCP tool {name!r} failed: {exc}",
            )

        content = _extract_text_content(result)
        return McpToolResult(
            content=content,
            ok=not result.isError,
            error=content if result.isError else None,
            data={"server": self._server_name, "tool": name},
        )

    async def disconnect(self) -> None:
        """Gracefully close the MCP server connection."""
        self._connected = False
        errors: list[str] = []

        if self._session_ctx is not None:
            try:
                await asyncio.wait_for(
                    self._session_ctx.__aexit__(None, None, None),
                    timeout=5.0,
                )
            except Exception as exc:
                errors.append(f"session close: {exc}")
            self._session_ctx = None
            self._session = None

        if self._ctx is not None:
            try:
                await asyncio.wait_for(
                    self._ctx.__aexit__(None, None, None),
                    timeout=5.0,
                )
            except Exception as exc:
                errors.append(f"stdio close: {exc}")
            self._ctx = None
            self._read_stream = None
            self._write_stream = None

        if errors:
            logger.warning(
                "Errors during MCP %r disconnect: %s",
                self._server_name,
                "; ".join(errors),
            )
        else:
            logger.info("Disconnected from MCP server %r", self._server_name)

    async def _cleanup_partial(self) -> None:
        """Best-effort cleanup of resources allocated before a failed connect."""
        self._connected = False
        if self._session_ctx is not None:
            with contextlib.suppress(Exception):
                await self._session_ctx.__aexit__(None, None, None)
            self._session_ctx = None
            self._session = None
        if self._ctx is not None:
            with contextlib.suppress(Exception):
                await self._ctx.__aexit__(None, None, None)
            self._ctx = None
            self._read_stream = None
            self._write_stream = None


class McpConnectionError(RuntimeError):
    """A reusable MCP server operation failed."""


def _extract_text_content(result: CallToolResult) -> str:
    """Flatten MCP tool result content items into a single string."""
    parts: list[str] = []
    for item in result.content:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
    if result.structuredContent and not parts:
        parts.append(str(result.structuredContent))
    return "\n".join(parts)
