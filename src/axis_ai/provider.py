"""Provider protocol for Axis model adapters."""

from collections.abc import AsyncIterator
from typing import Protocol

from axis_agent.messages import AgentMessage
from axis_agent.tools import AgentTool
from axis_ai.events import ProviderEvent


class CancellationToken(Protocol):
    """Minimal cancellation interface accepted by providers."""

    def is_cancelled(self) -> bool:
        """Return whether the current stream should stop."""
        ...


class ModelProvider(Protocol):
    """Provider-neutral interface for streaming model responses."""

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        """Stream one model response as provider-neutral events."""
        ...
