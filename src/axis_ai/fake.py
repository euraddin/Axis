"""Deterministic model provider for tests."""

from collections.abc import AsyncIterator, Iterable

from axis_agent.messages import AgentMessage
from axis_agent.tools import AgentTool
from axis_ai.events import ProviderEvent
from axis_ai.provider import CancellationToken


class FakeProvider:
    """Replay one predefined event stream per provider call."""

    def __init__(self, streams: Iterable[Iterable[ProviderEvent]]) -> None:
        self._streams = [list(stream) for stream in streams]
        self.calls: list[tuple[str, str, list[AgentMessage], list[AgentTool]]] = []

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        """Record this call and replay the next scripted stream."""
        self.calls.append((model, system, list(messages), list(tools)))
        stream = self._streams.pop(0) if self._streams else []

        async def iterator() -> AsyncIterator[ProviderEvent]:
            for event in stream:
                if signal is not None and signal.is_cancelled():
                    return
                yield event

        return iterator()
