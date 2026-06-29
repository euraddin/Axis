"""Shared contracts for Axis print-mode event renderers."""

from enum import StrEnum
from typing import Protocol

from axis_agent import AgentEvent


class PrintOutputMode(StrEnum):
    """Supported non-interactive output formats."""

    TEXT = "text"
    JSON = "json"
    TRANSCRIPT = "transcript"


class EventRenderer(Protocol):
    """Consume AgentEvents and produce one frontend representation."""

    def render(self, event: AgentEvent) -> None:
        """Consume one event."""
        ...

    def finish(self) -> bool:
        """Finish output and report whether the run succeeded."""
        ...
