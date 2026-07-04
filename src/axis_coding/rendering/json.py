"""JSON Lines renderer for Axis agent events."""

import sys
from typing import TextIO

from axis_agent import AgentEvent, ErrorEvent


class JsonEventRenderer:
    """Write every AgentEvent as one compact JSON object per line."""

    def __init__(self, *, stdout: TextIO | None = None) -> None:
        self._stdout = sys.stdout if stdout is None else stdout
        self._failed = False

    def render(self, event: AgentEvent) -> None:
        """Write and flush one event without adding non-JSON decoration."""
        if isinstance(event, ErrorEvent) and (
            not event.recoverable or bool(event.data and event.data.get("request_aborted") is True)
        ):
            self._failed = True
        self._stdout.write(f"{event.model_dump_json()}\n")
        self._stdout.flush()

    def finish(self) -> bool:
        """Return false when a non-recoverable error was rendered."""
        return not self._failed
