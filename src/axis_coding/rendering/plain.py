"""Final-text renderer for non-interactive Axis runs."""

import sys
from typing import TextIO

from axis_agent import AgentEvent, AssistantMessage, ErrorEvent, MessageEndEvent


class FinalTextRenderer:
    """Print only the final complete assistant text after an agent run."""

    def __init__(
        self,
        *,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        self._stdout = sys.stdout if stdout is None else stdout
        self._stderr = sys.stderr if stderr is None else stderr
        self._last_assistant_text = ""
        self._failed = False
        self._error_messages: list[str] = []

    def render(self, event: AgentEvent) -> None:
        """Record only events needed to produce final text or an error."""
        if isinstance(event, MessageEndEvent) and isinstance(event.message, AssistantMessage):
            self._last_assistant_text = event.message.content
        elif isinstance(event, ErrorEvent):
            if not event.recoverable or bool(
                event.data and event.data.get("request_aborted") is True
            ):
                self._failed = True
            self._error_messages.append(event.message)

    def finish(self) -> bool:
        """Write buffered output and return whether the run succeeded."""
        if self._failed:
            for message in self._error_messages:
                self._stderr.write(f"Error: {message}\n")
            return False
        if self._last_assistant_text:
            self._stdout.write(f"{self._last_assistant_text}\n")
        return True
