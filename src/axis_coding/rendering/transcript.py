"""Human-readable live transcript renderer."""

import json
import sys
from typing import TextIO

from axis_agent import (
    AgentEndEvent,
    AgentEvent,
    AssistantMessage,
    ContextCompactionEvent,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    RetryEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)


class TranscriptRenderer:
    """Stream assistant text and render tool/runtime activity to stderr."""

    def __init__(
        self,
        *,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        self._stdout = sys.stdout if stdout is None else stdout
        self._stderr = sys.stderr if stderr is None else stderr
        self._assistant_open = False
        self._assistant_has_output = False
        self._failed = False

    def render(self, event: AgentEvent) -> None:
        """Render one event while deliberately hiding thinking deltas."""
        if isinstance(event, MessageStartEvent):
            if event.message_role == "assistant":
                self._close_assistant()
                self._assistant_open = True
            return

        if isinstance(event, MessageDeltaEvent):
            self._assistant_open = True
            self._assistant_has_output = True
            self._write_stdout(event.delta)
            return

        if isinstance(event, MessageEndEvent):
            if isinstance(event.message, AssistantMessage):
                if not self._assistant_has_output and event.message.content:
                    self._assistant_open = True
                    self._assistant_has_output = True
                    self._write_stdout(event.message.content)
                self._close_assistant()
            return

        if isinstance(event, ToolExecutionStartEvent):
            self._close_assistant()
            arguments = json.dumps(
                event.tool_call.arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            suffix = f" {arguments}" if event.tool_call.arguments else ""
            self._write_stderr(f"→ {event.tool_call.name}{suffix}\n")
            return

        if isinstance(event, ContextCompactionEvent):
            self._close_assistant()
            self._write_stderr(
                "… Auto-compacted context "
                f"({event.before_tokens} → {event.after_tokens} tokens; "
                f"kept {event.retained_entries} entries).\n"
            )
            return

        if isinstance(event, ToolExecutionUpdateEvent):
            self._close_assistant()
            self._write_stderr(f"… {event.message}\n")
            return

        if isinstance(event, RetryEvent):
            self._close_assistant()
            self._write_stderr(f"… {event.message}\n")
            return

        if isinstance(event, ToolExecutionEndEvent):
            self._close_assistant()
            marker = "✓" if event.result.ok else "✗"
            self._write_stderr(f"{marker} {event.result.name}\n")
            for line in event.result.content.splitlines():
                self._write_stderr(f"  {line}\n")
            return

        if isinstance(event, ErrorEvent):
            if not event.recoverable or bool(
                event.data and event.data.get("request_aborted") is True
            ):
                self._failed = True
            self._close_assistant()
            self._write_stderr(f"Error: {event.message}\n")
            return

        if isinstance(event, AgentEndEvent):
            self._close_assistant()

    def finish(self) -> bool:
        """Close partial text and report whether the run succeeded."""
        self._close_assistant()
        return not self._failed

    def _close_assistant(self) -> None:
        if self._assistant_open and self._assistant_has_output:
            self._write_stdout("\n")
        self._assistant_open = False
        self._assistant_has_output = False

    def _write_stdout(self, value: str) -> None:
        self._stdout.write(value)
        self._stdout.flush()

    def _write_stderr(self, value: str) -> None:
        self._stderr.write(value)
        self._stderr.flush()
