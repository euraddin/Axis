"""Event renderers for Axis command-line output modes."""

from typing import TextIO

from axis_coding.rendering.base import EventRenderer, PrintOutputMode
from axis_coding.rendering.json import JsonEventRenderer
from axis_coding.rendering.plain import FinalTextRenderer
from axis_coding.rendering.transcript import TranscriptRenderer


def create_event_renderer(
    mode: PrintOutputMode,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> EventRenderer:
    """Create the renderer selected by one CLI output mode."""
    if mode is PrintOutputMode.TEXT:
        return FinalTextRenderer(stdout=stdout, stderr=stderr)
    if mode is PrintOutputMode.JSON:
        return JsonEventRenderer(stdout=stdout)
    return TranscriptRenderer(stdout=stdout, stderr=stderr)


__all__ = [
    "EventRenderer",
    "FinalTextRenderer",
    "JsonEventRenderer",
    "PrintOutputMode",
    "TranscriptRenderer",
    "create_event_renderer",
]
