"""Thinking-mode primitives for Axis coding sessions."""

from collections.abc import Sequence
from typing import Literal

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]
ThinkingParameter = Literal["reasoning_effort", "reasoning.effort"]

THINKING_LEVELS: tuple[ThinkingLevel, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)
DEFAULT_THINKING_LEVEL: ThinkingLevel = "xhigh"

THINKING_LEVEL_DESCRIPTIONS: dict[ThinkingLevel, str] = {
    "off": "No reasoning",
    "minimal": "Very brief reasoning",
    "low": "Light reasoning",
    "medium": "Moderate reasoning",
    "high": "Deep reasoning",
    "xhigh": "Maximum reasoning",
}


def normalize_thinking_level(value: str | None) -> ThinkingLevel:
    """Return one normalized thinking level or raise a user-facing error."""
    if value is None:
        return DEFAULT_THINKING_LEVEL
    normalized = value.strip().casefold()
    if normalized in THINKING_LEVELS:
        return normalized
    raise ValueError(
        f"Unknown thinking mode: {value}. Available modes: {', '.join(THINKING_LEVELS)}"
    )


def normalize_thinking_levels(values: Sequence[str]) -> tuple[ThinkingLevel, ...]:
    """Validate a non-empty, duplicate-free thinking-level sequence."""
    if isinstance(values, str) or not values:
        raise ValueError(
            "Thinking modes must be a non-empty list. "
            f"Available modes: {', '.join(THINKING_LEVELS)}"
        )
    normalized = tuple(normalize_thinking_level(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Thinking modes must be unique")
    return normalized


def next_thinking_level(
    current: str | None,
    *,
    available: tuple[ThinkingLevel, ...] = THINKING_LEVELS,
) -> ThinkingLevel:
    """Return the next available mode in a stable cycle."""
    if not available:
        return DEFAULT_THINKING_LEVEL
    try:
        index = available.index(normalize_thinking_level(current))
    except ValueError:
        return available[0]
    return available[(index + 1) % len(available)]


def reasoning_effort_for_level(level: str | None) -> str | None:
    """Map Axis UI levels to OpenAI-compatible reasoning effort values."""
    normalized = normalize_thinking_level(level)
    if normalized == "off":
        return None
    if normalized == "xhigh":
        return "max"
    return normalized
