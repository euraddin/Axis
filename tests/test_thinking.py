"""Tests for Axis thinking-mode normalization and provider mapping."""

import pytest

from axis_coding import (
    next_thinking_level,
    normalize_thinking_level,
    normalize_thinking_levels,
    reasoning_effort_for_level,
)


def test_thinking_levels_normalize_cycle_and_map_xhigh_to_deepseek_max() -> None:
    assert normalize_thinking_level(" XHIGH ") == "xhigh"
    assert normalize_thinking_levels(["high", "xhigh"]) == ("high", "xhigh")
    assert next_thinking_level("high", available=("high", "xhigh")) == "xhigh"
    assert next_thinking_level("xhigh", available=("high", "xhigh")) == "high"
    assert reasoning_effort_for_level("off") is None
    assert reasoning_effort_for_level("xhigh") == "max"


def test_thinking_levels_reject_unknown_empty_and_duplicates() -> None:
    with pytest.raises(ValueError, match="Unknown thinking mode"):
        normalize_thinking_level("ultra")
    with pytest.raises(ValueError, match="non-empty"):
        normalize_thinking_levels([])
    with pytest.raises(ValueError, match="unique"):
        normalize_thinking_levels(["high", "high"])
