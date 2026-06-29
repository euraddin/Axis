"""Tests for the initial Axis package layout."""

from importlib import import_module

import pytest


@pytest.mark.parametrize("package_name", ["axis_ai", "axis_agent", "axis_coding"])
def test_top_level_package_is_importable(package_name: str) -> None:
    package = import_module(package_name)

    assert package.__name__ == package_name
