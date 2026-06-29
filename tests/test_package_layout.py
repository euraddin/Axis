"""Tests for Axis package metadata and import layout."""

from importlib import import_module
from importlib.metadata import entry_points, version
from itertools import permutations
from subprocess import run
from sys import executable

import pytest

from axis_coding import __version__


@pytest.mark.parametrize("package_name", ["axis_ai", "axis_agent", "axis_coding"])
def test_top_level_package_is_importable(package_name: str) -> None:
    package = import_module(package_name)

    assert package.__name__ == package_name


def test_runtime_version_matches_distribution_metadata() -> None:
    assert __version__ == version("axis")


def test_axis_console_script_targets_the_typer_app() -> None:
    scripts = {entry.name: entry.value for entry in entry_points(group="console_scripts")}

    assert scripts["axis"] == "axis_coding.cli:app"


@pytest.mark.parametrize(
    "package_order",
    tuple(permutations(("axis_ai", "axis_agent", "axis_coding"))),
)
def test_packages_import_in_any_order_in_a_fresh_interpreter(
    package_order: tuple[str, ...],
) -> None:
    imports = "; ".join(f"import {package_name}" for package_name in package_order)

    completed = run(
        [executable, "-I", "-c", imports],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
