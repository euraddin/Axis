"""Executable dependency rules for Axis's portable agent layer."""

import ast
from pathlib import Path

FORBIDDEN_IMPORT_PREFIXES = ("axis_coding", "rich", "textual", "typer")


def test_axis_agent_does_not_import_application_or_ui_packages() -> None:
    package_root = Path(__file__).parents[1] / "src" / "axis_agent"
    violations: list[str] = []

    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules.append(node.module)

            for module in modules:
                if any(
                    module == prefix or module.startswith(f"{prefix}.")
                    for prefix in FORBIDDEN_IMPORT_PREFIXES
                ):
                    violations.append(f"{path.relative_to(package_root)} imports {module}")

    assert violations == []
