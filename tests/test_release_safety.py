"""Release guards for credentials and local-only files."""

import re
from pathlib import Path


def test_environment_example_contains_only_an_obvious_placeholder() -> None:
    repository = Path(__file__).parents[1]
    example = (repository / ".env.example").read_text(encoding="utf-8")

    assert "DEEPSEEK_API_KEY=replace-with-your-key" in example
    assert re.search(r"\bsk-[A-Za-z0-9]{20,}\b", example) is None


def test_local_credentials_and_claude_settings_are_ignored() -> None:
    repository = Path(__file__).parents[1]
    ignore = (repository / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in ignore
    assert ".claude/settings.local.json" in ignore
