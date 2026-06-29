"""Tests for shared Markdown resource parsing primitives."""

from pathlib import Path

from axis_coding import (
    ResourceDiagnostic,
    derive_markdown_description,
    parse_markdown_resource,
)


def test_parse_markdown_resource_normalizes_simple_frontmatter() -> None:
    metadata, content = parse_markdown_resource(
        "---\r\n"
        'description: "Review: Python"\r\n'
        "# ignored comment\r\n"
        "owner: axis\r\n"
        "---\r\n"
        "# Review\r\n"
        "Body"
    )

    assert metadata == {"description": "Review: Python", "owner": "axis"}
    assert content == "# Review\nBody"


def test_unclosed_frontmatter_is_preserved_as_markdown() -> None:
    text = "---\ndescription: incomplete\n# Body"

    assert parse_markdown_resource(text) == ({}, text)


def test_frontmatter_closing_delimiter_must_occupy_its_own_line() -> None:
    text = "---\ndescription: incomplete\n---oops\n# Body"

    assert parse_markdown_resource(text) == ({}, text)


def test_markdown_description_uses_first_heading_or_text() -> None:
    assert derive_markdown_description("\n## Python Testing\nRun pytest.") == "Python Testing"
    assert derive_markdown_description("\nRun the focused tests.\nMore") == (
        "Run the focused tests."
    )
    assert derive_markdown_description("\n\n") is None


def test_resource_diagnostic_formats_name_and_path() -> None:
    diagnostic = ResourceDiagnostic(
        kind="skill",
        name="review",
        path=Path("review.md"),
        message="overrides another resource",
    )

    assert diagnostic.format() == ("warning skill review: overrides another resource (review.md)")
