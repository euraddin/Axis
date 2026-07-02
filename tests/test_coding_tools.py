"""Tests for Axis's local coding tools."""

import asyncio
import base64
import shlex
import sys
from pathlib import Path
from time import monotonic

import pytest

from axis_coding import (
    ToolInputError,
    create_bash_tool,
    create_bash_tool_definition,
    create_coding_tools,
    create_edit_tool,
    create_edit_tool_definition,
    create_read_tool,
    create_read_tool_definition,
    create_write_tool,
    create_write_tool_definition,
    truncate_head,
    truncate_tail,
)
from axis_coding.tools import DEFAULT_MAX_OUTPUT_BYTES


class FakeCancellationToken:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def is_cancelled(self) -> bool:
        return self.cancelled


def test_read_definition_preserves_schema_and_prompt_metadata(tmp_path: Path) -> None:
    definition = create_read_tool_definition(cwd=tmp_path)
    tool = definition.to_agent_tool()
    properties = definition.input_schema["properties"]

    assert isinstance(properties, dict)
    assert properties["offset"]["type"] == "integer"
    assert properties["limit"]["type"] == "integer"
    assert tool.name == "read"
    assert tool.prompt_snippet == "Read file contents"
    assert tool.prompt_guidelines == definition.prompt_guidelines


def test_read_tool_resolves_relative_path_with_offset_and_limit(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    tool = create_read_tool(cwd=tmp_path)

    result = asyncio.run(tool.execute({"path": "notes.txt", "offset": 2, "limit": 1}))

    assert result.ok is True
    assert result.name == "read"
    assert result.content == "two\n\n[2 more lines in file. Use offset=3 to continue.]"
    assert result.data is not None
    assert result.data["path"] == str(path)
    assert isinstance(result.data["truncation"], dict)


def test_read_tool_treats_zero_offset_as_start_of_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    result = asyncio.run(
        create_read_tool(cwd=tmp_path).execute({"path": "notes.txt", "offset": 0, "limit": 1})
    )

    assert result.content == "one\n\n[3 more lines in file. Use offset=2 to continue.]"


def test_read_tool_allows_absolute_paths_under_local_trust_model(tmp_path: Path) -> None:
    path = tmp_path / "absolute.txt"
    path.write_text("absolute", encoding="utf-8")

    result = asyncio.run(create_read_tool(cwd=tmp_path / "other").execute({"path": str(path)}))

    assert result.content == "absolute"
    assert result.data is not None
    assert result.data["path"] == str(path)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "path must be a string"),
        ({"path": "missing.txt"}, "File not found"),
        ({"path": "notes.txt", "offset": -1}, "offset must be at least 0"),
        ({"path": "notes.txt", "limit": 0}, "limit must be at least 1"),
        ({"path": "notes.txt", "limit": True}, "limit must be an integer"),
        ({"path": "notes.txt", "offset": 99}, "beyond end of file"),
    ],
)
def test_read_tool_rejects_invalid_inputs(
    tmp_path: Path,
    arguments: dict[str, object],
    message: str,
) -> None:
    (tmp_path / "notes.txt").write_text("one\ntwo", encoding="utf-8")
    tool = create_read_tool(cwd=tmp_path)

    with pytest.raises(ToolInputError, match=message):
        asyncio.run(tool.execute(arguments))  # type: ignore[arg-type]


def test_read_tool_rejects_directories(tmp_path: Path) -> None:
    with pytest.raises(ToolInputError, match="Path is a directory"):
        asyncio.run(create_read_tool(cwd=tmp_path).execute({"path": "."}))


def test_truncate_head_respects_line_and_utf8_byte_limits() -> None:
    by_lines = truncate_head("one\ntwo\nthree", max_lines=2, max_bytes=100)
    by_bytes = truncate_head("éé\nnext", max_lines=10, max_bytes=5)

    assert by_lines.content == "one\ntwo"
    assert by_lines.truncated_by == "lines"
    assert by_bytes.content == "éé"
    assert by_bytes.output_bytes == 4
    assert by_bytes.truncated_by == "bytes"


def test_read_tool_truncates_large_files_with_continuation_hint(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("\n".join(str(index) for index in range(2_001)), encoding="utf-8")

    result = asyncio.run(create_read_tool(cwd=tmp_path).execute({"path": "large.txt"}))

    assert "[Showing lines 1-2000 of 2001. Use offset=2001 to continue.]" in result.content
    assert result.data is not None
    truncation = result.data["truncation"]
    assert isinstance(truncation, dict)
    assert truncation["truncated_by"] == "lines"


def test_read_tool_replaces_oversized_first_line_with_safe_hint(tmp_path: Path) -> None:
    path = tmp_path / "one-huge-line.txt"
    path.write_text("x" * (DEFAULT_MAX_OUTPUT_BYTES + 1), encoding="utf-8")

    result = asyncio.run(create_read_tool(cwd=tmp_path).execute({"path": path.name}))

    assert "Line 1" in result.content
    assert "exceeds 50KB limit" in result.content
    assert f"head -c {DEFAULT_MAX_OUTPUT_BYTES}" in result.content
    assert result.data is not None
    truncation = result.data["truncation"]
    assert isinstance(truncation, dict)
    assert truncation["first_line_exceeds_limit"] is True


def test_read_tool_returns_supported_images_as_base64_metadata(tmp_path: Path) -> None:
    image_bytes = b"\x89PNG\r\n\x1a\naxis"
    path = tmp_path / "sample.png"
    path.write_bytes(image_bytes)

    result = asyncio.run(create_read_tool(cwd=tmp_path).execute({"path": "sample.png"}))

    assert result.content == "Read image file [image/png]"
    assert result.data is not None
    assert result.data["mime_type"] == "image/png"
    assert result.data["bytes"] == len(image_bytes)
    assert result.data["image_base64"] == base64.b64encode(image_bytes).decode("ascii")


def test_write_definition_preserves_schema_and_prompt_metadata(tmp_path: Path) -> None:
    definition = create_write_tool_definition(cwd=tmp_path)
    tool = definition.to_agent_tool()

    assert definition.input_schema["required"] == ["path", "content"]
    assert tool.name == "write"
    assert tool.prompt_snippet == "Create or completely rewrite files"
    assert "complete rewrites" in tool.prompt_guidelines[0]


def test_write_tool_creates_parent_directories_and_returns_metadata(tmp_path: Path) -> None:
    tool = create_write_tool(cwd=tmp_path)

    result = asyncio.run(tool.execute({"path": "nested/file.txt", "content": "你好 Axis"}))

    path = tmp_path / "nested" / "file.txt"
    assert path.read_text(encoding="utf-8") == "你好 Axis"
    assert result.ok is True
    assert result.content == f"Successfully wrote to {path}."
    assert result.data == {"path": str(path), "characters": 7}


def test_write_tool_completely_overwrites_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "existing.txt"
    path.write_text("old content that is longer", encoding="utf-8")

    asyncio.run(create_write_tool(cwd=tmp_path).execute({"path": path.name, "content": "new"}))

    assert path.read_text(encoding="utf-8") == "new"


def test_write_tool_allows_absolute_paths_under_local_trust_model(tmp_path: Path) -> None:
    path = tmp_path / "absolute.txt"

    result = asyncio.run(
        create_write_tool(cwd=tmp_path / "other").execute(
            {"path": str(path), "content": "absolute"}
        )
    )

    assert path.read_text(encoding="utf-8") == "absolute"
    assert result.data == {"path": str(path), "characters": 8}


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"content": "hello"}, "path must be a string"),
        ({"path": "file.txt"}, "content must be a string"),
        ({"path": 42, "content": "hello"}, "path must be a string"),
        ({"path": "file.txt", "content": False}, "content must be a string"),
    ],
)
def test_write_tool_rejects_invalid_inputs(
    tmp_path: Path,
    arguments: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ToolInputError, match=message):
        asyncio.run(create_write_tool(cwd=tmp_path).execute(arguments))  # type: ignore[arg-type]


def test_write_and_read_tools_share_the_same_cwd_contract(tmp_path: Path) -> None:
    asyncio.run(
        create_write_tool(cwd=tmp_path).execute(
            {"path": "shared.txt", "content": "written then read"}
        )
    )

    result = asyncio.run(create_read_tool(cwd=tmp_path).execute({"path": "shared.txt"}))

    assert result.content == "written then read"


def test_edit_definition_exposes_exact_replacement_contract(tmp_path: Path) -> None:
    definition = create_edit_tool_definition(cwd=tmp_path)
    tool = definition.to_agent_tool()
    properties = definition.input_schema["properties"]

    assert isinstance(properties, dict)
    assert properties["edits"]["type"] == "array"
    assert tool.name == "edit"
    assert len(tool.prompt_guidelines) == 4


def test_edit_tool_applies_multiple_replacements_and_returns_diff_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "file.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    tool = create_edit_tool(cwd=tmp_path)

    result = asyncio.run(
        tool.execute(
            {
                "path": path.name,
                "edits": [
                    {"oldText": "alpha", "newText": "one"},
                    {"oldText": "gamma", "newText": "three"},
                ],
            }
        )
    )

    assert path.read_text(encoding="utf-8") == "one\nbeta\nthree\n"
    assert result.content == f"Successfully replaced 2 block(s) in {path}."
    assert result.data is not None
    assert result.data["edits"] == 2
    assert result.data["first_changed_line"] == 1
    assert "- alpha" in str(result.data["diff"])
    assert "+one" in str(result.data["patch"])


def test_edit_tool_rolls_back_when_any_replacement_fails(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    original = "alpha\nbeta\ngamma\n"
    path.write_text(original, encoding="utf-8")
    tool = create_edit_tool(cwd=tmp_path)

    with pytest.raises(ToolInputError, match=r"Could not find edits\[1\]"):
        asyncio.run(
            tool.execute(
                {
                    "path": path.name,
                    "edits": [
                        {"oldText": "alpha", "newText": "one"},
                        {"oldText": "missing", "newText": "nope"},
                    ],
                }
            )
        )

    assert path.read_text(encoding="utf-8") == original


def test_edit_tool_requires_each_old_text_to_be_unique(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("repeat\nrepeat\n", encoding="utf-8")

    with pytest.raises(ToolInputError, match="Found 2 occurrences"):
        asyncio.run(
            create_edit_tool(cwd=tmp_path).execute(
                {
                    "path": path.name,
                    "edits": [{"oldText": "repeat", "newText": "once"}],
                }
            )
        )


def test_edit_tool_rejects_overlapping_replacements_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    original = "abcdef"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ToolInputError, match="must not overlap"):
        asyncio.run(
            create_edit_tool(cwd=tmp_path).execute(
                {
                    "path": path.name,
                    "edits": [
                        {"oldText": "abc", "newText": "one"},
                        {"oldText": "bc", "newText": "two"},
                    ],
                }
            )
        )

    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("edits", "message"),
    [
        ([], "at least one replacement"),
        (["not an object"], r"edits\[0\] must be an object"),
        ([{"oldText": "alpha"}], r"edits\[0\].oldText and edits\[0\].newText"),
        ([{"oldText": "", "newText": "value"}], "oldText must not be empty"),
        ([{"oldText": "alpha", "newText": "alpha"}], "No changes made"),
    ],
)
def test_edit_tool_rejects_invalid_or_noop_replacements(
    tmp_path: Path,
    edits: list[object],
    message: str,
) -> None:
    path = tmp_path / "file.txt"
    path.write_text("alpha", encoding="utf-8")

    with pytest.raises(ToolInputError, match=message):
        asyncio.run(
            create_edit_tool(cwd=tmp_path).execute(
                {"path": path.name, "edits": edits}  # type: ignore[dict-item]
            )
        )

    assert path.read_text(encoding="utf-8") == "alpha"


def test_edit_tool_preserves_utf8_bom_and_crlf_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "windows.txt"
    path.write_bytes(b"\xef\xbb\xbfalpha\r\nbeta\r\n")

    asyncio.run(
        create_edit_tool(cwd=tmp_path).execute(
            {
                "path": path.name,
                "edits": [{"oldText": "alpha\nbeta", "newText": "one\ntwo"}],
            }
        )
    )

    assert path.read_bytes() == b"\xef\xbb\xbfone\r\ntwo\r\n"


def test_edit_tool_accepts_json_string_and_legacy_replacement_arguments(tmp_path: Path) -> None:
    json_path = tmp_path / "json.txt"
    legacy_path = tmp_path / "legacy.txt"
    json_path.write_text("alpha", encoding="utf-8")
    legacy_path.write_text("alpha", encoding="utf-8")
    tool = create_edit_tool(cwd=tmp_path)

    asyncio.run(
        tool.execute(
            {
                "path": json_path.name,
                "edits": '[{"oldText": "alpha", "newText": "json"}]',
            }
        )
    )
    asyncio.run(
        tool.execute(
            {
                "path": legacy_path.name,
                "oldText": "alpha",
                "newText": "legacy",
            }
        )
    )

    assert json_path.read_text(encoding="utf-8") == "json"
    assert legacy_path.read_text(encoding="utf-8") == "legacy"


@pytest.mark.parametrize(
    ("path_value", "message"),
    [
        ("missing.txt", "File not found"),
        (".", "Path is a directory"),
    ],
)
def test_edit_tool_rejects_missing_files_and_directories(
    tmp_path: Path,
    path_value: str,
    message: str,
) -> None:
    with pytest.raises(ToolInputError, match=message):
        asyncio.run(
            create_edit_tool(cwd=tmp_path).execute(
                {
                    "path": path_value,
                    "edits": [{"oldText": "alpha", "newText": "one"}],
                }
            )
        )


def test_create_coding_tools_returns_stable_default_order(tmp_path: Path) -> None:
    tools = create_coding_tools(cwd=tmp_path)

    assert [tool.name for tool in tools] == ["read", "write", "edit", "bash"]
    assert all(tool.requires_approval for tool in tools)


def test_bash_definition_exposes_optional_numeric_timeout(tmp_path: Path) -> None:
    definition = create_bash_tool_definition(cwd=tmp_path)
    properties = definition.input_schema["properties"]

    assert isinstance(properties, dict)
    assert properties["timeout"]["type"] == "number"
    assert definition.to_agent_tool().prompt_snippet == "Execute shell commands"


def test_bash_tool_runs_in_cwd_and_preserves_merged_output_order(tmp_path: Path) -> None:
    command = "pwd; printf out; printf err >&2; printf tail"

    result = asyncio.run(create_bash_tool(cwd=tmp_path).execute({"command": command}))

    output_lines = result.content.splitlines()
    assert Path(output_lines[0]).resolve() == tmp_path.resolve()
    assert output_lines[1] == "outerrtail"
    assert result.ok is True
    assert result.data is not None
    assert result.data["exit_code"] == 0
    assert result.data["timed_out"] is False
    assert result.data["cancelled"] is False


def test_bash_tool_reports_nonzero_exit_and_empty_success(tmp_path: Path) -> None:
    failed = asyncio.run(create_bash_tool(cwd=tmp_path).execute({"command": "printf bad; exit 7"}))
    empty = asyncio.run(create_bash_tool(cwd=tmp_path).execute({"command": ":"}))

    assert failed.ok is False
    assert failed.error == "Command exited with code 7"
    assert failed.content == "bad\n\nCommand exited with code 7"
    assert empty.ok is True
    assert empty.content == "(no output)"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "command must be a string"),
        ({"command": 42}, "command must be a string"),
        ({"command": ":", "timeout": 0}, "timeout must be greater than 0"),
        ({"command": ":", "timeout": True}, "timeout must be a number"),
        ({"command": ":", "timeout": "soon"}, "timeout must be a number"),
    ],
)
def test_bash_tool_rejects_invalid_arguments(
    tmp_path: Path,
    arguments: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ToolInputError, match=message):
        asyncio.run(create_bash_tool(cwd=tmp_path).execute(arguments))  # type: ignore[arg-type]


def test_bash_tool_rejects_already_cancelled_signal(tmp_path: Path) -> None:
    token = FakeCancellationToken()
    token.cancel()

    with pytest.raises(ToolInputError, match="Command cancelled"):
        asyncio.run(create_bash_tool(cwd=tmp_path).execute({"command": ":"}, signal=token))


def test_bash_tool_reports_timeout(tmp_path: Path) -> None:
    start = monotonic()
    result = asyncio.run(
        create_bash_tool(cwd=tmp_path).execute({"command": "sleep 1", "timeout": 0.01})
    )

    assert result.ok is False
    assert result.error == "Command timed out after 0.01 seconds"
    assert result.data is not None
    assert result.data["timed_out"] is True
    assert monotonic() - start < 0.5


def test_bash_timeout_kills_background_children(tmp_path: Path) -> None:
    marker = tmp_path / "marker"

    result = asyncio.run(
        create_bash_tool(cwd=tmp_path).execute(
            {"command": "(sleep 0.25; touch marker) & wait", "timeout": 0.01}
        )
    )
    asyncio.run(asyncio.sleep(0.35))

    assert result.ok is False
    assert result.data is not None
    assert result.data["timed_out"] is True
    assert not marker.exists()


def test_bash_cancellation_kills_process_group(tmp_path: Path) -> None:
    token = FakeCancellationToken()

    async def scenario() -> tuple[object, float]:
        task = asyncio.create_task(
            create_bash_tool(cwd=tmp_path).execute({"command": "sleep 1 & wait"}, signal=token)
        )
        await asyncio.sleep(0.05)
        token.cancel()
        start = monotonic()
        return await task, monotonic() - start

    result, cancellation_duration = asyncio.run(scenario())

    assert result.ok is False  # type: ignore[union-attr]
    assert result.error == "Command cancelled"  # type: ignore[union-attr]
    assert result.data["cancelled"] is True  # type: ignore[index,union-attr]
    assert cancellation_duration < 0.5


def test_truncate_tail_preserves_latest_lines_and_partial_oversized_line() -> None:
    by_lines = truncate_tail("one\ntwo\nthree", max_lines=2, max_bytes=100)
    partial_line = truncate_tail("prefix\nabcdefgh", max_lines=10, max_bytes=5)

    assert by_lines.content == "two\nthree"
    assert by_lines.truncated_by == "lines"
    assert partial_line.content == "defgh"
    assert partial_line.truncated_by == "bytes"
    assert partial_line.last_line_partial is True


def test_bash_truncates_tail_and_saves_complete_output(tmp_path: Path) -> None:
    script = 'print("\\n".join(str(index) for index in range(2001)))'
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    result = asyncio.run(create_bash_tool(cwd=tmp_path).execute({"command": command}))

    assert result.ok is True
    assert result.content.startswith("1\n2\n")
    assert "[Showing lines 2-2001 of 2001." in result.content
    assert result.data is not None
    full_output_path = result.data["full_output_path"]
    assert isinstance(full_output_path, str)
    saved_output = Path(full_output_path)
    try:
        assert saved_output.read_text(encoding="utf-8").startswith("0\n1\n")
        assert saved_output.read_text(encoding="utf-8").rstrip().endswith("2000")
    finally:
        saved_output.unlink(missing_ok=True)
