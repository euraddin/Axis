"""Tests for Axis git workflow tools."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from axis_coding.git_tools import (
    GitToolError,
    _parse_porcelain_status,
    _unescape_git_quoted_path,
    create_git_commit_tool,
    create_git_diff_tool,
    create_git_log_tool,
    create_git_status_tool,
    create_git_tools,
)
from axis_coding.tools import create_coding_tools

# ---------------------------------------------------------------------------
# git repo fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with at least one commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git("init", "-b", "main", cwd=repo)
    _run_git("config", "user.email", "test@axis.dev", cwd=repo)
    _run_git("config", "user.name", "Axis Test", cwd=repo)
    (repo / "README.md").write_text("# Test Repo\n")
    _run_git("add", "README.md", cwd=repo)
    _run_git("commit", "-m", "initial commit", cwd=repo)
    return repo


def _run_git(*args: str, cwd: Path) -> str:
    """Run git synchronously (for test fixtures) and return stdout."""
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# porcelain status parsing
# ---------------------------------------------------------------------------


class TestParsePorcelainStatus:
    def test_empty_output(self) -> None:
        parsed = _parse_porcelain_status("")
        assert all(len(v) == 0 for v in parsed.values())

    def test_modified_not_staged(self) -> None:
        parsed = _parse_porcelain_status(" M README.md")
        assert "README.md" in parsed["unstaged"]
        assert "README.md" not in parsed["staged"]

    def test_staged_new_file(self) -> None:
        parsed = _parse_porcelain_status("A  new_file.py")
        assert "new_file.py" in parsed["staged"]

    def test_untracked_file(self) -> None:
        parsed = _parse_porcelain_status("?? untracked.txt")
        assert "untracked.txt" in parsed["untracked"]

    def test_staged_and_modified(self) -> None:
        parsed = _parse_porcelain_status("MM double.py")
        assert "double.py" in parsed["staged"]
        assert "double.py" in parsed["unstaged"]

    def test_ignored_file(self) -> None:
        parsed = _parse_porcelain_status("!! build/")
        assert "build/" in parsed["ignored"]

    def test_deleted_file(self) -> None:
        parsed = _parse_porcelain_status(" D removed.py")
        assert "removed.py" in parsed["unstaged"]

    def test_renamed_file(self) -> None:
        parsed = _parse_porcelain_status("R  old.py -> new.py")
        assert "old.py -> new.py" in parsed["staged"]

    def test_multiple_files(self) -> None:
        output = "\n".join(
            [
                "M  staged_only.py",
                " M modified_only.py",
                "MM staged_and_modified.py",
                "?? new_file.txt",
                " D deleted.py",
            ]
        )
        parsed = _parse_porcelain_status(output)
        # " M" → unstaged only; " D" → unstaged only; "M " → staged only
        # "MM" → both staged and unstaged; "??" → untracked
        assert len(parsed["staged"]) == 2  # M , MM
        assert len(parsed["unstaged"]) == 3  #  M, MM,  D
        assert len(parsed["untracked"]) == 1

    def test_conflict_both_modified(self) -> None:
        parsed = _parse_porcelain_status("UU conflict.py")
        assert "conflict.py" in parsed["conflict"]


class TestUnescapeGitQuotedPath:
    def test_plain_path(self) -> None:
        assert _unescape_git_quoted_path('"README.md"') == "README.md"

    def test_escaped_backslash(self) -> None:
        result = _unescape_git_quoted_path('"a\\\\b"')
        assert result == "a\\b"

    def test_escaped_tab(self) -> None:
        result = _unescape_git_quoted_path('"a\\tb"')
        assert result == "a\tb"

    def test_unicode_characters(self) -> None:
        # Git quotes paths with octal escapes for non-ASCII characters.
        path = '"\\346\\265\\213\\350\\257\\225.py"'
        result = _unescape_git_quoted_path(path)
        assert result == "测试.py"


# ---------------------------------------------------------------------------
# git_status
# ---------------------------------------------------------------------------


class TestGitStatusTool:
    def test_clean_repo(self, git_repo: Path) -> None:
        async def run() -> None:
            tool = create_git_status_tool(cwd=git_repo)
            result = await tool.execute({})

            assert result.ok is True
            assert "main" in result.content
            assert (
                "clean" in result.content.lower()
                or "nothing to commit" in result.content.lower()
            )
            assert result.data is not None
            assert result.data["clean"] is True

        asyncio.run(run())

    def test_modified_file(self, git_repo: Path) -> None:
        (git_repo / "README.md").write_text("# Modified\n")
        _run_git("add", "README.md", cwd=git_repo)

        async def run() -> None:
            tool = create_git_status_tool(cwd=git_repo)
            result = await tool.execute({})

            assert result.ok is True
            assert result.data is not None
            assert "README.md" in str(result.data["staged"])

        asyncio.run(run())

    def test_untracked_file(self, git_repo: Path) -> None:
        (git_repo / "new.py").write_text("print(1)\n")

        async def run() -> None:
            tool = create_git_status_tool(cwd=git_repo)
            result = await tool.execute({})

            assert result.ok is True
            assert result.data is not None
            assert "new.py" in result.data["untracked"]

        asyncio.run(run())

    def test_detached_head(self, git_repo: Path) -> None:
        # Create a second commit, then checkout the first commit.
        (git_repo / "file.txt").write_text("content\n")
        _run_git("add", "file.txt", cwd=git_repo)
        _run_git("commit", "-m", "second commit", cwd=git_repo)
        # Detach HEAD at the first commit.
        first_commit = _run_git("rev-list", "--max-parents=0", "HEAD", cwd=git_repo)
        _run_git("checkout", first_commit, cwd=git_repo)

        async def run() -> None:
            tool = create_git_status_tool(cwd=git_repo)
            result = await tool.execute({})

            assert result.ok is True
            assert "detached" in result.content.lower()

        asyncio.run(run())

    def test_not_a_git_repo(self, tmp_path: Path) -> None:
        async def run() -> None:
            tool = create_git_status_tool(cwd=tmp_path)
            result = await tool.execute({})

            assert result.ok is False
            assert "fatal" in result.content.lower() or "not a git" in result.content.lower()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# git_diff
# ---------------------------------------------------------------------------


class TestGitDiffTool:
    def test_no_changes_in_clean_repo(self, git_repo: Path) -> None:
        async def run() -> None:
            tool = create_git_diff_tool(cwd=git_repo)
            result = await tool.execute({})

            assert result.ok is True
            assert "(no changes)" in result.content

        asyncio.run(run())

    def test_shows_unstaged_changes(self, git_repo: Path) -> None:
        (git_repo / "README.md").write_text("# Changed\n")

        async def run() -> None:
            tool = create_git_diff_tool(cwd=git_repo)
            result = await tool.execute({})

            assert result.ok is True
            assert "Changed" in result.content
            assert "# Changed" in result.content or "+# Changed" in result.content

        asyncio.run(run())

    def test_shows_staged_changes(self, git_repo: Path) -> None:
        (git_repo / "README.md").write_text("# Staged\n")
        _run_git("add", "README.md", cwd=git_repo)

        async def run() -> None:
            tool = create_git_diff_tool(cwd=git_repo)
            result = await tool.execute({"staged": True})

            assert result.ok is True
            assert "Staged" in result.content

        asyncio.run(run())

    def test_filter_by_path(self, git_repo: Path) -> None:
        (git_repo / "a.py").write_text("x = 1\n")
        (git_repo / "b.py").write_text("y = 2\n")
        _run_git("add", "a.py", "b.py", cwd=git_repo)
        # Modify only a.py after staging to make it show in unstaged diff.
        (git_repo / "a.py").write_text("x = 2\n")

        async def run() -> None:
            tool = create_git_diff_tool(cwd=git_repo)
            result = await tool.execute({"path": "a.py"})

            assert result.ok is True
            assert "a.py" in result.content
            assert "b.py" not in result.content

        asyncio.run(run())

    def test_includes_stats(self, git_repo: Path) -> None:
        (git_repo / "README.md").write_text("# Big change\nExtra line\n")

        async def run() -> None:
            tool = create_git_diff_tool(cwd=git_repo)
            result = await tool.execute({})

            assert result.ok is True
            assert "Stats" in result.content

        asyncio.run(run())


# ---------------------------------------------------------------------------
# git_log
# ---------------------------------------------------------------------------


class TestGitLogTool:
    def test_shows_commit(self, git_repo: Path) -> None:
        async def run() -> None:
            tool = create_git_log_tool(cwd=git_repo)
            result = await tool.execute({})

            assert result.ok is True
            assert "initial commit" in result.content
            assert result.data is not None
            assert result.data["shown"] >= 1

        asyncio.run(run())

    def test_respects_max_count(self, git_repo: Path) -> None:
        # Add a few more commits.
        for i in range(5):
            (git_repo / f"f{i}.txt").write_text(f"content {i}\n")
            _run_git("add", f"f{i}.txt", cwd=git_repo)
            _run_git("commit", "-m", f"commit {i}", cwd=git_repo)

        async def run() -> None:
            tool = create_git_log_tool(cwd=git_repo)
            result = await tool.execute({"max_count": 3})

            assert result.ok is True
            assert result.data is not None
            assert result.data["shown"] == 3

        asyncio.run(run())

    def test_filter_by_path(self, git_repo: Path) -> None:
        (git_repo / "tracked.py").write_text("print(1)\n")
        _run_git("add", "tracked.py", cwd=git_repo)
        _run_git("commit", "-m", "add tracked.py", cwd=git_repo)

        async def run() -> None:
            tool = create_git_log_tool(cwd=git_repo)
            result = await tool.execute({"path": "tracked.py"})

            assert result.ok is True
            assert "add tracked.py" in result.content

        asyncio.run(run())


# ---------------------------------------------------------------------------
# git_commit
# ---------------------------------------------------------------------------


class TestGitCommitTool:
    def test_commits_staged_changes(self, git_repo: Path) -> None:
        (git_repo / "feature.py").write_text("def foo(): pass\n")
        _run_git("add", "feature.py", cwd=git_repo)

        async def run() -> None:
            tool = create_git_commit_tool(cwd=git_repo)
            result = await tool.execute({"message": "Add feature module"})

            assert result.ok is True
            assert "Add feature module" in result.content
            assert result.data is not None
            assert result.data["commit"] is not None
            assert len(result.data["commit"]) >= 7  # short hash

        asyncio.run(run())

    def test_no_staged_changes(self, git_repo: Path) -> None:
        async def run() -> None:
            tool = create_git_commit_tool(cwd=git_repo)
            result = await tool.execute({"message": "nothing to commit"})

            assert result.ok is False
            assert "nothing to commit" in result.content.lower()

        asyncio.run(run())

    def test_empty_message_rejected(self, git_repo: Path) -> None:
        (git_repo / "x.py").write_text("x=1\n")
        _run_git("add", "x.py", cwd=git_repo)

        async def run() -> None:
            tool = create_git_commit_tool(cwd=git_repo)
            result = await tool.execute({"message": ""})

            assert result.ok is False
            assert "empty" in result.content.lower()

        asyncio.run(run())

    def test_whitespace_message_rejected(self, git_repo: Path) -> None:
        (git_repo / "y.py").write_text("y=2\n")
        _run_git("add", "y.py", cwd=git_repo)

        async def run() -> None:
            tool = create_git_commit_tool(cwd=git_repo)
            result = await tool.execute({"message": "   "})

            assert result.ok is False
            assert "empty" in result.content.lower()

        asyncio.run(run())

    def test_requires_approval_flag(self) -> None:
        # Use any Path here, the flag is set before cwd binding.
        tool = create_git_commit_tool(cwd=Path("/tmp"))
        assert tool.requires_approval is True


# ---------------------------------------------------------------------------
# factory and integration
# ---------------------------------------------------------------------------


class TestGitToolsFactory:
    def test_creates_four_tools(self, git_repo: Path) -> None:
        tools = create_git_tools(cwd=git_repo)
        names = [t.name for t in tools]
        assert names == ["git_status", "git_diff", "git_log", "git_commit"]

    def test_read_tools_are_auto_approved(self, git_repo: Path) -> None:
        tools = create_git_tools(cwd=git_repo)
        for tool in tools:
            if tool.name == "git_commit":
                assert tool.requires_approval is True, tool.name
            else:
                assert tool.requires_approval is False, tool.name

    def test_all_tools_have_metadata(self, git_repo: Path) -> None:
        tools = create_git_tools(cwd=git_repo)
        for tool in tools:
            assert tool.name
            assert tool.description
            assert tool.prompt_snippet
            assert tool.input_schema
            assert len(tool.prompt_guidelines) >= 1

    def test_all_tools_are_executable(self, git_repo: Path) -> None:
        tools = create_git_tools(cwd=git_repo)
        for tool in tools:
            assert callable(tool.executor)


class TestGitToolsInCodingTools:
    def test_included_by_default(self, tmp_path: Path) -> None:
        tools = create_coding_tools(cwd=tmp_path)
        names = {t.name for t in tools}
        for expected in {"git_status", "git_diff", "git_log", "git_commit"}:
            assert expected in names

    def test_can_exclude(self, tmp_path: Path) -> None:
        tools = create_coding_tools(cwd=tmp_path, include_git_tools=False)
        names = {t.name for t in tools}
        for excluded in {"git_status", "git_diff", "git_log", "git_commit"}:
            assert excluded not in names

    def test_stable_order(self, tmp_path: Path) -> None:
        tools = create_coding_tools(cwd=tmp_path)
        names = [t.name for t in tools]
        # Git tools after bash, before web tools.
        git_start = names.index("git_status")
        git_end = names.index("git_commit")
        assert names.index("bash") < git_start
        assert git_end < names.index("web_fetch")

    def test_approval_flags(self, tmp_path: Path) -> None:
        tools = create_coding_tools(cwd=tmp_path)
        for tool in tools:
            if tool.name == "git_commit":
                assert tool.requires_approval is True
            elif tool.name in {"git_status", "git_diff", "git_log"}:
                assert tool.requires_approval is False, tool.name


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------


class TestGitToolArgValidation:
    def test_git_commit_requires_message(self, git_repo: Path) -> None:
        async def run() -> None:
            tool = create_git_commit_tool(cwd=git_repo)
            # Missing required "message" key should raise GitToolError.
            with pytest.raises(GitToolError, match="message must be a string"):
                await tool.execute({})

        asyncio.run(run())

    def test_git_log_max_count_must_be_positive(self, git_repo: Path) -> None:
        async def run() -> None:
            tool = create_git_log_tool(cwd=git_repo)
            result = await tool.execute({"max_count": 0})
            assert result.ok is False
            assert "at least 1" in result.content.lower()

        asyncio.run(run())

    def test_bool_as_max_count_rejected(self, git_repo: Path) -> None:
        async def run() -> None:
            tool = create_git_log_tool(cwd=git_repo)
            result = await tool.execute({"max_count": True})
            assert result.ok is False
            assert "integer" in result.content.lower()

        asyncio.run(run())
