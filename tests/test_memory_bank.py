"""Tests for file-backed Memory Bank loading, proposals, and approved writes."""

from pathlib import Path

import pytest

from axis_agent import MemoryProposalEntry
from axis_coding import (
    MEMORY_ROUTES,
    MemoryBank,
    MemoryBankError,
    MemoryWriter,
    classify_memory_task,
    contains_secret_like,
    parse_memory_proposals,
    render_memory_proposals,
    sanitize_memory_evidence,
)

EXPECTED_FILES = {
    "index.md",
    "projectbrief.md",
    "architecture.md",
    "tech.md",
    "activeContext.md",
    "progress.md",
    "decisions.md",
    "pitfalls.md",
}


def test_memory_bank_initializes_templates_without_overwriting_existing(tmp_path: Path) -> None:
    bank = MemoryBank(tmp_path)
    bank.root.mkdir()
    existing = bank.root / "tech.md"
    existing.write_text("# Custom Tech\n", encoding="utf-8")

    result = bank.initialize()

    assert {path.name for path in bank.root.iterdir()} == EXPECTED_FILES
    assert existing.read_text(encoding="utf-8") == "# Custom Tech\n"
    assert existing in result.existing_files
    assert len(result.created_files) == 7


def test_missing_bank_and_missing_routed_files_are_non_fatal(tmp_path: Path) -> None:
    bank = MemoryBank(tmp_path)

    missing_bank = bank.load("default")
    assert missing_bank.initialized is False
    assert missing_bank.files == ()

    bank.root.mkdir()
    (bank.root / "index.md").write_text("# Index\n", encoding="utf-8")
    partial = bank.load("debug")
    assert [item.name for item in partial.files] == ["index.md"]
    assert len(partial.diagnostics) == 3


@pytest.mark.parametrize("task_type", list(MEMORY_ROUTES))
def test_memory_bank_loads_only_files_routed_for_task(tmp_path: Path, task_type: str) -> None:
    bank = MemoryBank(tmp_path)
    bank.initialize()

    result = bank.load(task_type)  # type: ignore[arg-type]

    assert tuple(item.name for item in result.files) == MEMORY_ROUTES[task_type]  # type: ignore[index]
    assert '<project_memory task_type="' in result.rendered
    assert "lower priority than core system rules" in result.rendered


def test_memory_bank_enforces_per_file_and_total_budgets(tmp_path: Path) -> None:
    bank = MemoryBank(tmp_path, total_token_budget=20, file_token_budget=10)
    bank.initialize()
    for name in MEMORY_ROUTES["default"]:
        (bank.root / name).write_text("x" * 400, encoding="utf-8")

    result = bank.load("default")

    assert sum(item.estimated_tokens for item in result.files) <= 20
    assert all(item.estimated_tokens <= 10 for item in result.files)
    assert any(item.truncated for item in result.files)
    assert result.diagnostics


def test_task_classifier_uses_bilingual_precedence() -> None:
    assert classify_memory_task("这个 traceback 为什么报错") == "debug"
    assert classify_memory_task("Design the module architecture and plan it") == "architecture"
    assert classify_memory_task("先做一个 implementation plan") == "planning"
    assert classify_memory_task("实现新的 loader") == "implementation"
    assert classify_memory_task("解释一下当前项目") == "default"


def test_secret_and_privacy_sanitizer_redacts_sensitive_values(tmp_path: Path) -> None:
    raw = (
        "OPENAI_API_KEY=sk-abcdefghijklmnop\n"
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz\n"
        "mail me at person@example.com\n"
        f"path={tmp_path}/src/app.py"
    )

    sanitized = sanitize_memory_evidence(raw, project_root=tmp_path)

    assert "sk-abcdefghijklmnop" not in sanitized
    assert "abcdefghijklmnopqrstuvwxyz" not in sanitized
    assert "person@example.com" not in sanitized
    assert "<project_root>" in sanitized
    assert contains_secret_like(raw) is True
    assert contains_secret_like("token budget is 16000") is False


def test_proposal_parser_validates_targets_and_filters_secret_content(tmp_path: Path) -> None:
    bank = MemoryBank(tmp_path)
    bank.initialize()
    payload = """{
      "proposals": [{
        "target_file": "decisions.md",
        "operation": "append",
        "section_heading": null,
        "reason": "A durable decision was made.",
        "proposed_content": "## 2026-07-05: Keep proposals reviewable",
        "confidence": 0.95,
        "requires_user_approval": true
      }]
    }"""

    proposals = parse_memory_proposals(
        payload,
        task_type="architecture",
        bank=bank,
        parent_id="parent",
    )

    assert len(proposals) == 1
    assert proposals[0].parent_id == "parent"
    assert proposals[0].requires_user_approval is True
    assert proposals[0].base_sha256 == bank.target_digest("decisions.md")

    unsafe = payload.replace(
        "## 2026-07-05: Keep proposals reviewable",
        "OPENAI_API_KEY=sk-abcdefghijklmnop",
    )
    with pytest.raises(MemoryBankError, match="secret-like"):
        parse_memory_proposals(
            unsafe,
            task_type="architecture",
            bank=bank,
            parent_id=None,
        )

    private = payload.replace(
        "## 2026-07-05: Keep proposals reviewable",
        "Contact person@example.com before applying.",
    )
    with pytest.raises(MemoryBankError, match="private"):
        parse_memory_proposals(
            private,
            task_type="architecture",
            bank=bank,
            parent_id=None,
        )

    extra_field = payload.replace(
        '"requires_user_approval": true',
        '"requires_user_approval": true, "unexpected": "value"',
    )
    with pytest.raises(MemoryBankError, match="strict schema"):
        parse_memory_proposals(
            extra_field,
            task_type="architecture",
            bank=bank,
            parent_id=None,
        )


def test_writer_applies_append_and_section_replace_with_history(tmp_path: Path) -> None:
    bank = MemoryBank(tmp_path)
    bank.initialize()
    writer = MemoryWriter(bank)
    append = MemoryProposalEntry(
        task_type="architecture",
        target_file="decisions.md",
        operation="append",
        reason="Record a decision",
        proposed_content="## Decision\n\nUse proposal mode.",
        confidence=1,
        base_sha256=bank.target_digest("decisions.md"),
    )

    appended = writer.apply(append)

    assert "Use proposal mode" in (bank.root / "decisions.md").read_text(encoding="utf-8")
    assert appended.audit_path.is_file()
    assert "```diff" in appended.audit_path.read_text(encoding="utf-8")

    replace = MemoryProposalEntry(
        task_type="implementation",
        target_file="activeContext.md",
        operation="replace_section",
        section_heading="## Current Focus",
        reason="Focus changed",
        proposed_content="Implement Auto Memory.",
        confidence=0.9,
        base_sha256=bank.target_digest("activeContext.md"),
    )
    writer.apply(replace)
    active = (bank.root / "activeContext.md").read_text(encoding="utf-8")
    assert "## Current Focus\n\nImplement Auto Memory." in active
    assert "## Recent Changes" in active


def test_writer_rejects_stale_or_agents_md_proposals(tmp_path: Path) -> None:
    bank = MemoryBank(tmp_path)
    bank.initialize()
    proposal = MemoryProposalEntry(
        task_type="implementation",
        target_file="tech.md",
        operation="append",
        reason="Add command",
        proposed_content="- `uv run pytest`",
        confidence=0.8,
        base_sha256=bank.target_digest("tech.md"),
    )
    (bank.root / "tech.md").write_text("# Changed\n", encoding="utf-8")
    with pytest.raises(MemoryBankError, match="changed"):
        MemoryWriter(bank).apply(proposal)

    agents = MemoryProposalEntry(
        task_type="default",
        target_file="AGENTS.md suggestion only",
        operation="suggest_promotion_to_agents_md",
        reason="Stable rule",
        proposed_content="Never auto-write memory.",
        confidence=1,
        base_sha256=bank.target_digest("AGENTS.md suggestion only"),
    )
    preview = MemoryWriter(bank).preview(agents)
    assert preview.applicable is False
    with pytest.raises(MemoryBankError, match="not writable"):
        MemoryWriter(bank).apply(agents)


def test_writer_rolls_back_target_when_history_audit_cannot_be_written(tmp_path: Path) -> None:
    bank = MemoryBank(tmp_path)
    bank.initialize()
    target = bank.root / "progress.md"
    original = target.read_text(encoding="utf-8")
    (bank.root / "history").write_text("not a directory", encoding="utf-8")
    proposal = MemoryProposalEntry(
        task_type="implementation",
        target_file="progress.md",
        operation="append",
        reason="Record completed work",
        proposed_content="- Durable milestone.",
        confidence=0.9,
        base_sha256=bank.target_digest("progress.md"),
    )

    with pytest.raises(OSError):
        MemoryWriter(bank).apply(proposal)

    assert target.read_text(encoding="utf-8") == original


def test_proposal_markdown_renderer_includes_diff_and_apply_command(tmp_path: Path) -> None:
    bank = MemoryBank(tmp_path)
    bank.initialize()
    proposal = MemoryProposalEntry(
        task_type="implementation",
        id="proposal-1",
        target_file="progress.md",
        operation="append",
        reason="Work completed",
        proposed_content="- Memory loader implemented.",
        confidence=0.9,
        base_sha256=bank.target_digest("progress.md"),
    )

    rendered = render_memory_proposals((proposal,), writer=MemoryWriter(bank))

    assert "# Memory Update Proposals" in rendered
    assert "proposal-1" in rendered
    assert "```diff" in rendered
    assert "/memory apply proposal-1" in rendered
