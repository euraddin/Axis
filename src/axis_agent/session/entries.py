"""Typed facts stored in an append-only Axis session log."""

from time import time
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from axis_agent.messages import AgentMessage


def new_entry_id() -> str:
    """Return a fresh opaque session-entry identifier."""
    return uuid4().hex


def current_timestamp() -> float:
    """Return the current Unix timestamp in seconds."""
    return time()


class BaseSessionEntry(BaseModel):
    """Fields shared by every append-only session fact."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str = Field(default_factory=new_entry_id, min_length=1)
    parent_id: str | None = None
    timestamp: float = Field(default_factory=current_timestamp, ge=0)


class MessageEntry(BaseSessionEntry):
    """One complete message added to the model transcript."""

    type: Literal["message"] = "message"
    message: AgentMessage


class ModelChangeEntry(BaseSessionEntry):
    """A change to the model selected for subsequent requests."""

    type: Literal["model_change"] = "model_change"
    model: str = Field(min_length=1)


class ThinkingLevelChangeEntry(BaseSessionEntry):
    """A reasoning-effort change for subsequent provider requests."""

    type: Literal["thinking_level_change"] = "thinking_level_change"
    thinking_level: str = Field(min_length=1)


class CompactionEntry(BaseSessionEntry):
    """A summary replacing selected earlier message entries during replay."""

    type: Literal["compaction"] = "compaction"
    summary: str = Field(min_length=1)
    replaces_entry_ids: list[str] = Field(default_factory=list)


class BranchSummaryEntry(BaseSessionEntry):
    """A summary of history abandoned when branching from an old entry."""

    type: Literal["branch_summary"] = "branch_summary"
    summary: str = Field(min_length=1)
    branch_root_id: str | None = None


type MemoryTargetFile = Literal[
    "activeContext.md",
    "progress.md",
    "decisions.md",
    "pitfalls.md",
    "tech.md",
    "architecture.md",
    "projectbrief.md",
    "AGENTS.md suggestion only",
]
type MemoryOperation = Literal[
    "append",
    "replace_section",
    "update_checkbox",
    "suggest_promotion_to_agents_md",
]


class MemoryProposalEntry(BaseSessionEntry):
    """One reviewable project-memory update proposed by a completed task."""

    type: Literal["memory_proposal"] = "memory_proposal"
    task_type: str = Field(min_length=1)
    target_file: MemoryTargetFile
    operation: MemoryOperation
    section_heading: str | None = None
    reason: str = Field(min_length=1, max_length=4_000)
    proposed_content: str = Field(min_length=1, max_length=16_000)
    confidence: float = Field(ge=0, le=1)
    requires_user_approval: Literal[True] = True
    base_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MemoryProposalDecisionEntry(BaseSessionEntry):
    """Append-only record that a memory proposal was applied or discarded."""

    type: Literal["memory_proposal_decision"] = "memory_proposal_decision"
    proposal_id: str = Field(min_length=1)
    decision: Literal["applied", "discarded"]
    audit_path: str | None = None
    message: str | None = None


class LeafEntry(BaseSessionEntry):
    """An append-only pointer to the active session-tree leaf."""

    type: Literal["leaf"] = "leaf"
    entry_id: str | None = None


class SessionInfoEntry(BaseSessionEntry):
    """Basic metadata describing one Axis session."""

    type: Literal["session_info"] = "session_info"
    created_at: float = Field(default_factory=current_timestamp, ge=0)
    cwd: str | None = None
    title: str | None = None
    system: str | None = None


type SessionEntry = Annotated[
    MessageEntry
    | ModelChangeEntry
    | ThinkingLevelChangeEntry
    | CompactionEntry
    | BranchSummaryEntry
    | MemoryProposalEntry
    | MemoryProposalDecisionEntry
    | LeafEntry
    | SessionInfoEntry,
    Field(discriminator="type"),
]
