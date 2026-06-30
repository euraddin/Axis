"""Typed summaries for explicit Axis resource reloads."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReloadCategorySummary:
    """Before/after counts and equality for one resource category."""

    before: int
    after: int
    changed: bool

    @property
    def delta(self) -> int:
        return self.after - self.before


@dataclass(frozen=True, slots=True)
class CodingReloadSummary:
    """Complete result of reloading local instructions and resources."""

    skills: ReloadCategorySummary
    prompt_templates: ReloadCategorySummary
    context_files: ReloadCategorySummary
    diagnostics: ReloadCategorySummary
    system_prompt_rebuilt: bool
