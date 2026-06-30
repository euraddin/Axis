"""Pure prompt-completion primitives for Axis's Textual frontend."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from axis_coding.prompt_templates import PromptTemplate
from axis_coding.skills import Skill

IGNORED_FILE_COMPLETION_DIRS = frozenset(
    {
        ".axis",
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)
MAX_FILE_COMPLETIONS = 50


@dataclass(frozen=True, slots=True)
class CompletionCommand:
    """Metadata for one slash command that is genuinely available."""

    name: str
    description: str
    aliases: tuple[str, ...] = ()
    search_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CompletionOption:
    """One possible command-argument value."""

    value: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class CompletionItem:
    """One selectable replacement over a half-open input-text span."""

    display: str
    replacement: str
    start: int
    end: int
    description: str | None = None
    category: str | None = None

    def apply(self, text: str) -> str:
        """Apply the completion without disturbing text outside its span."""
        return f"{text[: self.start]}{self.replacement}{text[self.end :]}"


@dataclass(frozen=True, slots=True)
class CompletionState:
    """Immutable suggestions plus the currently selected row."""

    items: tuple[CompletionItem, ...] = ()
    selected_index: int = 0

    @property
    def selected(self) -> CompletionItem | None:
        """Return the selected item, if suggestions exist."""
        if not self.items:
            return None
        return self.items[self.selected_index]

    def select_next(self) -> CompletionState:
        """Move selection forward, wrapping at the end."""
        if not self.items:
            return self
        return CompletionState(self.items, (self.selected_index + 1) % len(self.items))

    def select_previous(self) -> CompletionState:
        """Move selection backward, wrapping at the start."""
        if not self.items:
            return self
        return CompletionState(self.items, (self.selected_index - 1) % len(self.items))


def build_completion_state(
    text: str,
    *,
    commands: Sequence[CompletionCommand] = (),
    skills: Sequence[Skill] = (),
    prompt_templates: Sequence[PromptTemplate] = (),
    argument_options: Mapping[str, Sequence[CompletionOption]] | None = None,
    cwd: Path | None = None,
    shell_paths_enabled: bool = True,
) -> CompletionState:
    """Build deterministic suggestions for input text up to the cursor."""
    if not text.startswith("/") or text.startswith("//"):
        if cwd is None:
            return CompletionState()
        if shell_paths_enabled:
            shell_items = _shell_path_completions(text, cwd=cwd)
            if shell_items is not None:
                return CompletionState(shell_items)
        return CompletionState(_file_reference_completions(text, cwd=cwd))

    token_end = _first_token_end(text)
    token = text[:token_end]
    has_arguments = token_end < len(text)
    if token.startswith("/skill:"):
        if has_arguments and _is_complete_skill(token, skills):
            return CompletionState()
        return CompletionState(_skill_completions(token, token_end=token_end, skills=skills))
    if ":" in token:
        return CompletionState()

    values = argument_options or {}
    argument_items = _argument_completions(text, token_end=token_end, options=values)
    if argument_items is not None:
        return CompletionState(argument_items)

    if has_arguments and (
        _is_complete_command(token, commands) or _is_complete_template(token, prompt_templates)
    ):
        return CompletionState()
    return CompletionState(
        _slash_completions(
            token,
            token_end=token_end,
            commands=commands,
            prompt_templates=prompt_templates,
        )
    )


def _slash_completions(
    token: str,
    *,
    token_end: int,
    commands: Sequence[CompletionCommand],
    prompt_templates: Sequence[PromptTemplate],
) -> tuple[CompletionItem, ...]:
    prefix = token.removeprefix("/").casefold()
    command_items: list[CompletionItem] = []
    for command in commands:
        candidate_names = (
            (command.name,)
            if not prefix
            else (command.name, *command.aliases, *command.search_terms)
        )
        seen: set[str] = set()
        for candidate in candidate_names:
            if not candidate.casefold().startswith(prefix):
                continue
            replacement_name = (
                candidate if candidate in (command.name, *command.aliases) else command.name
            )
            replacement = f"/{replacement_name}"
            if command.name == "skill" and replacement_name == "skill":
                replacement = "/skill:"
            if replacement in seen:
                continue
            seen.add(replacement)
            command_items.append(
                CompletionItem(
                    display=replacement,
                    replacement=replacement,
                    start=0,
                    end=token_end,
                    description=command.description,
                    category="Commands",
                )
            )

    template_items = [
        CompletionItem(
            display=f"/{template.name}",
            replacement=f"/{template.name}",
            start=0,
            end=token_end,
            description=template.description or "Prompt template",
            category="Custom prompts",
        )
        for template in prompt_templates
        if template.name.casefold().startswith(prefix)
    ]
    return (
        *sorted(command_items, key=lambda item: _slash_sort_key(item, prefix)),
        *sorted(template_items, key=lambda item: _slash_sort_key(item, prefix)),
    )


def _slash_sort_key(item: CompletionItem, prefix: str) -> tuple[int, str]:
    name = item.display.removeprefix("/").removesuffix(":").casefold()
    rank = 0 if not prefix or name.startswith(prefix) else 1
    return rank, item.display.casefold()


def _skill_completions(
    token: str,
    *,
    token_end: int,
    skills: Sequence[Skill],
) -> tuple[CompletionItem, ...]:
    prefix = token.removeprefix("/skill:").casefold()
    return tuple(
        CompletionItem(
            display=f"/skill:{skill.name}",
            replacement=f"/skill:{skill.name}",
            start=0,
            end=token_end,
            description=skill.description,
        )
        for skill in sorted(skills, key=lambda item: (item.name.casefold(), item.name))
        if skill.name.casefold().startswith(prefix)
    )


def _argument_completions(
    text: str,
    *,
    token_end: int,
    options: Mapping[str, Sequence[CompletionOption]],
) -> tuple[CompletionItem, ...] | None:
    if token_end >= len(text):
        return None
    command_name = text[:token_end].removeprefix("/").casefold()
    candidates = options.get(command_name)
    if candidates is None:
        return None
    start = token_end + 1
    end = _argument_token_end(text, start)
    prefix = text[start:end].casefold()
    return tuple(
        CompletionItem(
            display=option.value,
            replacement=option.value,
            start=start,
            end=end,
            description=option.description,
        )
        for option in candidates
        if option.value.casefold().startswith(prefix)
    )


def _file_reference_completions(text: str, *, cwd: Path) -> tuple[CompletionItem, ...]:
    span = _active_file_reference_span(text)
    if span is None:
        return ()
    start, end = span
    prefix = text[start + 1 : end].casefold()
    items: list[CompletionItem] = []
    for path in _workspace_paths(cwd):
        relative = path.relative_to(cwd).as_posix()
        if prefix not in relative.casefold():
            continue
        value = f"@{relative}{'/' if path.is_dir() else ''}"
        items.append(
            CompletionItem(
                display=value,
                replacement=value,
                start=start,
                end=end,
                description="Directory" if path.is_dir() else "File reference",
            )
        )
        if len(items) == MAX_FILE_COMPLETIONS:
            break
    return tuple(items)


def _active_file_reference_span(text: str) -> tuple[int, int] | None:
    end = len(text)
    token_start = end
    while token_start > 0 and not text[token_start - 1].isspace():
        token_start -= 1
    at_index = text.rfind("@", token_start, end)
    return None if at_index < 0 else (at_index, end)


def _workspace_paths(cwd: Path) -> tuple[Path, ...]:
    if not cwd.is_dir():
        return ()
    found: list[Path] = []
    pending = [cwd]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(
                directory.iterdir(),
                key=lambda path: (path.name.casefold(), path.name),
                reverse=True,
            )
        except OSError:
            continue
        for child in children:
            if _ignored_path(child, cwd=cwd):
                continue
            found.append(child)
            if child.is_dir() and not child.is_symlink():
                pending.append(child)
    return tuple(sorted(found, key=lambda path: path.relative_to(cwd).as_posix().casefold()))


def _ignored_path(path: Path, *, cwd: Path) -> bool:
    try:
        parts = path.relative_to(cwd).parts
    except ValueError:
        return True
    return any(part.startswith(".") or part in IGNORED_FILE_COMPLETION_DIRS for part in parts)


def _shell_path_completions(
    text: str,
    *,
    cwd: Path,
) -> tuple[CompletionItem, ...] | None:
    command_start = _shell_command_start(text)
    if command_start is None:
        return None
    start, end = _active_shell_token(text, command_start=command_start)
    token = text[start:end]
    if not token:
        return ()
    parsed = _parse_shell_path(token)
    if parsed is None:
        return ()
    parent_text, name_prefix, replacement_prefix = parsed
    parent = cwd / parent_text if parent_text else cwd
    if not parent.is_dir() or (parent != cwd and _ignored_path(parent, cwd=cwd)):
        return ()
    try:
        children = sorted(parent.iterdir(), key=lambda path: (path.name.casefold(), path.name))
    except OSError:
        return ()
    items: list[CompletionItem] = []
    for child in children:
        if _ignored_path(child, cwd=cwd):
            continue
        if not child.name.casefold().startswith(name_prefix.casefold()):
            continue
        relative = child.relative_to(cwd).as_posix()
        replacement = f"{replacement_prefix}{relative}{'/' if child.is_dir() else ''}"
        if replacement == token:
            continue
        items.append(
            CompletionItem(
                display=replacement,
                replacement=replacement,
                start=start,
                end=end,
                description="Directory" if child.is_dir() else "File",
            )
        )
        if len(items) == MAX_FILE_COMPLETIONS:
            break
    return tuple(items)


def _shell_command_start(text: str) -> int | None:
    leading = len(text) - len(text.lstrip())
    stripped = text[leading:]
    if stripped.startswith("!!"):
        return leading + 2
    if stripped.startswith("!"):
        return leading + 1
    return None


def _active_shell_token(text: str, *, command_start: int) -> tuple[int, int]:
    end = len(text)
    token_start = command_start
    escaped = False
    for index in range(command_start, end):
        character = text[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character.isspace():
            token_start = index + 1
    return token_start, end


def _parse_shell_path(token: str) -> tuple[str, str, str] | None:
    replacement_prefix = ""
    path_text = token
    if path_text.startswith("./"):
        replacement_prefix = "./"
        path_text = path_text[2:]
    if path_text.startswith(("/", "~")) or any(character in path_text for character in "\"'`$*?[{"):
        return None
    parent_text, separator, name_prefix = path_text.rpartition("/")
    if separator and not parent_text:
        return None
    if any(part in {"", ".", ".."} for part in parent_text.split("/") if parent_text):
        return None
    return parent_text, name_prefix, replacement_prefix


def _is_complete_skill(token: str, skills: Sequence[Skill]) -> bool:
    name = token.removeprefix("/skill:").casefold()
    return any(skill.name.casefold() == name for skill in skills)


def _is_complete_template(token: str, templates: Sequence[PromptTemplate]) -> bool:
    name = token.removeprefix("/").casefold()
    return any(template.name.casefold() == name for template in templates)


def _is_complete_command(token: str, commands: Sequence[CompletionCommand]) -> bool:
    name = token.removeprefix("/").casefold()
    return any(
        name in {command.name.casefold(), *(alias.casefold() for alias in command.aliases)}
        for command in commands
    )


def _first_token_end(text: str) -> int:
    return next((index for index, character in enumerate(text) if character.isspace()), len(text))


def _argument_token_end(text: str, start: int) -> int:
    return next(
        (index for index in range(start, len(text)) if text[index].isspace()),
        len(text),
    )
