"""Durable configuration for Axis's Textual frontend."""

from dataclasses import dataclass, field
from json import JSONDecodeError, dumps, loads
from pathlib import Path
from typing import Any, Literal

from axis_coding.paths import AxisPaths


class TuiConfigError(ValueError):
    """Axis TUI settings are invalid or cannot be read."""


@dataclass(frozen=True, slots=True)
class TuiKeybindings:
    """Configurable keys for the complete Axis TUI interaction model."""

    cancel: str = "escape"
    command_palette: str = "ctrl+k"
    session_picker: str = "ctrl+r"
    queue_follow_up: str = "alt+enter"
    accept_completion: str = "tab"
    completion_next: str = "down"
    completion_previous: str = "up"
    thinking_cycle: str = "shift+tab"
    model_cycle: str = "ctrl+p"
    toggle_thinking: str = "ctrl+t"
    toggle_tool_results: str = "ctrl+o"
    copy_message: str = "ctrl+c"
    voice_record: str = "f2"
    quit: str = "ctrl+d"

    def to_json(self) -> dict[str, str]:
        """Return the stable settings-file representation."""
        return {
            "cancel": self.cancel,
            "command_palette": self.command_palette,
            "session_picker": self.session_picker,
            "queue_follow_up": self.queue_follow_up,
            "accept_completion": self.accept_completion,
            "completion_next": self.completion_next,
            "completion_previous": self.completion_previous,
            "thinking_cycle": self.thinking_cycle,
            "model_cycle": self.model_cycle,
            "toggle_thinking": self.toggle_thinking,
            "toggle_tool_results": self.toggle_tool_results,
            "copy_message": self.copy_message,
            "voice_record": self.voice_record,
            "quit": self.quit,
        }


type TuiThemeName = Literal[
    "axis-dark",
    "axis-light",
    "high-contrast",
    "omni",
    "terminal-native",
]


@dataclass(frozen=True, slots=True)
class TuiRoleStyle:
    """Border and body styles for one transcript role."""

    border: str
    body: str


@dataclass(frozen=True, slots=True)
class TuiTheme:
    """All colors needed by Axis's chrome, transcript, and Markdown widgets."""

    name: TuiThemeName
    screen_background: str
    screen_text: str
    chrome_background: str
    chrome_text: str
    muted_text: str
    sidebar_background: str
    border: str
    transcript_background: str
    prompt_background: str
    prompt_text: str
    prompt_border: str
    autocomplete_background: str
    accent: str
    highlight_background: str
    highlight_text: str
    markdown_heading: str
    markdown_table_header: str
    markdown_table_border: str
    markdown_inline_code: str
    markdown_code_block_background: str
    markdown_link: str
    markdown_bullet: str
    completion_selected: str
    completion_selected_description: str
    completion_description: str
    syntax_theme: str
    role_styles: dict[str, TuiRoleStyle]


AXIS_DARK_THEME = TuiTheme(
    name="axis-dark",
    screen_background="#000000",
    screen_text="#d8dee9",
    chrome_background="#000000",
    chrome_text="#d8dee9",
    muted_text="#667085",
    sidebar_background="#000000",
    border="#141922",
    transcript_background="#000000",
    prompt_background="#101419",
    prompt_text="#e5e7eb",
    prompt_border="#2d3748",
    autocomplete_background="#000000",
    accent="#db945a",
    highlight_background="#a7f3f0",
    highlight_text="#061a1a",
    markdown_heading="#db945a",
    markdown_table_header="#7b7b7b",
    markdown_table_border="#7b7b7b",
    markdown_inline_code="#759e95",
    markdown_code_block_background="#161b21",
    markdown_link="#93c5fd",
    markdown_bullet="#db945a",
    completion_selected="bold #061a1a on #a7f3f0",
    completion_selected_description="#123333 on #a7f3f0",
    completion_description="#667085",
    syntax_theme="ansi_dark",
    role_styles={
        "user": TuiRoleStyle(border="#7c8ea6", body="#d8dee9 on #000000"),
        "assistant": TuiRoleStyle(border="#6ea6a0", body="#d8dee9 on #000000"),
        "tool": TuiRoleStyle(border="#8a7a52", body="#cbd5e1 on #000000"),
        "error": TuiRoleStyle(border="#ff4f4f", body="#ffb4b4 on #000000"),
        "status": TuiRoleStyle(border="#526070", body="#aab4c2 on #000000"),
        "thinking": TuiRoleStyle(border="#4b5563", body="#9ca3af on #000000"),
        "skill": TuiRoleStyle(border="#b48ead", body="#e5d4ef on #000000"),
        "branch_summary": TuiRoleStyle(border="#c084fc", body="#e9d5ff on #000000"),
        "compaction_summary": TuiRoleStyle(border="#c084fc", body="#e9d5ff on #000000"),
    },
)

AXIS_LIGHT_THEME = TuiTheme(
    name="axis-light",
    screen_background="#ffffff",
    screen_text="#111827",
    chrome_background="#f3f4f6",
    chrome_text="#111827",
    muted_text="#475569",
    sidebar_background="#f8fafc",
    border="#cbd5e1",
    transcript_background="#ffffff",
    prompt_background="#f8fafc",
    prompt_text="#111827",
    prompt_border="#2563eb",
    autocomplete_background="#ffffff",
    accent="#0f766e",
    highlight_background="#dbeafe",
    highlight_text="#1d4ed8",
    markdown_heading="#b45309",
    markdown_table_header="#64748b",
    markdown_table_border="#cbd5e1",
    markdown_inline_code="#0f766e",
    markdown_code_block_background="#f1f5f9",
    markdown_link="#2563eb",
    markdown_bullet="#b45309",
    completion_selected="bold #0f172a on #dbeafe",
    completion_selected_description="#334155 on #dbeafe",
    completion_description="#667085",
    syntax_theme="ansi_light",
    role_styles={
        "user": TuiRoleStyle(border="#2563eb", body="#111827"),
        "assistant": TuiRoleStyle(border="#0f766e", body="#111827"),
        "tool": TuiRoleStyle(border="#a16207", body="#1f2937"),
        "error": TuiRoleStyle(border="#b91c1c", body="#7f1d1d"),
        "status": TuiRoleStyle(border="#64748b", body="#334155"),
        "thinking": TuiRoleStyle(border="#6b7280", body="#4b5563"),
        "skill": TuiRoleStyle(border="#7c3aed", body="#4c1d95"),
        "branch_summary": TuiRoleStyle(border="#9333ea", body="#581c87"),
        "compaction_summary": TuiRoleStyle(border="#9333ea", body="#581c87"),
    },
)

HIGH_CONTRAST_THEME = TuiTheme(
    name="high-contrast",
    screen_background="#000000",
    screen_text="#ffffff",
    chrome_background="#111111",
    chrome_text="#ffffff",
    muted_text="#d0d0d0",
    sidebar_background="#111111",
    border="#888888",
    transcript_background="#000000",
    prompt_background="#1a1a1a",
    prompt_text="#ffffff",
    prompt_border="#00ff66",
    autocomplete_background="#111111",
    accent="#ffb454",
    highlight_background="#7fffd4",
    highlight_text="#000000",
    markdown_heading="#ffb454",
    markdown_table_header="#d0d0d0",
    markdown_table_border="#d0d0d0",
    markdown_inline_code="#7fffd4",
    markdown_code_block_background="#161b21",
    markdown_link="#80d8ff",
    markdown_bullet="#ffb454",
    completion_selected="bold black on #7fffd4",
    completion_selected_description="black on #7fffd4",
    completion_description="white",
    syntax_theme="ansi_dark",
    role_styles={
        "user": TuiRoleStyle(border="#00b7ff", body="white on #001626"),
        "assistant": TuiRoleStyle(border="#00ff66", body="white on #001a0b"),
        "tool": TuiRoleStyle(border="#ffd000", body="white on #211900"),
        "error": TuiRoleStyle(border="#ff4f4f", body="white on #260000"),
        "status": TuiRoleStyle(border="#ffffff", body="white on #111111"),
        "thinking": TuiRoleStyle(border="#00b7ff", body="white on #001626"),
        "skill": TuiRoleStyle(border="#ff8cff", body="white on #260026"),
        "branch_summary": TuiRoleStyle(border="#d8b4fe", body="white on #260026"),
        "compaction_summary": TuiRoleStyle(border="#d8b4fe", body="white on #260026"),
    },
)

# Inspired by Omni Theme's MIT-licensed palette:
# https://github.com/getomni/omni
OMNI_THEME = TuiTheme(
    name="omni",
    screen_background="#191622",
    screen_text="#E1E1E6",
    chrome_background="#15121E",
    chrome_text="#E1E1E6",
    muted_text="#5A4B81",
    sidebar_background="#13111B",
    border="#5A4B81",
    transcript_background="#191622",
    prompt_background="#201B2D",
    prompt_text="#E1E1E6",
    prompt_border="#988BC7",
    autocomplete_background="#13111B",
    accent="#FF79C6",
    highlight_background="#41414D",
    highlight_text="#E1E1E6",
    markdown_heading="#78D1E1",
    markdown_table_header="#988BC7",
    markdown_table_border="#5A4B81",
    markdown_inline_code="#67E480",
    markdown_code_block_background="#201B2D",
    markdown_link="#78D1E1",
    markdown_bullet="#FF79C6",
    completion_selected="bold #E1E1E6 on #41414D",
    completion_selected_description="#E1E1E6 on #41414D",
    completion_description="#5A4B81",
    syntax_theme="ansi_dark",
    role_styles={
        "user": TuiRoleStyle(border="#988BC7", body="#E1E1E6 on #191622"),
        "assistant": TuiRoleStyle(border="#67E480", body="#E1E1E6 on #191622"),
        "tool": TuiRoleStyle(border="#E89E64", body="#E7DE79 on #191622"),
        "error": TuiRoleStyle(border="#E96379", body="#E96379 on #191622"),
        "status": TuiRoleStyle(border="#5A4B81", body="#988BC7 on #191622"),
        "thinking": TuiRoleStyle(border="#78D1E1", body="#988BC7 on #191622"),
        "skill": TuiRoleStyle(border="#FF79C6", body="#FF79C6 on #191622"),
        "branch_summary": TuiRoleStyle(border="#78D1E1", body="#78D1E1 on #191622"),
        "compaction_summary": TuiRoleStyle(border="#78D1E1", body="#78D1E1 on #191622"),
    },
)

TERMINAL_NATIVE_THEME = TuiTheme(
    name="terminal-native",
    screen_background="default",
    screen_text="default",
    chrome_background="default",
    chrome_text="default",
    muted_text="bright_black",
    sidebar_background="default",
    border="bright_black",
    transcript_background="default",
    prompt_background="default",
    prompt_text="default",
    prompt_border="bright_black",
    autocomplete_background="default",
    accent="bright_yellow",
    highlight_background="bright_white",
    highlight_text="black",
    markdown_heading="bright_yellow",
    markdown_table_header="default",
    markdown_table_border="bright_black",
    markdown_inline_code="bright_cyan",
    markdown_code_block_background="default",
    markdown_link="bright_blue",
    markdown_bullet="bright_yellow",
    completion_selected="bold black on bright_white",
    completion_selected_description="black on bright_white",
    completion_description="default",
    syntax_theme="ansi_dark",
    role_styles={
        "user": TuiRoleStyle(border="bright_blue", body="default"),
        "assistant": TuiRoleStyle(border="bright_green", body="default"),
        "tool": TuiRoleStyle(border="bright_yellow", body="default"),
        "error": TuiRoleStyle(border="bright_red", body="bold bright_red"),
        "status": TuiRoleStyle(border="bright_black", body="default"),
        "thinking": TuiRoleStyle(border="bright_cyan", body="default"),
        "skill": TuiRoleStyle(border="bright_magenta", body="default"),
        "branch_summary": TuiRoleStyle(border="bright_magenta", body="default"),
        "compaction_summary": TuiRoleStyle(border="bright_magenta", body="default"),
    },
)

_THEMES: dict[TuiThemeName, TuiTheme] = {
    AXIS_DARK_THEME.name: AXIS_DARK_THEME,
    AXIS_LIGHT_THEME.name: AXIS_LIGHT_THEME,
    HIGH_CONTRAST_THEME.name: HIGH_CONTRAST_THEME,
    OMNI_THEME.name: OMNI_THEME,
    TERMINAL_NATIVE_THEME.name: TERMINAL_NATIVE_THEME,
}
BUILTIN_TUI_THEME_NAMES: tuple[TuiThemeName, ...] = tuple(_THEMES)

_RICH_TO_TEXTUAL_ANSI_COLORS = {
    "default": "ansi_default",
    "black": "ansi_black",
    "red": "ansi_red",
    "green": "ansi_green",
    "yellow": "ansi_yellow",
    "blue": "ansi_blue",
    "magenta": "ansi_magenta",
    "cyan": "ansi_cyan",
    "white": "ansi_white",
    "bright_black": "ansi_bright_black",
    "bright_red": "ansi_bright_red",
    "bright_green": "ansi_bright_green",
    "bright_yellow": "ansi_bright_yellow",
    "bright_blue": "ansi_bright_blue",
    "bright_magenta": "ansi_bright_magenta",
    "bright_cyan": "ansi_bright_cyan",
    "bright_white": "ansi_bright_white",
}


def textual_color(color: str) -> str:
    """Translate Rich ANSI names at boundaries that expect Textual colors."""
    return _RICH_TO_TEXTUAL_ANSI_COLORS.get(color, color)


def get_tui_theme(name: TuiThemeName = "axis-dark") -> TuiTheme:
    """Resolve one built-in theme."""
    return _THEMES[name]


@dataclass(frozen=True, slots=True)
class TuiSettings:
    """Durable Axis TUI settings loaded from ``~/.axis/tui.json``."""

    keybindings: TuiKeybindings = field(default_factory=TuiKeybindings)
    theme: TuiThemeName = "axis-dark"
    auto_copy_selection: bool = False

    @property
    def resolved_theme(self) -> TuiTheme:
        """Resolve the selected built-in theme."""
        return get_tui_theme(self.theme)

    def to_json(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible mapping."""
        return {
            "auto_copy_selection": self.auto_copy_selection,
            "keybindings": self.keybindings.to_json(),
            "theme": self.theme,
        }


def tui_settings_path(paths: AxisPaths | None = None) -> Path:
    """Return the durable settings path without creating it."""
    return (paths or AxisPaths()).home / "tui.json"


def load_tui_settings(paths: AxisPaths | None = None) -> TuiSettings:
    """Load settings, returning defaults when the file is absent."""
    path = tui_settings_path(paths)
    if not path.exists():
        return TuiSettings()
    try:
        raw = loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, JSONDecodeError) as exc:
        raise TuiConfigError(f"Could not load TUI settings from {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TuiConfigError("TUI settings must be a JSON object")
    return tui_settings_from_json(raw)


def save_tui_settings(settings: TuiSettings, paths: AxisPaths | None = None) -> Path:
    """Write settings as readable stable JSON and return the path."""
    path = tui_settings_path(paths)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(settings.to_json(), indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise TuiConfigError(f"Could not save TUI settings to {path}: {exc}") from exc
    return path


def tui_settings_from_json(data: dict[str, Any]) -> TuiSettings:
    """Strictly parse a JSON-compatible settings object."""
    allowed_fields = {"auto_copy_selection", "keybindings", "theme"}
    if unknown_fields := set(data) - allowed_fields:
        raise TuiConfigError(f"Unknown TUI settings field: {sorted(unknown_fields)[0]}")

    raw_keybindings = data.get("keybindings", {})
    if not isinstance(raw_keybindings, dict):
        raise TuiConfigError("TUI keybindings must be a JSON object")
    return TuiSettings(
        keybindings=_keybindings_from_json(raw_keybindings),
        theme=_theme_name(data.get("theme", "axis-dark")),
        auto_copy_selection=_bool_setting(
            data.get("auto_copy_selection", False),
            "auto_copy_selection",
        ),
    )


def _keybindings_from_json(data: dict[str, Any]) -> TuiKeybindings:
    defaults = TuiKeybindings()
    default_values = defaults.to_json()
    legacy_fields = {"message_previous", "message_next"}
    if unknown_fields := set(data) - set(default_values) - legacy_fields:
        raise TuiConfigError(f"Unknown TUI keybinding: {sorted(unknown_fields)[0]}")
    values = {
        name: _key_string(data.get(name, default), name) for name, default in default_values.items()
    }
    _reject_duplicate_keys(values)
    return TuiKeybindings(**values)


def _key_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TuiConfigError(f"TUI keybinding must be a non-empty string: {field_name}")
    return value.strip()


def _bool_setting(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TuiConfigError(f"TUI setting must be a boolean: {field_name}")
    return value


def _theme_name(value: object) -> TuiThemeName:
    if not isinstance(value, str) or not value.strip():
        raise TuiConfigError("TUI theme must be a non-empty string")
    name = value.strip()
    if name in _THEMES:
        return name
    raise TuiConfigError(f"Unknown TUI theme: {name}")


def _reject_duplicate_keys(values: dict[str, str]) -> None:
    assigned: dict[str, str] = {}
    for action, key in values.items():
        if previous_action := assigned.get(key):
            raise TuiConfigError(
                f"TUI keybinding {key!r} is assigned to both {previous_action!r} and {action!r}"
            )
        assigned[key] = action
