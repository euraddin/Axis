"""Tests for durable Axis TUI settings and theme contracts."""

from pathlib import Path

import pytest

from axis_coding import AxisPaths
from axis_coding.tui.config import (
    AXIS_DARK_THEME,
    AXIS_LIGHT_THEME,
    HIGH_CONTRAST_THEME,
    OMNI_THEME,
    TERMINAL_NATIVE_THEME,
    TuiConfigError,
    TuiKeybindings,
    TuiSettings,
    get_tui_theme,
    load_tui_settings,
    save_tui_settings,
    tui_settings_from_json,
    tui_settings_path,
)


def _paths(tmp_path: Path) -> AxisPaths:
    return AxisPaths(home=tmp_path / ".axis", agents_home=tmp_path / ".agents")


def test_tui_settings_path_uses_axis_home(tmp_path: Path) -> None:
    assert tui_settings_path(_paths(tmp_path)) == tmp_path / ".axis" / "tui.json"


def test_missing_settings_file_uses_complete_defaults(tmp_path: Path) -> None:
    settings = load_tui_settings(_paths(tmp_path))

    assert settings == TuiSettings()
    assert settings.theme == "axis-dark"
    assert settings.keybindings.cancel == "escape"
    assert settings.keybindings.quit == "ctrl+d"


def test_load_tui_settings_merges_partial_keybindings(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    path = tui_settings_path(paths)
    path.parent.mkdir(parents=True)
    path.write_text(
        """
        {
          "auto_copy_selection": true,
          "keybindings": {
            "cancel": "f4",
            "command_palette": "ctrl+j",
            "quit": "f10"
          },
          "theme": "high-contrast"
        }
        """,
        encoding="utf-8",
    )

    settings = load_tui_settings(paths)

    assert settings.auto_copy_selection is True
    assert settings.theme == "high-contrast"
    assert settings.resolved_theme == HIGH_CONTRAST_THEME
    assert settings.keybindings.cancel == "f4"
    assert settings.keybindings.command_palette == "ctrl+j"
    assert settings.keybindings.quit == "f10"
    assert settings.keybindings.toggle_tool_results == "ctrl+o"


def test_save_tui_settings_writes_stable_json(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    settings = TuiSettings(
        keybindings=TuiKeybindings(cancel="f4"),
        theme="axis-light",
        auto_copy_selection=True,
    )

    path = save_tui_settings(settings, paths)

    assert path.read_text(encoding="utf-8").endswith("\n")
    assert load_tui_settings(paths) == settings


def test_load_tui_settings_reports_invalid_json_with_path(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    path = tui_settings_path(paths)
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")

    with pytest.raises(TuiConfigError, match=str(path)):
        load_tui_settings(paths)


def test_tui_settings_must_be_json_object(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    path = tui_settings_path(paths)
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(TuiConfigError, match="JSON object"):
        load_tui_settings(paths)


def test_tui_settings_reject_unknown_top_level_field() -> None:
    with pytest.raises(TuiConfigError, match="Unknown TUI settings field: palette"):
        tui_settings_from_json({"palette": {}})


def test_tui_settings_reject_unknown_keybinding() -> None:
    with pytest.raises(TuiConfigError, match="Unknown TUI keybinding: launch"):
        tui_settings_from_json({"keybindings": {"launch": "f1"}})


def test_tui_settings_ignore_removed_message_selection_bindings() -> None:
    settings = tui_settings_from_json(
        {"keybindings": {"message_previous": "alt+up", "message_next": "alt+down"}}
    )

    assert settings == TuiSettings()


def test_tui_keybindings_reject_empty_and_duplicate_keys() -> None:
    with pytest.raises(TuiConfigError, match="non-empty string: cancel"):
        tui_settings_from_json({"keybindings": {"cancel": "  "}})

    with pytest.raises(TuiConfigError, match="assigned to both"):
        tui_settings_from_json({"keybindings": {"cancel": "escape", "command_palette": "escape"}})


def test_tui_settings_reject_invalid_boolean_and_theme() -> None:
    with pytest.raises(TuiConfigError, match="auto_copy_selection"):
        tui_settings_from_json({"auto_copy_selection": "yes"})

    with pytest.raises(TuiConfigError, match="Unknown TUI theme: solarized"):
        tui_settings_from_json({"theme": "solarized"})


def test_tui_settings_serialization_contains_every_keybinding() -> None:
    serialized = TuiSettings().to_json()

    assert serialized == {
        "auto_copy_selection": False,
        "keybindings": TuiKeybindings().to_json(),
        "theme": "axis-dark",
    }
    assert set(serialized["keybindings"]) == {
        "cancel",
        "command_palette",
        "session_picker",
        "queue_follow_up",
        "accept_completion",
        "completion_next",
        "completion_previous",
        "thinking_cycle",
        "model_cycle",
        "toggle_thinking",
        "toggle_tool_results",
        "copy_message",
        "voice_record",
        "quit",
    }


def test_builtin_themes_match_axis_brand_names_and_tau_visual_contract() -> None:
    assert get_tui_theme() == AXIS_DARK_THEME
    assert get_tui_theme("axis-light") == AXIS_LIGHT_THEME
    assert get_tui_theme("high-contrast") == HIGH_CONTRAST_THEME
    assert get_tui_theme("omni") == OMNI_THEME
    assert get_tui_theme("terminal-native") == TERMINAL_NATIVE_THEME
    assert AXIS_DARK_THEME.screen_background == "#000000"
    assert AXIS_DARK_THEME.accent == "#db945a"
    assert AXIS_LIGHT_THEME.screen_background == "#ffffff"
    assert AXIS_LIGHT_THEME.syntax_theme == "ansi_light"
    assert HIGH_CONTRAST_THEME.prompt_border == "#00ff66"
    assert OMNI_THEME.screen_background == "#191622"
    assert OMNI_THEME.screen_text == "#E1E1E6"
    assert OMNI_THEME.accent == "#FF79C6"
    assert OMNI_THEME.markdown_inline_code == "#67E480"
    assert TERMINAL_NATIVE_THEME.screen_background == "default"
    assert TERMINAL_NATIVE_THEME.screen_text == "default"
    assert TERMINAL_NATIVE_THEME.prompt_background == "default"
    assert TERMINAL_NATIVE_THEME.completion_selected == "bold black on bright_white"


def test_omni_theme_round_trips_through_settings(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    settings = TuiSettings(theme="omni")

    save_tui_settings(settings, paths)

    assert load_tui_settings(paths) == settings
    assert load_tui_settings(paths).resolved_theme == OMNI_THEME


def test_terminal_native_theme_round_trips_through_settings(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    settings = TuiSettings(theme="terminal-native")

    save_tui_settings(settings, paths)

    assert load_tui_settings(paths) == settings
    assert load_tui_settings(paths).resolved_theme == TERMINAL_NATIVE_THEME
