from json import loads
from pathlib import Path

import pytest

from axis_coding.paths import AxisPaths
from axis_coding.voice import (
    VoiceConfigError,
    VoiceInputConfig,
    load_voice_config,
    save_voice_config,
    voice_config_from_json,
    voice_settings_path,
)


def _paths(tmp_path: Path) -> AxisPaths:
    return AxisPaths(home=tmp_path / ".axis", agents_home=tmp_path / ".agents")


def test_voice_config_defaults_and_round_trip(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = VoiceInputConfig(input_device=3, max_recording_seconds=42)
    assert config.blocksize == 1_600
    assert voice_settings_path(paths) == tmp_path / ".axis" / "voice.json"
    save_voice_config(config, paths)
    assert load_voice_config(paths) == config
    stored = loads(voice_settings_path(paths).read_text(encoding="utf-8"))
    assert stored["resource_id"] == "volc.seedasr.sauc.duration"
    assert "api_key" not in stored


def test_voice_config_rejects_unknown_and_invalid_values() -> None:
    with pytest.raises(VoiceConfigError, match="Unknown voice settings field"):
        voice_config_from_json({"secret": "nope"})
    with pytest.raises(VoiceConfigError, match="max_recording_seconds"):
        voice_config_from_json({"max_recording_seconds": 0})
    with pytest.raises(VoiceConfigError, match="Unsupported voice config version"):
        voice_config_from_json({"version": 2})
