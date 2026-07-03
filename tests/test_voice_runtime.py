from pathlib import Path

import pytest

from axis_coding.credentials import FileCredentialStore, credentials_path
from axis_coding.paths import AxisPaths
from axis_coding.voice import create_voice_input_controller, resolve_voice_api_key
from axis_coding.voice.config import VOLCENGINE_ASR_CREDENTIAL_NAME


def _paths(tmp_path: Path) -> AxisPaths:
    return AxisPaths(home=tmp_path / ".axis", agents_home=tmp_path / ".agents")


def test_voice_api_key_prefers_private_store_then_environment(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    environment = {"VOLCENGINE_ASR_API_KEY": "environment-key"}
    assert resolve_voice_api_key(paths, environment=environment) == "environment-key"

    FileCredentialStore(credentials_path(paths)).set(VOLCENGINE_ASR_CREDENTIAL_NAME, "stored-key")
    assert resolve_voice_api_key(paths, environment=environment) == "stored-key"


def test_voice_runtime_requires_asr_credentials(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="/voice setup"):
        create_voice_input_controller(paths=_paths(tmp_path))
