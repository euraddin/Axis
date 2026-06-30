"""Tests for private, fail-closed Axis credential storage."""

import json
from pathlib import Path

import pytest

from axis_coding import CredentialStoreError, FileCredentialStore


def test_credentials_round_trip_with_owner_only_permissions(tmp_path: Path) -> None:
    path = tmp_path / ".axis" / "credentials.json"
    store = FileCredentialStore(path)

    store.set("deepseek", " secret-key ")

    assert store.get("deepseek") == "secret-key"
    assert store.names() == ("deepseek",)
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text(encoding="utf-8")) == {"deepseek": "secret-key"}
    assert store.delete("deepseek") is True
    assert store.delete("deepseek") is False


def test_credentials_hard_fail_on_corruption(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    path.write_text('{"deepseek": 42}\n', encoding="utf-8")

    with pytest.raises(CredentialStoreError, match="non-empty string"):
        FileCredentialStore(path).get("deepseek")

    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(CredentialStoreError, match="JSON object"):
        FileCredentialStore(path).names()
