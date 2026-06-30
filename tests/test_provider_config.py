"""Tests for durable DeepSeek/OpenAI-compatible provider settings."""

import json
from pathlib import Path

import pytest

from axis_coding import (
    AxisPaths,
    FileCredentialStore,
    ProviderConfigError,
    ScopedModelConfig,
    load_provider_settings,
    openai_compatible_config_from_provider,
    provider_has_usable_credentials,
    provider_settings_path,
    save_provider_settings,
    set_default_provider_model,
    toggle_scoped_model,
)


def _paths(tmp_path: Path) -> AxisPaths:
    return AxisPaths(home=tmp_path / ".axis", agents_home=tmp_path / ".agents")


def test_default_provider_is_deepseek_and_settings_round_trip(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    settings = load_provider_settings(paths)
    provider = settings.get_provider()

    assert settings.default_provider == "deepseek"
    assert provider.api_key_env == "DEEPSEEK_API_KEY"
    assert provider.thinking_levels == ("high", "xhigh")
    path = save_provider_settings(settings, paths)
    assert path == provider_settings_path(paths)
    assert load_provider_settings(paths) == settings


def test_provider_settings_reject_unknown_fields_and_invalid_models(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    settings = load_provider_settings(paths).to_json()
    settings["unknown"] = True
    provider_settings_path(paths).parent.mkdir(parents=True)
    provider_settings_path(paths).write_text(json.dumps(settings), encoding="utf-8")

    with pytest.raises(ProviderConfigError, match="unknown fields"):
        load_provider_settings(paths)

    del settings["unknown"]
    settings["providers"][0]["default_model"] = "missing"
    provider_settings_path(paths).write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ProviderConfigError, match="default_model"):
        load_provider_settings(paths)


def test_provider_credentials_thinking_and_scoped_models(tmp_path: Path) -> None:
    settings = load_provider_settings(_paths(tmp_path))
    provider = settings.get_provider("deepseek")
    store = FileCredentialStore(tmp_path / "credentials.json")

    assert not provider_has_usable_credentials(
        provider,
        credential_reader=store,
        environment={},
    )
    store.set("deepseek", "stored-key")
    assert provider_has_usable_credentials(
        provider,
        credential_reader=store,
        environment={},
    )
    runtime = openai_compatible_config_from_provider(
        provider,
        credential_reader=store,
        model=provider.default_model,
        thinking_level="xhigh",
        environment={},
    )
    assert runtime.api_key == "stored-key"
    assert runtime.thinking_enabled is True
    assert runtime.reasoning_effort == "max"

    scoped = toggle_scoped_model(
        settings,
        provider_name="deepseek",
        model=provider.default_model,
    )
    assert scoped.scoped_models == (
        ScopedModelConfig(provider="deepseek", model=provider.default_model),
    )
    assert (
        toggle_scoped_model(
            scoped,
            provider_name="deepseek",
            model=provider.default_model,
        ).scoped_models
        == ()
    )
    assert (
        set_default_provider_model(
            settings,
            provider_name="deepseek",
            model=provider.default_model,
        ).default_provider
        == "deepseek"
    )
