"""Durable OpenAI-compatible provider settings for Axis."""

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field, replace
from json import JSONDecodeError, dumps, loads
from os import environ
from pathlib import Path
from shutil import copy2
from tempfile import NamedTemporaryFile
from typing import Any

from axis_ai import OpenAICompatibleConfig
from axis_ai.config import (
    DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES,
    DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS,
    DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
)
from axis_coding.credentials import FileCredentialStore, credentials_path
from axis_coding.paths import AxisPaths
from axis_coding.provider_catalog import BUILTIN_PROVIDER_CATALOG, builtin_provider_entry
from axis_coding.thinking import (
    DEFAULT_THINKING_LEVEL,
    ThinkingLevel,
    ThinkingParameter,
    normalize_thinking_level,
    normalize_thinking_levels,
    reasoning_effort_for_level,
)

DEFAULT_PROVIDER_NAME = "deepseek"


class ProviderConfigError(ValueError):
    """Durable provider settings are invalid or unavailable."""


@dataclass(frozen=True, slots=True)
class OpenAICompatibleProviderConfig:
    """Durable settings for one OpenAI-compatible endpoint."""

    name: str
    base_url: str
    api_key_env: str
    credential_name: str | None
    models: tuple[str, ...]
    default_model: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES
    max_retry_delay_seconds: float = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS
    max_tokens: int | None = None
    context_windows: dict[str, int] = field(default_factory=dict)
    thinking_levels: tuple[ThinkingLevel, ...] | None = None
    thinking_models: tuple[str, ...] = ()
    thinking_default: ThinkingLevel | None = None
    thinking_parameter: ThinkingParameter | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ProviderConfigError("Provider name must not be empty")
        if not self.base_url.strip():
            raise ProviderConfigError(f"Provider base_url must not be empty: {self.name}")
        if not self.api_key_env.strip():
            raise ProviderConfigError(f"Provider api_key_env must not be empty: {self.name}")
        if not self.models or any(not model.strip() for model in self.models):
            raise ProviderConfigError(f"Provider models must be a non-empty list: {self.name}")
        if len(set(self.models)) != len(self.models):
            raise ProviderConfigError(f"Provider models must be unique: {self.name}")
        if self.default_model not in self.models:
            raise ProviderConfigError(
                f"Provider default_model must appear in models: {self.name}:{self.default_model}"
            )
        if self.timeout_seconds <= 0:
            raise ProviderConfigError("Provider timeout_seconds must be greater than 0")
        if self.max_retries < 0 or self.max_retry_delay_seconds < 0:
            raise ProviderConfigError("Provider retry settings must be 0 or greater")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ProviderConfigError("Provider max_tokens must be greater than 0")
        if any(value <= 0 for value in self.context_windows.values()):
            raise ProviderConfigError("Provider context windows must be greater than 0")
        if self.thinking_levels is None:
            if self.thinking_default is not None or self.thinking_models or self.thinking_parameter:
                raise ProviderConfigError("Provider thinking metadata requires thinking_levels")
        else:
            normalized = normalize_thinking_levels(self.thinking_levels)
            if normalized != self.thinking_levels:
                raise ProviderConfigError("Provider thinking_levels must be normalized")
            if self.thinking_default is not None and self.thinking_default not in normalized:
                raise ProviderConfigError(
                    "Provider thinking_default must appear in thinking_levels"
                )
            if self.thinking_parameter is None:
                raise ProviderConfigError("Provider thinking_parameter is required")

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "openai-compatible",
            "name": self.name,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "credential_name": self.credential_name,
            "models": list(self.models),
            "default_model": self.default_model,
            "headers": dict(self.headers),
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_retry_delay_seconds": self.max_retry_delay_seconds,
            "max_tokens": self.max_tokens,
            "context_windows": dict(self.context_windows),
            "thinking_levels": (
                list(self.thinking_levels) if self.thinking_levels is not None else None
            ),
            "thinking_models": list(self.thinking_models),
            "thinking_default": self.thinking_default,
            "thinking_parameter": self.thinking_parameter,
        }


@dataclass(frozen=True, slots=True)
class ScopedModelConfig:
    provider: str
    model: str

    def to_json(self) -> dict[str, str]:
        return {"provider": self.provider, "model": self.model}


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    default_provider: str = DEFAULT_PROVIDER_NAME
    providers: tuple[OpenAICompatibleProviderConfig, ...] = field(
        default_factory=lambda: builtin_provider_configs()
    )
    scoped_models: tuple[ScopedModelConfig, ...] = ()

    def __post_init__(self) -> None:
        if not self.providers:
            raise ProviderConfigError("Provider settings must include at least one provider")
        names = [provider.name for provider in self.providers]
        if len(set(names)) != len(names):
            raise ProviderConfigError("Provider names must be unique")
        self.get_provider(self.default_provider)

    def get_provider(self, name: str | None = None) -> OpenAICompatibleProviderConfig:
        target = name or self.default_provider
        for provider in self.providers:
            if provider.name == target:
                return provider
        raise ProviderConfigError(f"Unknown provider: {target}")

    def to_json(self) -> dict[str, Any]:
        return {
            "default_provider": self.default_provider,
            "providers": [provider.to_json() for provider in self.providers],
            "scoped_models": [choice.to_json() for choice in self.scoped_models],
        }


@dataclass(frozen=True, slots=True)
class ProviderSelection:
    provider: OpenAICompatibleProviderConfig
    model: str


def builtin_provider_configs() -> tuple[OpenAICompatibleProviderConfig, ...]:
    return tuple(
        OpenAICompatibleProviderConfig(
            name=entry.name,
            base_url=entry.base_url,
            api_key_env=entry.api_key_env,
            credential_name=entry.credential_name,
            models=entry.models,
            default_model=entry.default_model,
            thinking_levels=entry.thinking_levels,
            thinking_models=entry.thinking_models,
            thinking_default=entry.thinking_default,
            thinking_parameter=entry.thinking_parameter,
        )
        for entry in BUILTIN_PROVIDER_CATALOG
    )


def provider_config_from_catalog_entry(name: str) -> OpenAICompatibleProviderConfig:
    entry = builtin_provider_entry(name)
    if entry is None:
        raise ProviderConfigError(f"Unknown built-in provider: {name}")
    return OpenAICompatibleProviderConfig(
        name=entry.name,
        base_url=entry.base_url,
        api_key_env=entry.api_key_env,
        credential_name=entry.credential_name,
        models=entry.models,
        default_model=entry.default_model,
        thinking_levels=entry.thinking_levels,
        thinking_models=entry.thinking_models,
        thinking_default=entry.thinking_default,
        thinking_parameter=entry.thinking_parameter,
    )


def upsert_provider(
    settings: ProviderSettings,
    provider: OpenAICompatibleProviderConfig,
) -> ProviderSettings:
    """Return settings containing a new or replacement provider entry."""
    providers = {item.name: item for item in settings.providers}
    providers[provider.name] = provider
    default = settings.default_provider
    if default not in providers:
        default = provider.name
    return ProviderSettings(
        default_provider=default,
        providers=tuple(providers[name] for name in sorted(providers)),
        scoped_models=settings.scoped_models,
    )


def provider_settings_path(paths: AxisPaths | None = None) -> Path:
    return (paths or AxisPaths()).home / "providers.json"


def load_provider_settings(paths: AxisPaths | None = None) -> ProviderSettings:
    path = provider_settings_path(paths)
    if not path.exists():
        return ProviderSettings()
    try:
        raw = loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, JSONDecodeError) as exc:
        raise ProviderConfigError(f"Could not load provider settings from {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProviderConfigError("Provider settings must be a JSON object")
    return provider_settings_from_json(raw)


def save_provider_settings(
    settings: ProviderSettings,
    paths: AxisPaths | None = None,
) -> Path:
    path = provider_settings_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with suppress(OSError):
            copy2(path, path.with_suffix(".json.bak"))
    _atomic_write(path, dumps(settings.to_json(), indent=2, sort_keys=True) + "\n")
    return path


def provider_settings_from_json(data: Mapping[str, object]) -> ProviderSettings:
    allowed = {"default_provider", "providers", "scoped_models"}
    _reject_unknown_fields(data, allowed, "Provider settings")
    default_provider = _required_string(data.get("default_provider"), "default_provider")
    providers_raw = data.get("providers")
    if not isinstance(providers_raw, list) or not providers_raw:
        raise ProviderConfigError("Provider settings must include at least one provider")
    providers = tuple(_provider_from_json(item) for item in providers_raw)
    scoped = _scoped_models_from_json(data.get("scoped_models", []))
    return ProviderSettings(
        default_provider=default_provider,
        providers=providers,
        scoped_models=scoped,
    )


def resolve_provider_selection(
    settings: ProviderSettings,
    *,
    provider_name: str | None = None,
    model: str | None = None,
) -> ProviderSelection:
    provider = settings.get_provider(provider_name)
    selected_model = model or provider.default_model
    if selected_model not in provider.models:
        raise ProviderConfigError(f"Model is not configured: {provider.name}:{selected_model}")
    return ProviderSelection(provider=provider, model=selected_model)


def set_default_provider_model(
    settings: ProviderSettings,
    *,
    provider_name: str,
    model: str,
) -> ProviderSettings:
    provider = settings.get_provider(provider_name)
    if model not in provider.models:
        raise ProviderConfigError(f"Model is not configured: {provider_name}:{model}")
    updated_provider = replace(provider, default_model=model)
    return ProviderSettings(
        default_provider=provider_name,
        providers=tuple(
            updated_provider if item.name == provider_name else item for item in settings.providers
        ),
        scoped_models=settings.scoped_models,
    )


def toggle_scoped_model(
    settings: ProviderSettings,
    *,
    provider_name: str,
    model: str,
) -> ProviderSettings:
    provider = settings.get_provider(provider_name)
    if model not in provider.models:
        raise ProviderConfigError(f"Model is not configured: {provider_name}:{model}")
    target = ScopedModelConfig(provider=provider_name, model=model)
    scoped = tuple(item for item in settings.scoped_models if item != target)
    if target not in settings.scoped_models:
        scoped = (*scoped, target)
    return replace(settings, scoped_models=scoped)


def provider_thinking_levels(
    provider: OpenAICompatibleProviderConfig,
    *,
    model: str | None = None,
) -> tuple[ThinkingLevel, ...]:
    if provider.thinking_levels is None:
        return ()
    selected_model = model or provider.default_model
    if provider.thinking_models and selected_model not in provider.thinking_models:
        return ()
    return provider.thinking_levels


def provider_default_thinking_level(
    provider: OpenAICompatibleProviderConfig,
    *,
    model: str | None = None,
) -> ThinkingLevel | None:
    levels = provider_thinking_levels(provider, model=model)
    if not levels:
        return None
    if provider.thinking_default in levels:
        return provider.thinking_default
    if DEFAULT_THINKING_LEVEL in levels:
        return DEFAULT_THINKING_LEVEL
    return levels[0]


def provider_has_usable_credentials(
    provider: OpenAICompatibleProviderConfig,
    *,
    credential_reader: FileCredentialStore | None = None,
    environment: Mapping[str, str] | None = None,
) -> bool:
    values = environ if environment is None else environment
    if values.get(provider.api_key_env):
        return True
    return bool(
        provider.credential_name
        and credential_reader is not None
        and credential_reader.get(provider.credential_name)
    )


def openai_compatible_config_from_provider(
    provider: OpenAICompatibleProviderConfig,
    *,
    credential_reader: FileCredentialStore,
    model: str,
    thinking_level: ThinkingLevel,
    environment: Mapping[str, str] | None = None,
) -> OpenAICompatibleConfig:
    values = environ if environment is None else environment
    api_key = (
        credential_reader.get(provider.credential_name)
        if provider.credential_name is not None
        else None
    ) or values.get(provider.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing credentials for provider {provider.name}. Run /login {provider.name}."
        )
    levels = provider_thinking_levels(provider, model=model)
    if levels and thinking_level not in levels:
        raise ProviderConfigError(
            f"Thinking mode {thinking_level} is not available for {provider.name}:{model}. "
            f"Available modes: {', '.join(levels)}"
        )
    effort = reasoning_effort_for_level(thinking_level) if levels else None
    base_url = provider.base_url
    if provider.name == "deepseek":
        base_url = values.get("DEEPSEEK_BASE_URL", base_url)
    return OpenAICompatibleConfig(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        headers=provider.headers,
        timeout_seconds=provider.timeout_seconds,
        max_retries=provider.max_retries,
        max_retry_delay_seconds=provider.max_retry_delay_seconds,
        max_tokens=provider.max_tokens,
        thinking_enabled=bool(levels and thinking_level != "off"),
        reasoning_effort=effort,
        reasoning_effort_parameter=provider.thinking_parameter or "reasoning_effort",
    )


def credential_store_for_paths(paths: AxisPaths | None) -> FileCredentialStore:
    return FileCredentialStore(credentials_path(paths))


def _provider_from_json(value: object) -> OpenAICompatibleProviderConfig:
    if not isinstance(value, dict):
        raise ProviderConfigError("Provider entries must be JSON objects")
    allowed = {
        "type",
        "name",
        "base_url",
        "api_key_env",
        "credential_name",
        "models",
        "default_model",
        "headers",
        "timeout_seconds",
        "max_retries",
        "max_retry_delay_seconds",
        "max_tokens",
        "context_windows",
        "thinking_levels",
        "thinking_models",
        "thinking_default",
        "thinking_parameter",
    }
    _reject_unknown_fields(value, allowed, "Provider entry")
    if value.get("type") != "openai-compatible":
        raise ProviderConfigError("Axis only supports provider type: openai-compatible")
    models = _string_tuple(value.get("models"), "models")
    thinking_raw = value.get("thinking_levels")
    thinking_levels = (
        None
        if thinking_raw is None
        else normalize_thinking_levels(_string_tuple(thinking_raw, "thinking_levels"))
    )
    thinking_default_raw = value.get("thinking_default")
    thinking_default = (
        normalize_thinking_level(thinking_default_raw)
        if isinstance(thinking_default_raw, str)
        else None
    )
    parameter = value.get("thinking_parameter")
    if parameter not in {None, "reasoning_effort", "reasoning.effort"}:
        raise ProviderConfigError("Unknown thinking_parameter")
    credential_name = value.get("credential_name")
    if credential_name is not None and not isinstance(credential_name, str):
        raise ProviderConfigError("Provider credential_name must be a string or null")
    return OpenAICompatibleProviderConfig(
        name=_required_string(value.get("name"), "name"),
        base_url=_required_string(value.get("base_url"), "base_url"),
        api_key_env=_required_string(value.get("api_key_env"), "api_key_env"),
        credential_name=credential_name,
        models=models,
        default_model=_required_string(value.get("default_model"), "default_model"),
        headers=_string_dict(value.get("headers", {}), "headers"),
        timeout_seconds=_number(value.get("timeout_seconds", 60.0), "timeout_seconds"),
        max_retries=_integer(value.get("max_retries", 2), "max_retries"),
        max_retry_delay_seconds=_number(
            value.get("max_retry_delay_seconds", 1.0),
            "max_retry_delay_seconds",
        ),
        max_tokens=_optional_integer(value.get("max_tokens"), "max_tokens"),
        context_windows=_integer_dict(value.get("context_windows", {}), "context_windows"),
        thinking_levels=thinking_levels,
        thinking_models=_string_tuple(
            value.get("thinking_models", []),
            "thinking_models",
            allow_empty=True,
        ),
        thinking_default=thinking_default,
        thinking_parameter=parameter,
    )


def _scoped_models_from_json(value: object) -> tuple[ScopedModelConfig, ...]:
    if not isinstance(value, list):
        raise ProviderConfigError("Provider scoped_models must be a list")
    result: list[ScopedModelConfig] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"provider", "model"}:
            raise ProviderConfigError("Scoped model entries require provider and model")
        choice = ScopedModelConfig(
            provider=_required_string(item.get("provider"), "scoped_models.provider"),
            model=_required_string(item.get("model"), "scoped_models.model"),
        )
        if choice not in result:
            result.append(choice)
    return tuple(result)


def _reject_unknown_fields(
    value: Mapping[str, object],
    allowed: set[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ProviderConfigError(f"{label} has unknown fields: {', '.join(sorted(unknown))}")


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderConfigError(f"Provider field must be a non-empty string: {field_name}")
    return value.strip()


def _string_tuple(value: object, field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ProviderConfigError(f"Provider field must be a list: {field_name}")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ProviderConfigError(f"Provider list values must be strings: {field_name}")
    return tuple(item.strip() for item in value if isinstance(item, str))


def _string_dict(value: object, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
    ):
        raise ProviderConfigError(f"Provider field must be a string map: {field_name}")
    return dict(value)


def _integer_dict(value: object, field_name: str) -> dict[str, int]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, int) or isinstance(item, bool)
        for key, item in value.items()
    ):
        raise ProviderConfigError(f"Provider field must be an integer map: {field_name}")
    return dict(value)


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderConfigError(f"Provider field must be a number: {field_name}")
    return float(value)


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProviderConfigError(f"Provider field must be an integer: {field_name}")
    return value


def _optional_integer(value: object, field_name: str) -> int | None:
    return None if value is None else _integer(value, field_name)


def _atomic_write(path: Path, content: str) -> None:
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)
            temp_file.flush()
        temp_path.replace(path)
    except OSError as exc:
        if temp_path is not None:
            with suppress(OSError):
                temp_path.unlink()
        raise ProviderConfigError(f"Could not save provider settings to {path}: {exc}") from exc
