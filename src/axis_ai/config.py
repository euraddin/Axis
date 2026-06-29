"""Configuration for Axis model-provider adapters."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS = 60.0
DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES = 2
DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS = 1.0
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_REASONING_EFFORT = "max"


@dataclass(frozen=True, slots=True)
class OpenAICompatibleConfig:
    """Configuration for an OpenAI-compatible Chat Completions endpoint."""

    api_key: str
    base_url: str = DEFAULT_OPENAI_COMPATIBLE_BASE_URL
    headers: Mapping[str, str] | None = None
    timeout_seconds: float = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES
    max_retry_delay_seconds: float = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS
    max_tokens: int | None = None
    thinking_enabled: bool = False
    reasoning_effort: str | None = None
    reasoning_effort_parameter: str = "reasoning_effort"


def openai_compatible_config_from_env(
    *,
    environment: Mapping[str, str] | None = None,
    api_key_var: str = "OPENAI_API_KEY",
    base_url_var: str = "OPENAI_BASE_URL",
    timeout_seconds_var: str = "OPENAI_TIMEOUT_SECONDS",
    max_retries_var: str = "OPENAI_MAX_RETRIES",
    max_retry_delay_seconds_var: str = "OPENAI_MAX_RETRY_DELAY_SECONDS",
) -> OpenAICompatibleConfig:
    """Load OpenAI-compatible settings from environment variables."""
    values = os.environ if environment is None else environment
    api_key = values.get(api_key_var)
    if not api_key:
        raise RuntimeError(f"Missing required environment variable: {api_key_var}")

    timeout_seconds = _positive_float(
        values.get(timeout_seconds_var),
        timeout_seconds_var,
        DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
    )
    max_retries = _non_negative_int(
        values.get(max_retries_var),
        max_retries_var,
        DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES,
    )
    max_retry_delay_seconds = _non_negative_float(
        values.get(max_retry_delay_seconds_var),
        max_retry_delay_seconds_var,
        DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS,
    )
    return OpenAICompatibleConfig(
        api_key=api_key,
        base_url=values.get(base_url_var, DEFAULT_OPENAI_COMPATIBLE_BASE_URL).rstrip("/"),
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_retry_delay_seconds=max_retry_delay_seconds,
    )


def deepseek_v4_config_from_env(
    *,
    environment: Mapping[str, str] | None = None,
    api_key_var: str = "DEEPSEEK_API_KEY",
    base_url_var: str = "DEEPSEEK_BASE_URL",
    timeout_seconds_var: str = "DEEPSEEK_TIMEOUT_SECONDS",
    max_retries_var: str = "DEEPSEEK_MAX_RETRIES",
    max_retry_delay_seconds_var: str = "DEEPSEEK_MAX_RETRY_DELAY_SECONDS",
    max_tokens_var: str = "DEEPSEEK_MAX_TOKENS",
    reasoning_effort_var: str = "DEEPSEEK_REASONING_EFFORT",
) -> OpenAICompatibleConfig:
    """Load Axis's DeepSeek V4 defaults from environment variables."""
    values = os.environ if environment is None else environment
    api_key = values.get(api_key_var)
    if not api_key:
        raise RuntimeError(f"Missing required environment variable: {api_key_var}")

    reasoning_effort = values.get(
        reasoning_effort_var,
        DEFAULT_DEEPSEEK_REASONING_EFFORT,
    )
    if reasoning_effort not in {"high", "max"}:
        raise RuntimeError(f"Environment variable must be 'high' or 'max': {reasoning_effort_var}")

    return OpenAICompatibleConfig(
        api_key=api_key,
        base_url=values.get(base_url_var, DEFAULT_DEEPSEEK_BASE_URL).rstrip("/"),
        timeout_seconds=_positive_float(
            values.get(timeout_seconds_var),
            timeout_seconds_var,
            DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
        ),
        max_retries=_non_negative_int(
            values.get(max_retries_var),
            max_retries_var,
            DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES,
        ),
        max_retry_delay_seconds=_non_negative_float(
            values.get(max_retry_delay_seconds_var),
            max_retry_delay_seconds_var,
            DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS,
        ),
        max_tokens=_optional_positive_int(
            values.get(max_tokens_var),
            max_tokens_var,
        ),
        thinking_enabled=True,
        reasoning_effort=reasoning_effort,
    )


def deepseek_model_from_env(
    *,
    environment: Mapping[str, str] | None = None,
    model_var: str = "DEEPSEEK_MODEL",
) -> str:
    """Return the configured DeepSeek model or Axis's V4-Pro default."""
    values = os.environ if environment is None else environment
    model = values.get(model_var, DEFAULT_DEEPSEEK_MODEL).strip()
    if not model:
        raise RuntimeError(f"Environment variable must not be empty: {model_var}")
    return model


def _positive_float(raw: str | None, name: str, default: float) -> float:
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable must be a number: {name}") from exc
    if value <= 0:
        raise RuntimeError(f"Environment variable must be greater than 0: {name}")
    return value


def _non_negative_int(raw: str | None, name: str, default: int) -> int:
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable must be an integer: {name}") from exc
    if value < 0:
        raise RuntimeError(f"Environment variable must be 0 or greater: {name}")
    return value


def _non_negative_float(raw: str | None, name: str, default: float) -> float:
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable must be a number: {name}") from exc
    if value < 0:
        raise RuntimeError(f"Environment variable must be 0 or greater: {name}")
    return value


def _optional_positive_int(raw: str | None, name: str) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable must be an integer: {name}") from exc
    if value <= 0:
        raise RuntimeError(f"Environment variable must be greater than 0: {name}")
    return value
