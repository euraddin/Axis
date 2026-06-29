"""Model-provider and streaming adapter layer for Axis."""

from axis_ai.config import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_DEEPSEEK_REASONING_EFFORT,
    OpenAICompatibleConfig,
    deepseek_model_from_env,
    deepseek_v4_config_from_env,
    openai_compatible_config_from_env,
)
from axis_ai.events import (
    ProviderErrorEvent,
    ProviderEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderRetryEvent,
    ProviderTextDeltaEvent,
    ProviderThinkingDeltaEvent,
    ProviderToolCallEvent,
)
from axis_ai.fake import FakeProvider
from axis_ai.openai_compatible import OpenAICompatibleProvider
from axis_ai.provider import CancellationToken, ModelProvider

__all__ = [
    "CancellationToken",
    "DEFAULT_DEEPSEEK_BASE_URL",
    "DEFAULT_DEEPSEEK_MODEL",
    "DEFAULT_DEEPSEEK_REASONING_EFFORT",
    "FakeProvider",
    "ModelProvider",
    "OpenAICompatibleConfig",
    "OpenAICompatibleProvider",
    "ProviderErrorEvent",
    "ProviderEvent",
    "ProviderResponseEndEvent",
    "ProviderResponseStartEvent",
    "ProviderRetryEvent",
    "ProviderTextDeltaEvent",
    "ProviderThinkingDeltaEvent",
    "ProviderToolCallEvent",
    "deepseek_model_from_env",
    "deepseek_v4_config_from_env",
    "openai_compatible_config_from_env",
]
