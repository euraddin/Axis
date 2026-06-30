"""Built-in OpenAI-compatible providers presented by Axis setup flows."""

from dataclasses import dataclass
from typing import Literal

from axis_ai import DEFAULT_DEEPSEEK_BASE_URL, DEFAULT_DEEPSEEK_MODEL
from axis_coding.thinking import ThinkingLevel, ThinkingParameter

ProviderKind = Literal["openai-compatible"]


@dataclass(frozen=True, slots=True)
class ProviderCatalogEntry:
    name: str
    display_name: str
    kind: ProviderKind
    base_url: str
    api_key_env: str
    credential_name: str
    models: tuple[str, ...]
    default_model: str
    docs_url: str
    thinking_levels: tuple[ThinkingLevel, ...] | None = None
    thinking_models: tuple[str, ...] = ()
    thinking_default: ThinkingLevel | None = None
    thinking_parameter: ThinkingParameter | None = None


BUILTIN_PROVIDER_CATALOG: tuple[ProviderCatalogEntry, ...] = (
    ProviderCatalogEntry(
        name="deepseek",
        display_name="DeepSeek",
        kind="openai-compatible",
        base_url=DEFAULT_DEEPSEEK_BASE_URL,
        api_key_env="DEEPSEEK_API_KEY",
        credential_name="deepseek",
        models=(DEFAULT_DEEPSEEK_MODEL,),
        default_model=DEFAULT_DEEPSEEK_MODEL,
        docs_url="https://api-docs.deepseek.com/",
        thinking_levels=("high", "xhigh"),
        thinking_models=(DEFAULT_DEEPSEEK_MODEL,),
        thinking_default="xhigh",
        thinking_parameter="reasoning_effort",
    ),
)


def builtin_provider_entry(name: str) -> ProviderCatalogEntry | None:
    normalized = name.strip().casefold()
    return next((entry for entry in BUILTIN_PROVIDER_CATALOG if entry.name == normalized), None)
