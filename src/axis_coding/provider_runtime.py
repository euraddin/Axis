"""Runtime provider construction from Axis's durable settings."""

from collections.abc import AsyncIterator
from typing import Protocol

from axis_agent import AgentMessage, AgentTool
from axis_ai import (
    CancellationToken,
    ModelProvider,
    OpenAICompatibleProvider,
    ProviderErrorEvent,
    ProviderEvent,
)
from axis_coding.credentials import FileCredentialStore
from axis_coding.provider_config import (
    OpenAICompatibleProviderConfig,
    openai_compatible_config_from_provider,
)
from axis_coding.thinking import ThinkingLevel


class ClosableModelProvider(ModelProvider, Protocol):
    async def aclose(self) -> None: ...


class LoginRequiredProvider:
    """Open the TUI safely before a provider credential exists."""

    def __init__(self, message: str) -> None:
        self.message = message

    async def aclose(self) -> None:
        return None

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        del model, system, messages, tools, signal

        async def iterator() -> AsyncIterator[ProviderEvent]:
            yield ProviderErrorEvent(message=self.message)

        return iterator()


def create_model_provider(
    provider: OpenAICompatibleProviderConfig,
    *,
    credential_store: FileCredentialStore,
    model: str,
    thinking_level: ThinkingLevel,
) -> ClosableModelProvider:
    return OpenAICompatibleProvider(
        openai_compatible_config_from_provider(
            provider,
            credential_reader=credential_store,
            model=model,
            thinking_level=thinking_level,
        )
    )
