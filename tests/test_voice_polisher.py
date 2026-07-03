import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from axis_agent import AssistantMessage
from axis_ai import (
    FakeProvider,
    ProviderErrorEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
)
from axis_coding.credentials import FileCredentialStore
from axis_coding.provider_config import ProviderSettings
from axis_coding.voice import (
    DeepSeekVoicePolisher,
    VoiceContextSnapshot,
    VoicePolishError,
    create_deepseek_voice_polisher,
)


def test_voice_polisher_uses_ephemeral_no_tools_request() -> None:
    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="deepseek-v4-pro"),
                ProviderResponseEndEvent(
                    message=AssistantMessage(content="请修改 ParserService。")
                ),
            ]
        ]
    )
    polisher = DeepSeekVoicePolisher(provider)
    result = asyncio.run(
        polisher.polish(
            "呃请修改 parser service",
            VoiceContextSnapshot(recent_terms=("ParserService",)),
        )
    )
    assert result.text == "请修改 ParserService。"
    assert result.breakdown.kind == "Voice polish"
    model, system, messages, tools = provider.calls[0]
    assert model == "deepseek-v4-pro"
    assert "Never answer" in system
    assert tools == []
    assert len(messages) == 1


def test_voice_polisher_turns_provider_errors_into_fallback_signal() -> None:
    provider = FakeProvider([[ProviderErrorEvent(message="offline")]])
    with pytest.raises(VoicePolishError, match="offline"):
        asyncio.run(DeepSeekVoicePolisher(provider).polish("hello", VoiceContextSnapshot()))


def test_real_voice_polisher_factory_forces_no_thinking(tmp_path: Path) -> None:
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set("deepseek", "test-key")
    polisher = create_deepseek_voice_polisher(ProviderSettings(), store)

    config = cast(Any, polisher.provider)._config
    assert config.thinking_enabled is False
    assert config.reasoning_effort is None
