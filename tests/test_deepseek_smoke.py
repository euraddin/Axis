"""Tests for the opt-in live DeepSeek smoke-test driver."""

import asyncio

import pytest

from axis_agent import AssistantMessage
from axis_ai import (
    FakeProvider,
    ProviderErrorEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
    ProviderThinkingDeltaEvent,
)
from axis_ai.smoke import DeepSeekSmokeResult, main, run_deepseek_smoke


def test_smoke_driver_validates_one_complete_stream() -> None:
    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="deepseek-v4-pro"),
                ProviderThinkingDeltaEvent(delta="brief thought"),
                ProviderTextDeltaEvent(delta="AXIS_"),
                ProviderTextDeltaEvent(delta="OK"),
                ProviderResponseEndEvent(
                    message=AssistantMessage(
                        content="AXIS_OK",
                        provider_data={"reasoning_content": "brief thought"},
                    ),
                    finish_reason="stop",
                ),
            ]
        ]
    )

    result = asyncio.run(run_deepseek_smoke(provider, model="deepseek-v4-pro"))

    assert result == DeepSeekSmokeResult(
        model="deepseek-v4-pro",
        thinking_characters=13,
        text="AXIS_OK",
        finish_reason="stop",
    )


def test_smoke_driver_fails_on_provider_error() -> None:
    provider = FakeProvider(
        [[ProviderErrorEvent(message="unauthorized", data={"status_code": 401})]]
    )

    with pytest.raises(RuntimeError, match="unauthorized"):
        asyncio.run(run_deepseek_smoke(provider, model="deepseek-v4-pro"))


def test_smoke_entrypoint_requires_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="DEEPSEEK_API_KEY"):
        main()
