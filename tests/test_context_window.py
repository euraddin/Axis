"""Tests for deterministic context usage estimates shown by the TUI."""

import asyncio
from pathlib import Path

from axis_agent import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from axis_agent.session import JsonlSessionStorage
from axis_ai import FakeProvider
from axis_coding import (
    CodingSession,
    CodingSessionConfig,
    ContextUsageEstimate,
    OpenAICompatibleProviderConfig,
    ProviderSettings,
    create_coding_tools,
    estimate_context_tokens,
    estimate_context_usage,
    estimate_message_tokens,
    estimate_text_tokens,
    estimate_tool_tokens,
)


def test_context_estimates_are_deterministic_and_include_protocol_data() -> None:
    plain = UserMessage(content="abcd")
    assistant = AssistantMessage(
        content="working",
        tool_calls=[ToolCall(id="call-1", name="read", arguments={"path": "README.md"})],
        provider_data={"reasoning_content": "inspect first"},
    )
    result = ToolResultMessage(
        tool_call_id="call-1",
        name="read",
        content="contents",
    )

    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abcd") == 1
    assert estimate_message_tokens(assistant) > estimate_message_tokens(plain)
    assert estimate_context_tokens(
        system="system",
        messages=(plain, assistant, result),
    ) == estimate_context_tokens(
        system="system",
        messages=(plain, assistant, result),
    )


def test_unicode_estimate_uses_encoded_size_instead_of_character_count() -> None:
    assert estimate_text_tokens("你好世界") > estimate_text_tokens("abcd")


def test_context_usage_reports_system_message_and_tool_proportions(tmp_path: Path) -> None:
    messages = (UserMessage(content="inspect this project"),)
    tools = tuple(create_coding_tools(cwd=tmp_path))

    usage = estimate_context_usage(
        system="You are Axis.",
        messages=messages,
        tools=tools,
    )

    assert isinstance(usage, ContextUsageEstimate)
    assert usage.system_tokens == estimate_text_tokens("You are Axis.")
    assert usage.message_tokens == estimate_message_tokens(messages[0])
    assert usage.tool_tokens == sum(estimate_tool_tokens(tool) for tool in tools)
    assert usage.total_tokens == (usage.system_tokens + usage.message_tokens + usage.tool_tokens)
    assert usage.message_count == 1
    assert usage.tool_count == 4
    assert (
        estimate_context_tokens(
            system="You are Axis.",
            messages=messages,
            tools=tools,
        )
        == usage.total_tokens
    )


def test_coding_session_exposes_provider_window_and_display_threshold(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        provider_config = OpenAICompatibleProviderConfig(
            name="local",
            base_url="https://local.invalid/v1",
            api_key_env="AXIS_LOCAL_KEY",
            credential_name="local",
            models=("fake",),
            default_model="fake",
            context_windows={"fake": 200_000},
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "context-session.jsonl"),
                cwd=tmp_path,
                tools=[],
                provider_name="local",
                provider_settings=ProviderSettings(
                    default_provider="local",
                    providers=(provider_config,),
                ),
                auto_compact_token_threshold=64_000,
            )
        )

        assert session.context_window_tokens == 200_000
        assert session.auto_compact_token_threshold == 64_000
        assert session.context_usage.total_tokens == session.context_token_estimate
        assert session.context_usage.system_tokens > 0
        assert session.context_usage.tool_tokens == 0

    asyncio.run(scenario())
