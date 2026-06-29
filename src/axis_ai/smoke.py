"""Opt-in live DeepSeek smoke test for the Axis provider adapter."""

import asyncio
from dataclasses import dataclass, replace

from axis_agent.messages import UserMessage
from axis_ai.config import deepseek_model_from_env, deepseek_v4_config_from_env
from axis_ai.events import (
    ProviderErrorEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
    ProviderThinkingDeltaEvent,
)
from axis_ai.openai_compatible import OpenAICompatibleProvider
from axis_ai.provider import ModelProvider

SMOKE_MAX_TOKENS = 128


@dataclass(frozen=True, slots=True)
class DeepSeekSmokeResult:
    """Non-secret facts proving that one live stream completed correctly."""

    model: str
    thinking_characters: int
    text: str
    finish_reason: str | None


async def run_deepseek_smoke(
    provider: ModelProvider,
    *,
    model: str,
) -> DeepSeekSmokeResult:
    """Run and validate one tiny provider stream without printing reasoning."""
    response_starts = 0
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    complete: ProviderResponseEndEvent | None = None

    async for event in provider.stream_response(
        model=model,
        system="You are Axis. Follow the user's output format exactly.",
        messages=[UserMessage(content="Reply with exactly AXIS_OK and nothing else.")],
        tools=[],
    ):
        if isinstance(event, ProviderResponseStartEvent):
            response_starts += 1
        elif isinstance(event, ProviderThinkingDeltaEvent):
            thinking_parts.append(event.delta)
        elif isinstance(event, ProviderTextDeltaEvent):
            text_parts.append(event.delta)
        elif isinstance(event, ProviderResponseEndEvent):
            complete = event
        elif isinstance(event, ProviderErrorEvent):
            raise RuntimeError(f"DeepSeek smoke request failed: {event.message}; {event.data}")

    if response_starts != 1:
        raise RuntimeError(f"Expected one response start, received {response_starts}")
    if complete is None:
        raise RuntimeError("DeepSeek stream ended without a complete assistant message")

    streamed_text = "".join(text_parts)
    if streamed_text != complete.message.content:
        raise RuntimeError("Streamed text did not match the complete assistant message")

    reasoning_content = complete.message.provider_data.get("reasoning_content", "")
    streamed_thinking = "".join(thinking_parts)
    if reasoning_content != streamed_thinking:
        raise RuntimeError("Streamed thinking did not match stored reasoning_content")

    return DeepSeekSmokeResult(
        model=model,
        thinking_characters=len(streamed_thinking),
        text=streamed_text,
        finish_reason=complete.finish_reason,
    )


async def _run_from_environment() -> DeepSeekSmokeResult:
    config = deepseek_v4_config_from_env()
    if config.max_tokens is None:
        config = replace(config, max_tokens=SMOKE_MAX_TOKENS)
    provider = OpenAICompatibleProvider(config)
    try:
        return await run_deepseek_smoke(
            provider,
            model=deepseek_model_from_env(),
        )
    finally:
        await provider.aclose()


def main() -> None:
    """Run the opt-in smoke test using only environment configuration."""
    try:
        result = asyncio.run(_run_from_environment())
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    print(f"model={result.model}")
    print(f"thinking_characters={result.thinking_characters}")
    print(f"text={result.text!r}")
    print(f"finish_reason={result.finish_reason}")


if __name__ == "__main__":
    main()
