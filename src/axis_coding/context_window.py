"""Deterministic, provider-neutral context usage estimates for the TUI."""

from json import dumps

from axis_agent import AgentMessage, AssistantMessage, ToolResultMessage

DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000


def estimate_text_tokens(text: str) -> int:
    """Return a deliberately rough UTF-8-aware token estimate."""
    if not text:
        return 0
    byte_count = len(text.encode("utf-8"))
    return max(1, (byte_count + 3) // 4)


def estimate_message_tokens(message: AgentMessage) -> int:
    """Estimate one serialized transcript message including protocol overhead."""
    estimate = 4 + estimate_text_tokens(message.content)
    if isinstance(message, AssistantMessage):
        for call in message.tool_calls:
            estimate += 6 + estimate_text_tokens(call.name)
            estimate += estimate_text_tokens(dumps(call.arguments, sort_keys=True))
        reasoning = message.provider_data.get("reasoning_content")
        if isinstance(reasoning, str):
            estimate += estimate_text_tokens(reasoning)
    elif isinstance(message, ToolResultMessage):
        estimate += 4 + estimate_text_tokens(message.name)
    return estimate


def estimate_context_tokens(
    *,
    system: str,
    messages: tuple[AgentMessage, ...],
) -> int:
    """Estimate the next request's system and transcript token footprint."""
    return estimate_text_tokens(system) + sum(
        estimate_message_tokens(message) for message in messages
    )
