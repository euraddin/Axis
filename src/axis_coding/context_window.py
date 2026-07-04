"""Deterministic, provider-neutral context usage estimates for the TUI."""

from dataclasses import dataclass
from json import dumps

from axis_agent import AgentMessage, AgentTool, AssistantMessage, ToolResultMessage, UserMessage

DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000
DEFAULT_AUTO_COMPACT_RATIO = 0.8
DEFAULT_COMPACT_RETAIN_TOKENS = 20_000
TOOL_DEFINITION_OVERHEAD_TOKENS = 16


@dataclass(frozen=True, slots=True)
class ContextUsageEstimate:
    """Estimated input-token composition of one Provider request."""

    total_tokens: int
    system_tokens: int
    message_tokens: int
    tool_tokens: int
    message_count: int
    tool_count: int


@dataclass(frozen=True, slots=True)
class RequestContextPart:
    """One named component of an estimated provider request."""

    name: str
    estimated_tokens: int


@dataclass(frozen=True, slots=True)
class RequestContextBreakdown:
    """Provider-neutral named token estimate for one outbound request."""

    kind: str
    parts: tuple[RequestContextPart, ...]

    @property
    def total_tokens(self) -> int:
        return sum(part.estimated_tokens for part in self.parts)


@dataclass(frozen=True, slots=True)
class ContextRetentionPlan:
    """A whole-user-turn split between summarized and verbatim context."""

    summarized_entry_ids: tuple[str, ...]
    summarized_messages: tuple[AgentMessage, ...]
    retained_entry_ids: tuple[str, ...]
    retained_messages: tuple[AgentMessage, ...]
    retained_tokens: int


def context_usage_breakdown(usage: ContextUsageEstimate) -> RequestContextBreakdown:
    """Adapt the legacy agent estimate to the generic request representation."""
    return RequestContextBreakdown(
        kind="Agent",
        parts=(
            RequestContextPart("system", usage.system_tokens),
            RequestContextPart("messages", usage.message_tokens),
            RequestContextPart("tools", usage.tool_tokens),
        ),
    )


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


def estimate_tool_tokens(tool: AgentTool) -> int:
    """Estimate one serialized Provider tool definition."""
    return (
        TOOL_DEFINITION_OVERHEAD_TOKENS
        + estimate_text_tokens(tool.name)
        + estimate_text_tokens(tool.description)
        + estimate_text_tokens(dumps(dict(tool.input_schema), sort_keys=True))
    )


def estimate_context_usage(
    *,
    system: str,
    messages: tuple[AgentMessage, ...],
    tools: tuple[AgentTool, ...] = (),
) -> ContextUsageEstimate:
    """Estimate the system/messages/tools composition of one request."""
    system_tokens = estimate_text_tokens(system)
    message_tokens = sum(estimate_message_tokens(message) for message in messages)
    tool_tokens = sum(estimate_tool_tokens(tool) for tool in tools)
    return ContextUsageEstimate(
        total_tokens=system_tokens + message_tokens + tool_tokens,
        system_tokens=system_tokens,
        message_tokens=message_tokens,
        tool_tokens=tool_tokens,
        message_count=len(messages),
        tool_count=len(tools),
    )


def estimate_context_tokens(
    *,
    system: str,
    messages: tuple[AgentMessage, ...],
    tools: tuple[AgentTool, ...] = (),
) -> int:
    """Estimate the next request's system and transcript token footprint."""
    return estimate_context_usage(
        system=system,
        messages=messages,
        tools=tools,
    ).total_tokens


def plan_context_retention(
    *,
    entry_ids: tuple[str, ...],
    messages: tuple[AgentMessage, ...],
    retain_tokens: int = DEFAULT_COMPACT_RETAIN_TOKENS,
) -> ContextRetentionPlan:
    """Retain newest complete user turns until their estimate reaches the target."""
    if retain_tokens <= 0:
        raise ValueError("retain_tokens must be greater than 0")
    if len(entry_ids) != len(messages):
        raise ValueError("entry_ids and messages must have the same length")

    groups: list[list[tuple[str, AgentMessage]]] = []
    for entry_id, message in zip(entry_ids, messages, strict=True):
        if isinstance(message, UserMessage) or not groups:
            groups.append([])
        groups[-1].append((entry_id, message))

    retained_group_index = len(groups)
    retained_tokens = 0
    for index in range(len(groups) - 1, -1, -1):
        retained_group_index = index
        retained_tokens += sum(estimate_message_tokens(message) for _, message in groups[index])
        if retained_tokens >= retain_tokens:
            break

    summarized_rows = [row for group in groups[:retained_group_index] for row in group]
    retained_rows = [row for group in groups[retained_group_index:] for row in group]
    return ContextRetentionPlan(
        summarized_entry_ids=tuple(entry_id for entry_id, _ in summarized_rows),
        summarized_messages=tuple(message for _, message in summarized_rows),
        retained_entry_ids=tuple(entry_id for entry_id, _ in retained_rows),
        retained_messages=tuple(message for _, message in retained_rows),
        retained_tokens=retained_tokens,
    )
