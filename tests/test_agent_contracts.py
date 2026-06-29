"""Tests for Axis's portable message and tool contracts."""

import asyncio
from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from axis_agent import (
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from axis_agent.types import JSONValue


def test_user_message_serializes_with_stable_role() -> None:
    message = UserMessage(content="hello")

    assert message.model_dump() == {"role": "user", "content": "hello"}


def test_assistant_message_contains_structured_tool_calls() -> None:
    tool_call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    message = AssistantMessage(
        content="I will read it.",
        tool_calls=[tool_call],
        provider_data={"reasoning_content": "I should inspect the file."},
    )

    assert message.model_dump() == {
        "role": "assistant",
        "content": "I will read it.",
        "tool_calls": [
            {
                "id": "call-1",
                "name": "read",
                "arguments": {"path": "README.md"},
            }
        ],
        "provider_data": {"reasoning_content": "I should inspect the file."},
    }


def test_tool_result_message_preserves_structured_metadata() -> None:
    message = ToolResultMessage(
        tool_call_id="call-1",
        name="read",
        content="file contents",
        data={"path": "README.md"},
        details={"bytes": 13},
    )

    assert message.role == "tool"
    assert message.ok is True
    assert message.data == {"path": "README.md"}
    assert message.details == {"bytes": 13}


def test_contract_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        UserMessage(content="hello", unexpected=True)  # type: ignore[call-arg]


def test_agent_tool_delegates_to_async_executor() -> None:
    class FakeCancellationToken:
        def is_cancelled(self) -> bool:
            return False

    observed: list[tuple[Mapping[str, JSONValue], object | None]] = []

    async def executor(
        arguments: Mapping[str, JSONValue],
        signal: object | None = None,
    ) -> AgentToolResult:
        observed.append((arguments, signal))
        return AgentToolResult(
            tool_call_id="call-1",
            name="echo",
            ok=True,
            content=str(arguments["text"]),
        )

    tool = AgentTool(
        name="echo",
        description="Echo text.",
        input_schema={"type": "object"},
        executor=executor,
    )
    signal = FakeCancellationToken()

    result = asyncio.run(tool.execute({"text": "hello"}, signal=signal))

    assert result.content == "hello"
    assert observed == [({"text": "hello"}, signal)]
