"""Tests for Axis's OpenAI-compatible Chat Completions adapter."""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping

import httpx
import pytest

from axis_agent import (
    AgentEvent,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from axis_agent.loop import run_agent_loop
from axis_agent.types import JSONValue
from axis_ai import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_DEEPSEEK_REASONING_EFFORT,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    ProviderErrorEvent,
    ProviderEvent,
    ProviderResponseEndEvent,
    ProviderRetryEvent,
    ProviderThinkingDeltaEvent,
    ProviderToolCallEvent,
    deepseek_model_from_env,
    deepseek_v4_config_from_env,
    openai_compatible_config_from_env,
)
from axis_ai.retry import retry_delay_seconds


async def collect_events(stream: AsyncIterator[ProviderEvent]) -> list[ProviderEvent]:
    return [event async for event in stream]


async def collect_agent_events(stream: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [event async for event in stream]


def test_openai_compatible_config_loads_and_normalizes_environment() -> None:
    config = openai_compatible_config_from_env(
        environment={
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://example.test/v1/",
            "OPENAI_TIMEOUT_SECONDS": "12.5",
            "OPENAI_MAX_RETRIES": "4",
            "OPENAI_MAX_RETRY_DELAY_SECONDS": "0.75",
        }
    )

    assert config == OpenAICompatibleConfig(
        api_key="test-key",
        base_url="https://example.test/v1",
        timeout_seconds=12.5,
        max_retries=4,
        max_retry_delay_seconds=0.75,
    )


def test_openai_compatible_config_requires_api_key() -> None:
    with pytest.raises(RuntimeError, match="Missing required environment variable"):
        openai_compatible_config_from_env(environment={})


def test_deepseek_v4_config_uses_axis_defaults() -> None:
    config = deepseek_v4_config_from_env(environment={"DEEPSEEK_API_KEY": "deepseek-key"})

    assert DEFAULT_DEEPSEEK_BASE_URL == "https://api.deepseek.com"
    assert DEFAULT_DEEPSEEK_MODEL == "deepseek-v4-pro"
    assert DEFAULT_DEEPSEEK_REASONING_EFFORT == "max"
    assert config == OpenAICompatibleConfig(
        api_key="deepseek-key",
        base_url="https://api.deepseek.com",
        thinking_enabled=True,
        reasoning_effort="max",
    )
    assert deepseek_model_from_env(environment={}) == "deepseek-v4-pro"


def test_deepseek_v4_config_accepts_explicit_overrides() -> None:
    config = deepseek_v4_config_from_env(
        environment={
            "DEEPSEEK_API_KEY": "deepseek-key",
            "DEEPSEEK_BASE_URL": "https://gateway.example.test/deepseek/",
            "DEEPSEEK_TIMEOUT_SECONDS": "30",
            "DEEPSEEK_MAX_RETRIES": "5",
            "DEEPSEEK_MAX_RETRY_DELAY_SECONDS": "2.5",
            "DEEPSEEK_MAX_TOKENS": "128",
            "DEEPSEEK_REASONING_EFFORT": "high",
        }
    )

    assert config == OpenAICompatibleConfig(
        api_key="deepseek-key",
        base_url="https://gateway.example.test/deepseek",
        timeout_seconds=30,
        max_retries=5,
        max_retry_delay_seconds=2.5,
        max_tokens=128,
        thinking_enabled=True,
        reasoning_effort="high",
    )
    assert (
        deepseek_model_from_env(environment={"DEEPSEEK_MODEL": "deepseek-v4-flash"})
        == "deepseek-v4-flash"
    )


def test_deepseek_v4_config_rejects_missing_or_invalid_values() -> None:
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        deepseek_v4_config_from_env(environment={})

    with pytest.raises(RuntimeError, match="high.*max"):
        deepseek_v4_config_from_env(
            environment={
                "DEEPSEEK_API_KEY": "deepseek-key",
                "DEEPSEEK_REASONING_EFFORT": "medium",
            }
        )

    with pytest.raises(RuntimeError, match="must not be empty"):
        deepseek_model_from_env(environment={"DEEPSEEK_MODEL": "  "})

    with pytest.raises(RuntimeError, match="greater than 0"):
        deepseek_v4_config_from_env(
            environment={
                "DEEPSEEK_API_KEY": "deepseek-key",
                "DEEPSEEK_MAX_TOKENS": "0",
            }
        )

    with pytest.raises(RuntimeError, match="must be an integer"):
        deepseek_v4_config_from_env(
            environment={
                "DEEPSEEK_API_KEY": "deepseek-key",
                "DEEPSEEK_MAX_TOKENS": "many",
            }
        )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("OPENAI_TIMEOUT_SECONDS", "0", "greater than 0"),
        ("OPENAI_TIMEOUT_SECONDS", "later", "must be a number"),
        ("OPENAI_MAX_RETRIES", "-1", "0 or greater"),
        ("OPENAI_MAX_RETRIES", "many", "must be an integer"),
        ("OPENAI_MAX_RETRY_DELAY_SECONDS", "-0.1", "0 or greater"),
    ],
)
def test_openai_compatible_config_rejects_invalid_numbers(
    name: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        openai_compatible_config_from_env(environment={"OPENAI_API_KEY": "test-key", name: value})


def test_provider_formats_full_request_and_streams_text() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async def executor(
        arguments: Mapping[str, JSONValue],
        signal: object | None = None,
    ) -> AgentToolResult:
        del arguments, signal
        return AgentToolResult(tool_call_id="", name="read", ok=True, content="")

    tool = AgentTool(
        name="read",
        description="Read a file.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        executor=executor,
    )
    tool_call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    messages = [
        UserMessage(content="Read the file"),
        AssistantMessage(tool_calls=[tool_call]),
        ToolResultMessage(
            tool_call_id="call-1",
            name="read",
            content="file contents",
        ),
    ]

    async def scenario() -> list[ProviderEvent]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(
                    api_key="test-key",
                    base_url="https://example.test/v1/",
                    headers={"X-Organization": "axis", "Authorization": "wrong"},
                ),
                client=client,
            )
            return await collect_events(
                provider.stream_response(
                    model="test-model",
                    system="You are Axis.",
                    messages=messages,
                    tools=[tool],
                )
            )

    events = asyncio.run(scenario())

    assert [event.type for event in events] == [
        "response_start",
        "text_delta",
        "text_delta",
        "response_end",
    ]
    assert isinstance(events[-1], ProviderResponseEndEvent)
    assert events[-1].message.content == "Hello"
    assert events[-1].finish_reason == "stop"

    request = requests[0]
    assert str(request.url) == "https://example.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["x-organization"] == "axis"
    payload = json.loads(request.content)
    assert payload["model"] == "test-model"
    assert payload["stream"] is True
    assert payload["messages"] == [
        {"role": "system", "content": "You are Axis."},
        {"role": "user", "content": "Read the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"path": "README.md"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "read",
            "content": "file contents",
        },
    ]
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]


def test_provider_supports_configured_reasoning_parameter_shapes() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async def run(parameter: str) -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(
                    api_key="test-key",
                    base_url="https://example.test/v1",
                    reasoning_effort="high",
                    reasoning_effort_parameter=parameter,
                ),
                client=client,
            )
            await collect_events(
                provider.stream_response(
                    model="test-model",
                    system="",
                    messages=[],
                    tools=[],
                )
            )

    asyncio.run(run("reasoning_effort"))
    asyncio.run(run("reasoning.effort"))

    first = json.loads(requests[0].content)
    second = json.loads(requests[1].content)
    assert first["reasoning_effort"] == "high"
    assert second["reasoning"] == {"effort": "high"}


def test_deepseek_v4_config_builds_thinking_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"ok"},'
                '"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async def scenario() -> list[ProviderEvent]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatibleProvider(
                deepseek_v4_config_from_env(
                    environment={
                        "DEEPSEEK_API_KEY": "deepseek-key",
                        "DEEPSEEK_MAX_TOKENS": "128",
                    }
                ),
                client=client,
            )
            return await collect_events(
                provider.stream_response(
                    model=deepseek_model_from_env(environment={}),
                    system="You are Axis.",
                    messages=[UserMessage(content="hello")],
                    tools=[],
                )
            )

    events = asyncio.run(scenario())

    assert [event.type for event in events] == [
        "response_start",
        "text_delta",
        "response_end",
    ]
    assert str(requests[0].url) == "https://api.deepseek.com/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer deepseek-key"
    assert json.loads(requests[0].content) == {
        "model": "deepseek-v4-pro",
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are Axis."},
            {"role": "user", "content": "hello"},
        ],
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
        "max_tokens": 128,
    }


def test_provider_streams_reasoning_and_fragmented_parallel_tool_calls() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"reasoning_content":"inspect "}}]}\n\n'
                'data: {"choices":[{"delta":{"reasoning_content":"files",'
                '"tool_calls":[{"index":1,"id":"call-bash","function":'
                '{"name":"bash","arguments":"{\\"command\\":"}},'
                '{"index":0,"id":"call-read","function":'
                '{"name":"read","arguments":"{\\"path\\":"}}]}}]}\n\n'
                'data: {"choices":[{"delta":{"tool_calls":['
                '{"index":0,"function":{"arguments":"\\"README.md\\"}"}},'
                '{"index":1,"function":{"arguments":"\\"pwd\\"}"}}]},'
                '"finish_reason":"tool_calls"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async def scenario() -> list[ProviderEvent]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(
                    api_key="test-key",
                    base_url="https://example.test/v1",
                    max_retries=0,
                ),
                client=client,
            )
            return await collect_events(
                provider.stream_response(
                    model="test-model",
                    system="",
                    messages=[UserMessage(content="Inspect")],
                    tools=[],
                )
            )

    events = asyncio.run(scenario())

    assert [event.type for event in events] == [
        "response_start",
        "thinking_delta",
        "thinking_delta",
        "tool_call",
        "tool_call",
        "response_end",
    ]
    thinking = [event for event in events if isinstance(event, ProviderThinkingDeltaEvent)]
    assert [event.delta for event in thinking] == ["inspect ", "files"]
    calls = [event.tool_call for event in events if isinstance(event, ProviderToolCallEvent)]
    assert calls == [
        ToolCall(id="call-read", name="read", arguments={"path": "README.md"}),
        ToolCall(id="call-bash", name="bash", arguments={"command": "pwd"}),
    ]
    assert isinstance(events[-1], ProviderResponseEndEvent)
    assert events[-1].message == AssistantMessage(
        tool_calls=calls,
        provider_data={"reasoning_content": "inspect files"},
    )
    assert events[-1].finish_reason == "tool_calls"


def test_deepseek_tool_round_trip_returns_reasoning_content() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                text=(
                    'data: {"choices":[{"delta":{"reasoning_content":"Need read.",'
                    '"tool_calls":[{"index":0,"id":"call-1","function":'
                    '{"name":"read","arguments":"{\\"path\\":\\"README.md\\"}"}}]},'
                    '"finish_reason":"tool_calls"}]}\n\n'
                    "data: [DONE]\n\n"
                ),
                headers={"content-type": "text/event-stream"},
            )

        payload = json.loads(request.content)
        assert payload["messages"] == [
            {"role": "system", "content": "You are Axis."},
            {"role": "user", "content": "Read README.md"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path": "README.md"}',
                        },
                    }
                ],
                "reasoning_content": "Need read.",
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "read",
                "content": "README contents",
            },
        ]
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"Done"},'
                '"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async def executor(
        arguments: Mapping[str, JSONValue],
        signal: object | None = None,
    ) -> AgentToolResult:
        del signal
        assert arguments == {"path": "README.md"}
        return AgentToolResult(
            tool_call_id="call-1",
            name="read",
            ok=True,
            content="README contents",
        )

    tool = AgentTool(
        name="read",
        description="Read a file.",
        input_schema={"type": "object"},
        executor=executor,
    )
    messages = [UserMessage(content="Read README.md")]

    async def scenario() -> list[AgentEvent]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatibleProvider(
                deepseek_v4_config_from_env(
                    environment={
                        "DEEPSEEK_API_KEY": "deepseek-key",
                        "DEEPSEEK_MAX_RETRIES": "0",
                    }
                ),
                client=client,
            )
            return await collect_agent_events(
                run_agent_loop(
                    provider=provider,
                    model=deepseek_model_from_env(environment={}),
                    system="You are Axis.",
                    messages=messages,
                    tools=[tool],
                )
            )

    events = asyncio.run(scenario())

    assert len(requests) == 2
    assert [event.type for event in events].count("thinking_delta") == 1
    assert messages[1] == AssistantMessage(
        tool_calls=[ToolCall(id="call-1", name="read", arguments={"path": "README.md"})],
        provider_data={"reasoning_content": "Need read."},
    )
    assert messages[-1] == AssistantMessage(content="Done")


def test_provider_preserves_invalid_tool_arguments_for_tool_error_handling() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"tool_calls":['
                '{"index":0,"id":"call-1","function":'
                '{"name":"read","arguments":"{"}}]},'
                '"finish_reason":"tool_calls"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async def scenario() -> list[ProviderEvent]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(api_key="test-key", max_retries=0),
                client=client,
            )
            return await collect_events(
                provider.stream_response(
                    model="test-model",
                    system="",
                    messages=[],
                    tools=[],
                )
            )

    events = asyncio.run(scenario())

    assert isinstance(events[-1], ProviderResponseEndEvent)
    assert events[-1].message.tool_calls == [
        ToolCall(id="call-1", name="read", arguments={"_raw_arguments": "{"})
    ]


def test_provider_rejects_tool_call_without_provider_id() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"tool_calls":['
                '{"index":0,"function":{"name":"read","arguments":"{}"}}]},'
                '"finish_reason":"tool_calls"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async def scenario() -> list[ProviderEvent]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(api_key="test-key", max_retries=0),
                client=client,
            )
            return await collect_events(
                provider.stream_response(
                    model="test-model",
                    system="",
                    messages=[],
                    tools=[],
                )
            )

    events = asyncio.run(scenario())

    assert [event.type for event in events] == ["response_start", "error"]
    assert isinstance(events[-1], ProviderErrorEvent)
    assert events[-1].message == "Provider tool call at index 0 has no id"


def test_provider_rejects_duplicate_tool_call_ids() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"tool_calls":['
                '{"index":0,"id":"call-1","function":{"name":"read",'
                '"arguments":"{}"}},{"index":1,"id":"call-1","function":'
                '{"name":"bash","arguments":"{}"}}]},'
                '"finish_reason":"tool_calls"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async def scenario() -> list[ProviderEvent]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(api_key="test-key", max_retries=0),
                client=client,
            )
            return await collect_events(
                provider.stream_response(
                    model="test-model",
                    system="",
                    messages=[],
                    tools=[],
                )
            )

    events = asyncio.run(scenario())

    assert [event.type for event in events] == ["response_start", "error"]
    assert isinstance(events[-1], ProviderErrorEvent)
    assert events[-1].message == "Provider returned duplicate tool-call ids"


def test_provider_turns_http_invalid_json_and_network_failures_into_events() -> None:
    def status_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    def json_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="data: not-json\n\n",
            headers={"content-type": "text/event-stream"},
        )

    def network_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    async def run(handler: object) -> list[ProviderEvent]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:  # type: ignore[arg-type]
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(
                    api_key="test-key",
                    base_url="https://example.test/v1",
                    max_retries=0,
                ),
                client=client,
            )
            return await collect_events(
                provider.stream_response(
                    model="test-model",
                    system="",
                    messages=[],
                    tools=[],
                )
            )

    status_events = asyncio.run(run(status_handler))
    json_events = asyncio.run(run(json_handler))
    network_events = asyncio.run(run(network_handler))

    assert isinstance(status_events[0], ProviderErrorEvent)
    assert status_events[0].data == {
        "status_code": 400,
        "body": "bad request",
        "attempts": 1,
    }
    assert [event.type for event in json_events] == ["response_start", "error"]
    assert isinstance(network_events[0], ProviderErrorEvent)
    assert network_events[0].data == {"error_type": "ConnectError", "attempts": 1}


def test_provider_retries_transient_http_status_then_succeeds() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, text="try later")
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"ok"},'
                '"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async def scenario() -> list[ProviderEvent]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(
                    api_key="test-key",
                    max_retries=1,
                    max_retry_delay_seconds=0,
                ),
                client=client,
            )
            return await collect_events(
                provider.stream_response(
                    model="test-model",
                    system="",
                    messages=[],
                    tools=[],
                )
            )

    events = asyncio.run(scenario())

    assert len(requests) == 2
    assert [event.type for event in events] == [
        "retry",
        "response_start",
        "text_delta",
        "response_end",
    ]
    assert isinstance(events[0], ProviderRetryEvent)
    assert events[0].attempt == 2
    assert events[0].max_attempts == 2
    assert events[0].delay_seconds == 0
    assert events[0].data == {"status_code": 503, "body": "try later"}


def test_retry_backoff_is_exponential_and_capped() -> None:
    assert retry_delay_seconds(0, max_delay_seconds=1) == 0.25
    assert retry_delay_seconds(1, max_delay_seconds=1) == 0.5
    assert retry_delay_seconds(2, max_delay_seconds=1) == 1
    assert retry_delay_seconds(8, max_delay_seconds=1) == 1
    assert retry_delay_seconds(0, max_delay_seconds=0.1) == 0.1
    assert retry_delay_seconds(0, max_delay_seconds=0) == 0


def test_provider_retries_network_failure_only_before_response_starts() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ConnectError("connection failed", request=request)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async def scenario() -> list[ProviderEvent]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(
                    api_key="test-key",
                    max_retries=1,
                    max_retry_delay_seconds=0,
                ),
                client=client,
            )
            return await collect_events(
                provider.stream_response(
                    model="test-model",
                    system="",
                    messages=[],
                    tools=[],
                )
            )

    events = asyncio.run(scenario())

    assert len(requests) == 2
    assert [event.type for event in events] == [
        "retry",
        "response_start",
        "text_delta",
        "response_end",
    ]
    assert isinstance(events[0], ProviderRetryEvent)
    assert events[0].data == {
        "error": "connection failed",
        "error_type": "ConnectError",
    }


def test_provider_does_not_retry_after_successful_response_starts() -> None:
    class BrokenSseStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise httpx.ReadError("stream interrupted")

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            stream=BrokenSseStream(),
            headers={"content-type": "text/event-stream"},
        )

    async def scenario() -> list[ProviderEvent]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(api_key="test-key", max_retries=3),
                client=client,
            )
            return await collect_events(
                provider.stream_response(
                    model="test-model",
                    system="",
                    messages=[],
                    tools=[],
                )
            )

    events = asyncio.run(scenario())

    assert len(requests) == 1
    assert [event.type for event in events] == ["response_start", "text_delta", "error"]
    assert isinstance(events[-1], ProviderErrorEvent)
    assert events[-1].data == {"error_type": "ReadError", "attempts": 1}


def test_provider_does_not_retry_non_transient_http_status() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(400, text="bad request")

    async def scenario() -> list[ProviderEvent]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(api_key="test-key", max_retries=3),
                client=client,
            )
            return await collect_events(
                provider.stream_response(
                    model="test-model",
                    system="",
                    messages=[],
                    tools=[],
                )
            )

    events = asyncio.run(scenario())

    assert len(requests) == 1
    assert [event.type for event in events] == ["error"]
    assert isinstance(events[0], ProviderErrorEvent)
    assert events[0].data == {
        "status_code": 400,
        "body": "bad request",
        "attempts": 1,
    }


def test_provider_cancellation_interrupts_retry_backoff() -> None:
    class CancellationSignal:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

        def is_cancelled(self) -> bool:
            return self.cancelled

    signal = CancellationSignal()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503, text="try later")

    async def scenario() -> list[ProviderEvent]:
        events: list[ProviderEvent] = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(
                    api_key="test-key",
                    max_retries=2,
                    max_retry_delay_seconds=1,
                ),
                client=client,
            )
            async for event in provider.stream_response(
                model="test-model",
                system="",
                messages=[],
                tools=[],
                signal=signal,
            ):
                events.append(event)
                if isinstance(event, ProviderRetryEvent):
                    signal.cancel()
        return events

    events = asyncio.run(scenario())

    assert len(requests) == 1
    assert [event.type for event in events] == ["retry"]


def test_provider_observes_pre_cancel_without_sending_request() -> None:
    class CancelledSignal:
        def is_cancelled(self) -> bool:
            return True

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text="data: [DONE]\n\n")

    async def scenario() -> list[ProviderEvent]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(api_key="test-key"),
                client=client,
            )
            return await collect_events(
                provider.stream_response(
                    model="test-model",
                    system="",
                    messages=[],
                    tools=[],
                    signal=CancelledSignal(),
                )
            )

    assert asyncio.run(scenario()) == []
    assert requests == []


def test_provider_closes_only_clients_it_owns() -> None:
    async def scenario() -> None:
        external = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
        injected_provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test-key"),
            client=external,
        )
        await injected_provider.aclose()
        assert external.is_closed is False
        await external.aclose()

        owning_provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test-key", timeout_seconds=7.5)
        )
        owned = owning_provider._get_client()
        assert owned.timeout.connect == 7.5
        await owning_provider.aclose()
        assert owned.is_closed is True

    asyncio.run(scenario())
