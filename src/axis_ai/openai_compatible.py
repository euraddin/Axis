"""OpenAI-compatible Chat Completions provider adapter."""

from collections.abc import AsyncIterator, Mapping
from json import JSONDecodeError, dumps, loads
from typing import Any

import httpx

from axis_agent.messages import AgentMessage, AssistantMessage, ToolResultMessage, UserMessage
from axis_agent.tools import AgentTool, ToolCall
from axis_agent.types import JSONValue
from axis_ai.config import OpenAICompatibleConfig
from axis_ai.events import (
    ProviderErrorEvent,
    ProviderEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
    ProviderThinkingDeltaEvent,
    ProviderToolCallEvent,
)
from axis_ai.provider import CancellationToken
from axis_ai.retry import make_retry_event, retry_delay_seconds, wait_for_retry


class OpenAICompatibleProvider:
    """Translate an OpenAI-compatible ``/chat/completions`` SSE stream."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close only an HTTP client created by this provider."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        """Stream one model response as provider-neutral events."""

        async def iterator() -> AsyncIterator[ProviderEvent]:
            if signal is not None and signal.is_cancelled():
                return

            client = self._get_client()
            payload = _build_chat_payload(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=self._config.max_tokens,
                thinking_enabled=self._config.thinking_enabled,
                reasoning_effort=self._config.reasoning_effort,
                reasoning_effort_parameter=self._config.reasoning_effort_parameter,
            )
            headers = {
                **dict(self._config.headers or {}),
                "Authorization": f"Bearer {self._config.api_key}",
            }
            url = f"{self._config.base_url.rstrip('/')}/chat/completions"

            attempt = 0
            while True:
                response_started = False
                try:
                    async with client.stream(
                        "POST", url, json=payload, headers=headers
                    ) as response:
                        if response.status_code >= 400:
                            body = (await response.aread()).decode(errors="replace")
                            if self._should_retry(attempt, status_code=response.status_code):
                                delay = retry_delay_seconds(
                                    attempt,
                                    max_delay_seconds=self._config.max_retry_delay_seconds,
                                )
                                yield make_retry_event(
                                    attempt=attempt,
                                    max_retries=self._config.max_retries,
                                    delay_seconds=delay,
                                    reason=f"HTTP {response.status_code}",
                                    data={
                                        "status_code": response.status_code,
                                        "body": body,
                                    },
                                )
                                attempt += 1
                                if not await wait_for_retry(delay, signal=signal):
                                    return
                                continue
                            yield ProviderErrorEvent(
                                message=(
                                    f"Provider request failed with status {response.status_code}"
                                ),
                                data={
                                    "status_code": response.status_code,
                                    "body": body,
                                    "attempts": attempt + 1,
                                },
                            )
                            return

                        response_started = True
                        yield ProviderResponseStartEvent(model=model)
                        content_parts: list[str] = []
                        reasoning_content_parts: list[str] = []
                        tool_call_builders: dict[int, _ToolCallBuilder] = {}
                        finish_reason: str | None = None

                        async for line in response.aiter_lines():
                            if signal is not None and signal.is_cancelled():
                                return

                            event = _parse_sse_line(line)
                            if event is None:
                                continue
                            if event == "[DONE]":
                                break

                            chunk = _loads_object(event)
                            if chunk is None:
                                yield ProviderErrorEvent(
                                    message="Provider returned invalid JSON chunk"
                                )
                                return

                            choice = _first_choice(chunk)
                            if choice is None:
                                continue
                            finish_reason = choice.get("finish_reason") or finish_reason
                            delta = choice.get("delta")
                            if not isinstance(delta, Mapping):
                                continue

                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                content_parts.append(content)
                                yield ProviderTextDeltaEvent(delta=content)

                            reasoning_content = delta.get("reasoning_content")
                            if isinstance(reasoning_content, str) and reasoning_content:
                                reasoning_content_parts.append(reasoning_content)

                            thinking = _thinking_delta_text(delta)
                            if thinking:
                                yield ProviderThinkingDeltaEvent(delta=thinking)

                            for tool_call_delta in _tool_call_deltas(delta):
                                index = _tool_call_index(tool_call_delta)
                                if index is None:
                                    yield ProviderErrorEvent(
                                        message="Provider returned an invalid tool-call index"
                                    )
                                    return
                                builder = tool_call_builders.setdefault(index, _ToolCallBuilder())
                                builder.add_delta(tool_call_delta)

                        try:
                            tool_calls = [
                                builder.build(index)
                                for index, builder in sorted(tool_call_builders.items())
                            ]
                            _ensure_unique_tool_call_ids(tool_calls)
                        except ValueError as exc:
                            yield ProviderErrorEvent(message=str(exc))
                            return

                        for tool_call in tool_calls:
                            yield ProviderToolCallEvent(tool_call=tool_call)

                        provider_data: dict[str, JSONValue] = {}
                        if reasoning_content_parts:
                            provider_data["reasoning_content"] = "".join(reasoning_content_parts)
                        yield ProviderResponseEndEvent(
                            message=AssistantMessage(
                                content="".join(content_parts),
                                tool_calls=tool_calls,
                                provider_data=provider_data,
                            ),
                            finish_reason=finish_reason,
                        )
                        return
                except httpx.HTTPError as exc:
                    if not response_started and self._should_retry(attempt):
                        delay = retry_delay_seconds(
                            attempt,
                            max_delay_seconds=self._config.max_retry_delay_seconds,
                        )
                        yield make_retry_event(
                            attempt=attempt,
                            max_retries=self._config.max_retries,
                            delay_seconds=delay,
                            reason="network error",
                            data={
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                            },
                        )
                        attempt += 1
                        if not await wait_for_retry(delay, signal=signal):
                            return
                        continue
                    yield ProviderErrorEvent(
                        message=str(exc),
                        data={
                            "error_type": type(exc).__name__,
                            "attempts": attempt + 1,
                        },
                    )
                    return

        return iterator()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._config.timeout_seconds)
        return self._client

    def _should_retry(self, attempt: int, *, status_code: int | None = None) -> bool:
        if attempt >= self._config.max_retries:
            return False
        return status_code is None or _is_transient_status(status_code)


class _ToolCallBuilder:
    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.arguments_parts: list[str] = []

    def add_delta(self, delta: Mapping[str, Any]) -> None:
        call_id = delta.get("id")
        if isinstance(call_id, str) and call_id:
            self.id = call_id

        function = delta.get("function")
        if not isinstance(function, Mapping):
            return

        name = function.get("name")
        if isinstance(name, str) and name:
            self.name = name

        arguments = function.get("arguments")
        if isinstance(arguments, str):
            self.arguments_parts.append(arguments)

    def build(self, index: int) -> ToolCall:
        if not self.id:
            raise ValueError(f"Provider tool call at index {index} has no id")
        if not self.name:
            raise ValueError(f"Provider tool call at index {index} has no function name")

        arguments_text = "".join(self.arguments_parts)
        arguments = _loads_object(arguments_text) if arguments_text else {}
        if arguments is None:
            arguments = {"_raw_arguments": arguments_text}
        return ToolCall(id=self.id, name=self.name, arguments=arguments)


def _build_chat_payload(
    *,
    model: str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    max_tokens: int | None = None,
    thinking_enabled: bool = False,
    reasoning_effort: str | None = None,
    reasoning_effort_parameter: str = "reasoning_effort",
) -> dict[str, JSONValue]:
    payload: dict[str, JSONValue] = {
        "model": model,
        "stream": True,
        "messages": [
            {"role": "system", "content": system},
            *[_message_to_openai(message) for message in messages],
        ],
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if thinking_enabled:
        payload["thinking"] = {"type": "enabled"}
    if reasoning_effort is not None:
        if reasoning_effort_parameter == "reasoning.effort":
            payload["reasoning"] = {"effort": reasoning_effort}
        else:
            payload["reasoning_effort"] = reasoning_effort
    if tools:
        payload["tools"] = [_tool_to_openai(tool) for tool in tools]
    return payload


def _message_to_openai(message: AgentMessage) -> dict[str, JSONValue]:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AssistantMessage):
        item: dict[str, JSONValue] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            item["tool_calls"] = [
                _tool_call_to_openai(tool_call) for tool_call in message.tool_calls
            ]
            reasoning_content = message.provider_data.get("reasoning_content")
            if isinstance(reasoning_content, str):
                item["reasoning_content"] = reasoning_content
        return item
    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "content": message.content,
        }


def _tool_to_openai(tool: AgentTool) -> dict[str, JSONValue]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.input_schema),
        },
    }


def _tool_call_to_openai(tool_call: ToolCall) -> dict[str, JSONValue]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": dumps(tool_call.arguments),
        },
    }


def _parse_sse_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or not stripped.startswith("data:"):
        return None
    return stripped.removeprefix("data:").strip()


def _loads_object(value: str) -> dict[str, JSONValue] | None:
    try:
        loaded = loads(value)
    except JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _first_choice(chunk: Mapping[str, Any]) -> Mapping[str, Any] | None:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    return choice if isinstance(choice, Mapping) else None


def _tool_call_deltas(delta: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [tool_call for tool_call in tool_calls if isinstance(tool_call, Mapping)]


def _tool_call_index(delta: Mapping[str, Any]) -> int | None:
    index = delta.get("index", 0)
    if isinstance(index, bool):
        return None
    if isinstance(index, int):
        return index if index >= 0 else None
    if isinstance(index, str) and index.isdigit():
        return int(index)
    return None


def _thinking_delta_text(delta: Mapping[str, Any]) -> str:
    for field_name in ("reasoning_content", "reasoning", "thinking"):
        value = delta.get(field_name)
        if isinstance(value, str) and value:
            return value
    return ""


def _ensure_unique_tool_call_ids(tool_calls: list[ToolCall]) -> None:
    ids = [tool_call.id for tool_call in tool_calls]
    if len(ids) != len(set(ids)):
        raise ValueError("Provider returned duplicate tool-call ids")


def _is_transient_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429} or status_code >= 500
