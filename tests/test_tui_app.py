"""Headless Textual tests for Axis's basic interactive frontend."""

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path

from textual.widgets import Input

from axis_agent import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    ErrorEvent,
    JsonlSessionStorage,
    ToolCall,
    TurnEndEvent,
    TurnStartEvent,
)
from axis_agent.types import JSONValue
from axis_ai import (
    FakeProvider,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
    ProviderThinkingDeltaEvent,
)
from axis_coding import CodingSession, CodingSessionConfig
from axis_coding.tui import AxisTuiApp, TuiMessageItem, TuiNoticeItem, TuiToolItem


async def wait_until(predicate: Callable[[], bool], *, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        await asyncio.sleep(0.01)


def test_tui_submits_prompt_and_renders_streamed_thinking_and_text(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        provider = FakeProvider(
            [
                [
                    ProviderResponseStartEvent(model="fake"),
                    ProviderThinkingDeltaEvent(delta="Inspect first."),
                    ProviderTextDeltaEvent(delta="Done"),
                    ProviderResponseEndEvent(message=AssistantMessage(content="Done")),
                ]
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
                cwd=tmp_path,
                tools=[],
            )
        )
        app = AxisTuiApp(session)

        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = "Hello Axis"
            await pilot.press("enter")
            await wait_until(lambda: len(provider.calls) == 1 and not prompt.disabled)

            messages = [item for item in app.state.items if isinstance(item, TuiMessageItem)]
            assert [(item.role, item.text) for item in messages] == [
                ("user", "Hello Axis"),
                ("thinking", "Inspect first."),
                ("assistant", "Done"),
            ]
            assert app.state.running is False
            assert prompt.value == ""
            assert prompt.has_focus is True

    asyncio.run(scenario())


def test_tui_surfaces_prompt_expansion_errors_and_reenables_input(tmp_path: Path) -> None:
    async def scenario() -> None:
        provider = FakeProvider([])
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
                cwd=tmp_path,
                tools=[],
            )
        )
        app = AxisTuiApp(session)

        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = "/skill:missing"
            await pilot.press("enter")
            await wait_until(lambda: app.state.error is not None and not prompt.disabled)

            assert app.state.error == "Unknown skill: missing"
            assert provider.calls == []
            notice = app.state.items[-1]
            assert isinstance(notice, TuiNoticeItem)
            assert notice.level == "error"

    asyncio.run(scenario())


def test_tui_tracks_a_complete_tool_round_trip(tmp_path: Path) -> None:
    async def scenario() -> None:
        async def execute(
            arguments: Mapping[str, JSONValue],
            signal: object | None = None,
        ) -> AgentToolResult:
            del signal
            return AgentToolResult(
                tool_call_id="provider-will-replace",
                name="read",
                ok=True,
                content=f"contents of {arguments['path']}",
            )

        call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
        provider = FakeProvider(
            [
                [ProviderResponseEndEvent(message=AssistantMessage(tool_calls=[call]))],
                [ProviderResponseEndEvent(message=AssistantMessage(content="Finished"))],
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "tool-session.jsonl"),
                cwd=tmp_path,
                tools=[AgentTool("read", "Read", {"type": "object"}, execute)],
            )
        )
        app = AxisTuiApp(session)

        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = "Read README"
            await pilot.press("enter")
            await wait_until(lambda: len(provider.calls) == 2 and not prompt.disabled)

            tool_items = [item for item in app.state.items if isinstance(item, TuiToolItem)]
            assert len(tool_items) == 1
            assert tool_items[0].status == "succeeded"
            assert tool_items[0].result is not None
            assert tool_items[0].result.tool_call_id == "call-1"
            assert tool_items[0].result.content == "contents of README.md"

    asyncio.run(scenario())


class CancellableSession:
    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd
        self.model = "fake"
        self._running = False
        self.cancel_called = False
        self._cancel_event = asyncio.Event()

    @property
    def is_running(self) -> bool:
        return self._running

    async def prompt(self, content: str) -> AsyncIterator[AgentEvent]:
        del content
        self._running = True
        try:
            yield AgentStartEvent()
            yield TurnStartEvent(turn=1)
            await self._cancel_event.wait()
            yield ErrorEvent(message="Agent run cancelled", recoverable=True)
            yield TurnEndEvent(turn=1)
            yield AgentEndEvent()
        finally:
            self._running = False

    def cancel(self) -> None:
        self.cancel_called = True
        self._cancel_event.set()


def test_escape_requests_cooperative_cancellation(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = CancellableSession(tmp_path)
        app = AxisTuiApp(session)

        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = "Long task"
            await pilot.press("enter")
            await wait_until(lambda: session.is_running)
            await pilot.press("escape")
            await wait_until(lambda: app.state.cancelled and not prompt.disabled)

            assert session.cancel_called is True
            assert app.state.error is None
            assert app.state.running is False
            assert any(
                isinstance(item, TuiNoticeItem) and item.text == "Agent run cancelled."
                for item in app.state.items
            )

    asyncio.run(scenario())


def test_ctrl_d_exits_the_app(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = CancellableSession(tmp_path)
        app = AxisTuiApp(session)

        async with app.run_test() as pilot:
            assert app.is_running is True
            await pilot.press("ctrl+d")
            await wait_until(lambda: not app.is_running)

    asyncio.run(scenario())
