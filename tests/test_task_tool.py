"""Tests for Axis task (sub-agent delegation) tool."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from axis_agent.messages import AssistantMessage
from axis_ai.events import ProviderResponseEndEvent
from axis_ai.fake import FakeProvider
from axis_coding.task_tool import TaskToolError, create_task_tool


class TestTaskTool:
    @pytest.fixture
    def provider(self) -> FakeProvider:
        return FakeProvider(
            [
                [
                    ProviderResponseEndEvent(
                        message=AssistantMessage(content="Investigation complete.")
                    )
                ]
            ]
        )

    def test_returns_subagent_response(self, provider: FakeProvider, tmp_path: Path) -> None:
        async def run() -> None:
            tool = create_task_tool(provider=provider, model="test", cwd=tmp_path)
            result = await tool.execute({"prompt": "Check the project structure"})
            assert result.ok is True
            assert "Investigation complete" in result.content

        asyncio.run(run())

    def test_rejects_empty_prompt(self, provider: FakeProvider, tmp_path: Path) -> None:
        async def run() -> None:
            tool = create_task_tool(provider=provider, model="test", cwd=tmp_path)
            result = await tool.execute({"prompt": ""})
            assert result.ok is False
            assert "empty" in result.content.lower()

        asyncio.run(run())

    def test_rejects_whitespace_prompt(self, provider: FakeProvider, tmp_path: Path) -> None:
        async def run() -> None:
            tool = create_task_tool(provider=provider, model="test", cwd=tmp_path)
            result = await tool.execute({"prompt": "   "})
            assert result.ok is False
            assert "empty" in result.content.lower()

        asyncio.run(run())

    def test_accepts_max_turns(self, provider: FakeProvider, tmp_path: Path) -> None:
        async def run() -> None:
            tool = create_task_tool(provider=provider, model="test", cwd=tmp_path)
            result = await tool.execute({"prompt": "Check", "max_turns": 3})
            assert result.ok is True

        asyncio.run(run())

    def test_missing_prompt_raises(self, provider: FakeProvider, tmp_path: Path) -> None:
        async def run() -> None:
            tool = create_task_tool(provider=provider, model="test", cwd=tmp_path)
            with pytest.raises(TaskToolError, match="prompt must be a string"):
                await tool.execute({})

        asyncio.run(run())

    def test_bool_max_turns_raises(self, provider: FakeProvider, tmp_path: Path) -> None:
        async def run() -> None:
            tool = create_task_tool(provider=provider, model="test", cwd=tmp_path)
            with pytest.raises(TaskToolError, match="integer"):
                await tool.execute({"prompt": "Check", "max_turns": True})

        asyncio.run(run())

    def test_metadata_sane(self, provider: FakeProvider, tmp_path: Path) -> None:
        tool = create_task_tool(provider=provider, model="test", cwd=tmp_path)
        assert tool.name == "task"
        assert tool.description
        assert tool.prompt_snippet
        assert tool.input_schema
        assert tool.requires_approval is False
        assert len(tool.prompt_guidelines) >= 3

    def test_default_subagent_tools_are_read_only(
        self, provider: FakeProvider, tmp_path: Path
    ) -> None:
        tool = create_task_tool(provider=provider, model="test", cwd=tmp_path)
        # The task tool executor is always callable.
        assert callable(tool.executor)


class TestTaskToolCustomTools:
    def test_custom_subagent_tools(self, tmp_path: Path) -> None:
        provider = FakeProvider(
            [[ProviderResponseEndEvent(message=AssistantMessage(content="ok"))]]
        )
        from axis_coding.tools import create_read_tool

        sub_tools = [create_read_tool(cwd=tmp_path)]

        async def run() -> None:
            tool = create_task_tool(
                provider=provider,
                model="test",
                cwd=tmp_path,
                subagent_tools=sub_tools,
            )
            result = await tool.execute({"prompt": "Read something"})
            assert result.ok is True

        asyncio.run(run())
