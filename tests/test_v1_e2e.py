"""Release-level FakeProvider acceptance test for the complete Axis v1 stack."""

import asyncio
import json
from io import StringIO
from pathlib import Path

from axis_agent import (
    AssistantMessage,
    JsonlSessionStorage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from axis_ai import (
    FakeProvider,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
    ProviderThinkingDeltaEvent,
)
from axis_coding import (
    AxisPaths,
    AxisResourcePaths,
    CodingSession,
    CodingSessionConfig,
    ToolApprovalPolicy,
)
from axis_coding.cli import run_print_mode
from axis_coding.rendering import PrintOutputMode


def test_v1_fake_provider_tool_session_persistence_and_restart(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    (project / "README.md").write_text("Axis v1 fixture\n", encoding="utf-8")
    (project / "AGENTS.md").write_text("Always verify the result.", encoding="utf-8")
    skill_path = project / ".axis" / "skills" / "review.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\ndescription: Review release changes\n---\n# Review\nFull skill body.",
        encoding="utf-8",
    )
    paths = AxisResourcePaths(
        paths=AxisPaths(
            home=tmp_path / "user" / ".axis",
            agents_home=tmp_path / "user" / ".agents",
        ),
        cwd=project,
    )
    storage = JsonlSessionStorage(tmp_path / "axis-session.jsonl")
    call = ToolCall(id="call-read", name="read", arguments={"path": "README.md"})
    tool_request = AssistantMessage(
        tool_calls=[call],
        provider_data={"reasoning_content": "I should inspect the project README."},
    )
    final = AssistantMessage(
        content="AXIS_V1_OK",
        provider_data={"reasoning_content": "The file was read successfully."},
    )
    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="fake-v1"),
                ProviderThinkingDeltaEvent(delta="I should inspect the project README."),
                ProviderResponseEndEvent(message=tool_request, finish_reason="tool_calls"),
            ],
            [
                ProviderResponseStartEvent(model="fake-v1"),
                ProviderThinkingDeltaEvent(delta="The file was read successfully."),
                ProviderTextDeltaEvent(delta="AXIS_V1_OK"),
                ProviderResponseEndEvent(message=final, finish_reason="stop"),
            ],
        ]
    )
    stdout = StringIO()
    stderr = StringIO()

    succeeded = asyncio.run(
        run_print_mode(
            prompt="Inspect README and report readiness",
            model="fake-v1",
            cwd=project,
            provider=provider,
            storage=storage,
            resource_paths=paths,
            output=PrintOutputMode.JSON,
            tool_policy=ToolApprovalPolicy.ALLOW,
            stdout=stdout,
            stderr=stderr,
        )
    )

    events = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert succeeded is True
    assert stderr.getvalue() == ""
    assert events[0]["type"] == "memory_context"
    assert events[1]["type"] == "agent_start"
    assert events[-1]["type"] == "agent_end"
    assert [event["type"] for event in events].count("turn_start") == 2
    assert [event["type"] for event in events].count("tool_execution_start") == 1
    assert [event["type"] for event in events].count("tool_execution_end") == 1
    assert len(provider.calls) == 2
    assert "Always verify the result." in provider.calls[0][1]
    assert "Review release changes" in provider.calls[0][1]
    assert "Full skill body." not in provider.calls[0][1]

    second_messages = provider.calls[1][2]
    assert second_messages[:2] == [
        UserMessage(content="Inspect README and report readiness"),
        tool_request,
    ]
    tool_result = second_messages[2]
    assert isinstance(tool_result, ToolResultMessage)
    assert tool_result.tool_call_id == "call-read"
    assert tool_result.name == "read"
    assert tool_result.ok is True
    assert tool_result.content == "Axis v1 fixture\n"

    restarted = asyncio.run(
        CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="ignored-new-default",
                storage=JsonlSessionStorage(storage.path),
                cwd=project,
                resource_paths=paths,
            )
        )
    )

    assert restarted.model == "fake-v1"
    assert restarted.system == provider.calls[0][1]
    assert restarted.messages == (
        UserMessage(content="Inspect README and report readiness"),
        tool_request,
        tool_result,
        final,
    )
    assert restarted.state.active_leaf_id == restarted.state.context_entry_ids[-1]
