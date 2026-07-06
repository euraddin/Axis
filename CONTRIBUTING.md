# Contributing to Axis

Thanks for your interest in contributing! Axis is a personal coding agent
built from first principles, and contributions that improve its transparency,
reliability, or extensibility are welcome.

## Getting started

1. **Fork** the repository and clone your fork.
2. Install the development environment: `uv sync --dev`
3. Create a branch: `git checkout -b your-feature`
4. Make your changes and verify:
   ```bash
   uv run pytest -q -W error
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy
   ```
5. Push and open a pull request against `main`.

## What to contribute

- **Bug fixes** — if you find a bug, please open an issue first so we can
  discuss the best fix.
- **New tools** — adding new built-in tools (e.g., a `notebook_edit` tool
  for Jupyter notebooks) is a great way to contribute.  Follow the patterns
  in `src/axis_coding/tools.py`.
- **New providers** — adding native support for non-OpenAI-compatible APIs
  (Anthropic, Google Gemini, etc.) via the `ModelProvider` protocol.
- **Documentation** — improving README, docstrings, tutorials, or adding
  examples.
- **Tests** — filling gaps in test coverage, adding integration tests.

## Design principles

1. **Keep the layers separate.**  `axis_agent` must not import `axis_coding`,
   `textual`, or `rich`.  This is enforced by an architecture test in
   `tests/test_architecture.py`.

2. **Provider-neutral.**  The agent loop in `axis_agent/loop.py` works with
   any `ModelProvider` implementation.  New tools should not depend on a
   specific LLM provider.

3. **Deterministic when possible.**  Use `FakeProvider` for tests.  The
   agent loop should be testable without real HTTP calls.

4. **Typed.**  Axis uses `mypy --strict`.  All new code must pass type
   checking without errors or `# type: ignore` comments (unless there is
   a documented reason).

5. **Small PRs.**  Break large changes into logical, self-contained commits.
   Each commit should pass CI independently.

## Code conventions

- Python 3.14+ with modern syntax (`match`/`case` welcome, `|` unions).
- Line length: 100 characters (configured in `pyproject.toml`).
- Double quotes for strings (enforced by `ruff format`).
- Imports sorted with `ruff check --fix`.
- Pydantic models for data (use `ConfigDict(extra="forbid")`).

## Project structure

```
src/
  axis_ai/           Model provider layer
  axis_agent/         Portable agent layer
  axis_coding/        Coding agent product layer
    tools.py            read/write/edit/bash
    git_tools.py        git_status/git_diff/git_log/git_commit
    web_tools.py        web_fetch/web_search
    lint_tools.py       project linter integration
    task_tool.py        sub-agent delegation
    tui/                Textual terminal UI
    mcp/                MCP client & manager
    memory_bank.py      durable project memory
tests/                Tests mirror the src structure
```

## Testing

Write tests using `pytest`.  Prefer synchronous tests with `asyncio.run()`
for async code (this is the existing project convention).  Use `tmp_path`
for file-system isolation and `FakeProvider` for LLM-call simulation.

```python
from axis_ai.fake import FakeProvider
from axis_ai.events import ProviderResponseEndEvent
from axis_agent.messages import AssistantMessage

def test_my_tool():
    provider = FakeProvider([
        [ProviderResponseEndEvent(message=AssistantMessage(content="result"))]
    ])
    # ...
```

## Questions?

Open an issue or start a discussion.  If you're unsure whether something fits,
ask before writing code — it saves everyone time.
