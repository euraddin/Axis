# Axis

**A personal coding agent, built from scratch to learn how these things work.**

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![tests](https://img.shields.io/badge/tests-624%20passed-brightgreen.svg)](.)
[![mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](pyproject.toml)
[![ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

Axis is a terminal-native coding agent I built to understand how AI coding
assistants work under the hood.  It runs locally, talks to DeepSeek (or any
OpenAI-compatible API), and can read, write, search, and operate on your
codebase — with your explicit approval at every step.

## Installation

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/euraddin/Axis.git
cd Axis
uv sync --dev
uv run axis
```

Set your API key:

```bash
export DEEPSEEK_API_KEY="sk-..."
```

## Usage

```bash
# Interactive TUI
axis

# One-shot prompt
axis -p "Explain the architecture of this project"

# Target a different directory with a specific model
axis -p "Fix the failing tests" --cwd ~/my-project --model deepseek-v4-flash

# JSON output for scripting
axis -p "List all Python files" --output json
```

## What it can do

12 built-in tools:

| Tool | Purpose |
|------|---------|
| `read` | Read files, with image support |
| `write` | Create or overwrite a file |
| `edit` | Exact string replacements with atomic validation |
| `bash` | Run shell commands (read-only commands are auto-approved) |
| `git_status` / `git_diff` / `git_log` / `git_commit` | Git workflow |
| `lint` | Run the project linter |
| `web_fetch` / `web_search` | Fetch web pages and search the web |
| `task` | Delegate investigation to a read-only sub-agent |

Plus streaming TUI, automatic context compaction, session persistence (JSONL),
multi-model switching, MCP integration, a project memory bank, voice input
(Volcengine ASR), and a Docker sandbox mode for bash.

This is a personal project — it will never be as capable as Claude Code or
Cursor.  But if you want to understand how a coding agent works internally,
this codebase (~6,500 lines) is small enough to read in an afternoon.

## Architecture

Three layers with strict, test-enforced one-way dependencies:

```
axis_ai         Model adapters — provider protocol + OpenAI-compatible SSE
    ↓
axis_agent      Portable agent layer — event system, messages, tools, agent loop
    ↓
axis_coding     Product layer — 12 tools, CLI, TUI, MCP, memory, sessions
```

The agent loop is provider-agnostic: it doesn't know which LLM it's talking to.
The TUI and print-mode CLI consume the same event stream — there's no second
agent implementation.

## Development

```bash
uv sync --dev
uv run pytest                 # 624 tests
uv run ruff check .           # lint
uv run mypy                   # strict type check
```

## License

MIT — see [LICENSE](LICENSE).
