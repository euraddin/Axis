# Changelog

All notable changes to Axis are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — unreleased

### Added

- Core three-layer architecture: `axis_ai`, `axis_agent`, `axis_coding`
- Provider-neutral agent loop with streaming event system (19 event types)
- OpenAI-compatible SSE provider adapter with configurable retry logic
- Deterministic `FakeProvider` for testing
- Textual TUI with streaming thinking/response display
- Print mode (non-interactive CLI) with `text`, `json`, and `transcript` output
- Twelve built-in tools: `read`, `write`, `edit`, `bash`, `git_status`,
  `git_diff`, `git_log`, `git_commit`, `lint`, `web_fetch`, `web_search`, `task`
- Automatic context compaction with configurable thresholds
- Append-only JSONL session storage with tree-based branching and export
- Multi-provider support with runtime model switching
- MCP (Model Context Protocol) integration for external tools
- Memory Bank — durable project memory with auto-proposal generation
- Voice input via Volcengine Seed ASR 2.0 with context-aware polishing
- Slash commands: `/help`, `/model`, `/thinking`, `/theme`, `/memory`,
  `/reload`, `/export`, `/session`, `/login`, `/voice`, `/exit`
- Skills system with per-project and per-user precedence resolution
- Prompt templates with frontmatter and slash-command invocation
- Sandbox mode for bash (Docker container with read-only mount)
- SSRF protection for `web_fetch` (DNS resolution + private IP blocking)
- Sub-agent delegation with isolated context window (`task` tool)
- Five built-in TUI themes: `axis-dark`, `axis-light`, `high-contrast`,
  `omni`, `terminal-native`
- Bash command auto-approval classifier (recognises read-only patterns)
- Tool approval system with per-call and per-session decisions
- Session export to HTML and JSONL
- Configuration via environment variables, `~/.axis/` files, and CLI flags
- Python 3.14+, `mypy --strict`, `ruff` linting + formatting
- 624 test cases across ~50 test files
