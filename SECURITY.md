# Security Policy

## Supported Versions

Axis is currently in active development (pre-1.0).  Security fixes are
applied to the `main` branch.

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public
issue.  Instead, report it via
[GitHub Security Advisory](https://github.com/euraddin/Axis/security/advisories/new)
or email the maintainer directly.

You can expect:

- An acknowledgement within 48 hours
- A status update within 5 business days
- Credit in the release notes (unless you prefer to remain anonymous)

## Security Model

Axis is a **local coding agent**.  It runs on your machine with your user
permissions.  Understanding this model is important for using it safely:

### What Axis does

- Reads and writes files in directories you specify
- Executes shell commands you approve
- Sends prompts and tool outputs to your configured LLM provider
- Stores conversation history in `~/.axis/sessions/`

### What Axis does NOT do

- Send data to any server other than your configured LLM provider
- Collect telemetry, analytics, or usage statistics
- Access the network except for provider API calls, MCP servers, and
  explicit `web_fetch` / `web_search` tool invocations

### Protections in place

1. **Tool approval** — destructive tools (`write`, `edit`, `bash`,
   `git_commit`) require explicit user approval.  The TUI shows the full
   tool parameters before asking.

2. **Bash auto-approval** — read-only shell commands (e.g., `ls`, `git diff`,
   `grep`) are classified and can be auto-approved.  Destructive patterns
   (output redirection, `tee`) are rejected by the classifier and always
   require approval.

3. **SSRF protection** — the `web_fetch` tool resolves DNS and rejects URLs
   that resolve to private/internal IP ranges (RFC 1918, loopback,
   link-local, multicast).

4. **Sandbox mode** — the bash tool supports an optional Docker sandbox
   (`"sandbox": true`) that runs commands in an isolated container with
   the working directory mounted read-only and networking disabled.

5. **Credential isolation** — API keys are only read from environment
   variables or `~/.axis/credentials.json` (permissions 600).  They are
   never written to session files or committed to the repository.
   `.env.example` contains placeholder values only.

6. **No secrets in session files** — the memory bank sanitises file paths
   and evidence before generating proposals.  Secrets, command output, and
   personal data are stripped from project memory.

### Important caveats

- **Axis is NOT a sandbox by default.**  When `sandbox` is not enabled,
  bash commands run with your full user permissions.  The approval system
  is a UX safety net, not an OS-level security boundary.

- **The LLM provider receives your data.**  Prompts, file contents (when
  read), and command outputs are sent to your configured LLM provider.
  Choose a provider you trust, and do not use Axis with sensitive code or
  data if you are uncomfortable with this.

- **MCP servers have their own security model.**  Review the permissions
  of any MCP server you connect before enabling it.

## Best Practices

1. **Use sandbox mode** for commands you don't fully trust:
   ```json
   {"command": "pip install some-package", "sandbox": true}
   ```
2. **Review staged changes** before committing:
   ```
   Run git_diff with staged=true to see what will be committed.
   ```
3. **Rotate API keys** if they are accidentally exposed in session logs or
   environment dumps.
4. **Keep Axis updated** — security fixes are released on `main`.
5. **Use a dedicated API key** for Axis with minimal permissions rather than
   sharing keys across applications.
