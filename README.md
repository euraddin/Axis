# Axis

Axis 是一个从底层开始构建的个人 Python coding agent。项目采用三层架构，并把每个阶段都当成可运行、可测试、可讲清楚的学习单元。

## 当前状态

Axis v1 的功能、测试、构建、隔离安装与真实 DeepSeek smoke 均已通过。曾误写入 tracked `.env.example` 的旧 DeepSeek key 已在服务端撤销并轮换；`main` 的可达 Git 历史、当前工作树和发布产物均已完成凭据扫描。新 key 只保存在被 Git 忽略的本地 `.env` 中。

## 架构

```text
axis_ai       模型 provider 与流式事件适配层
axis_agent    可复用的消息、工具合同、agent loop 与 harness
axis_coding   coding tools、会话、资源、CLI 与 TUI 产品层
```

## 开发环境

Axis 使用 Python 3.14 和 uv：

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

## 安装

在仓库中创建开发环境并安装 `axis` 命令：

```bash
uv sync --dev
uv run axis --version
```

也可以安装为独立的本地命令：

```bash
uv tool install .
axis --version
```

Axis 需要 Python 3.14。

## DeepSeek smoke test

密钥只通过环境变量提供，不写入源码或提交到 Git：

```bash
export DEEPSEEK_API_KEY="你的密钥"
uv run python -m axis_ai.smoke
unset DEEPSEEK_API_KEY
```

smoke 默认使用 `deepseek-v4-pro`、thinking 与 128 token 输出上限。可通过 `DEEPSEEK_MODEL`、`DEEPSEEK_REASONING_EFFORT` 和 `DEEPSEEK_MAX_TOKENS` 临时覆盖。

## 安全边界

- 密钥只能通过环境变量或被 Git 忽略的本地 `.env` 提供；`.env.example` 必须保持为明显的无效占位符。
- 如果凭据曾进入 Git，即使当前文件已经删除或替换，也必须先在服务端撤销/轮换，再处理公开历史。
- 模型发起 `read`、`write`、`edit` 或 `bash` 时，TUI 会在执行前展示参数并要求用户选择单次允许、本会话允许该工具或拒绝。
- 工具仍采用本地信任模型：允许绝对路径和任意 shell；人工审批不是操作系统权限沙箱。
- JSONL session 会保存完整 prompt、reasoning metadata、工具结果和 system snapshot，应按本地敏感开发数据保护。

## Print CLI

```bash
axis --version
axis -p "解释当前项目结构"
axis -p "运行测试并修复失败" --cwd /path/to/project
axis -p "检查代码" --model deepseek-v4-flash
axis -p "检查代码" --output json
axis -p "检查代码" --output transcript
axis -p "运行测试" --tool-policy allow
```

Print CLI 会启用四个本地工具，`--tool-policy ask|deny|allow` 默认使用 `ask`。没有交互 TTY 时 `ask` 会安全拒绝工具；自动化若要执行工具，必须显式传入 `--tool-policy allow`。该审批层不是权限沙箱。

每次真实 DeepSeek print-mode 运行都会创建独立 JSONL 会话：

```text
~/.axis/sessions/<project-slug>-<path-hash>/<session-id>.jsonl
```

Axis 会按“用户通用 → 项目具体”的顺序读取 `AGENTS.md`：`~/.axis`、`~/.agents`、项目根到当前目录的层级文件、项目根 `.axis`、项目根 `.agents`。项目边界由最近的常见项目标记确定；找不到标记时只读取当前目录，避免越界吸收无关规则。

- `text`：成功后只向 stdout 写最终 assistant 文本；
- `json`：每个 AgentEvent 都是独立一行 JSON，适合程序消费；
- `transcript`：assistant 文本实时写 stdout，工具、retry 和错误写 stderr，thinking 默认隐藏。

## Skills 与 prompt templates

Skills 支持两种布局：

```text
~/.axis/skills/python-testing/SKILL.md
~/.agents/skills/git-review.md
<project>/.axis/skills/...
<project>/.agents/skills/...
```

显式调用：

```bash
axis -p "/skill:python-testing 为 parser.py 补测试"
```

Prompt template 是 `prompts/` 中的 Markdown 文件，文件名就是 slash command。`review.md` 可通过 `/review src/app.py` 调用；模板内可使用 `{{ arguments }}` 或 `{{ args }}`。

两类资源都支持简单 frontmatter：

```md
---
description: Review Python changes safely.
---
```

同名资源按大小写不敏感匹配，项目资源覆盖用户资源；所有覆盖、冲突与读取错误都会留下诊断。

## System prompt

Axis 的 canonical system prompt 按固定顺序装配：身份、工作原则、工具、工具指南、skill 轻量索引、项目指令和环境。Skill 正文不会常驻 system prompt；模型只在任务匹配时通过 `read` 加载对应绝对路径。

新会话会把完整 system snapshot 保存到 `SessionInfoEntry`。跨进程恢复时优先使用该快照，因此日期或本地资源后来发生变化，也不会改写模型已经建立的历史上下文。

`CodingSessionConfig.system` 是完全覆盖：只要显式传入（包括空字符串），Axis 就不会偷偷追加默认身份、资源或环境段落。

## Textual TUI

不带 `-p` 运行 `axis` 会进入交互界面：

```bash
axis
axis --cwd /path/to/project
axis --model deepseek-v4-flash
```

- 输入任务并按 Enter 提交；
- Escape 请求协作式取消当前 run；
- Ctrl+D 取消活动任务并退出；
- thinking、assistant 文本、工具状态、retry 和错误按事件实时显示。

TUI 与 print mode 使用同一个 CodingSession、system prompt、工具和 JSONL 存储，不维护第二套 agent 逻辑。

每次请求 Provider 前，CodingSession 会估算 system、messages 和工具定义的总 token；默认达到模型上下文窗口的 80% 时自动压缩。压缩保留最近约 20K token 的完整用户轮次，把更早内容整理为结构化摘要，原始 JSONL 历史不会删除。可用 `--auto-compact-threshold` 覆盖触发值，用 `--compact-retain-tokens` 调整原文保留窗口。

### 上下文感知语音输入

TUI 支持“豆包 Seed ASR 2.0 实时转写 → DeepSeekV4pro 上下文润色 → 编辑器草稿”的语音管线。它不会自动提交，也不会保存音频或未提交的原始转写。

首次使用，在 TUI 中运行：

```text
/voice setup
```

输入火山引擎 Seed ASR 2.0 API key，选择麦克风并完成 3 秒测试。密钥保存在权限受限的 `~/.axis/credentials.json`，非秘密设置保存在 `~/.axis/voice.json`。也可以只注入环境变量：

```bash
export VOLCENGINE_ASR_API_KEY="你的火山语音 API key"
```

日常操作：

- F2 开始录音，再按 F2 停止并润色；
- Escape 优先丢弃当前录音，不会误取消正在运行的 Agent；
- 临时识别只显示在状态区，最终文本才插入冻结的光标或选区；
- Agent 正在回答或执行工具时仍可录音，完成后由你决定用 Enter steering 还是 Alt+Enter follow-up；
- DeepSeek 润色失败时保留原始 ASR 文本，并在 TUI 中明确警告；
- `/voice` 查看当前设备、凭据和快捷键状态。

默认快捷键可在 `~/.axis/tui.json` 中修改：

```json
{
  "keybindings": {
    "voice_record": "f2"
  }
}
```

如果终端本身使用了透明背景，可在 TUI 中运行：

```text
/theme terminal-native
```

该主题使用终端默认前景色、背景色和 ANSI 调色板，因此会继承终端软件的配色与透明度；焦点、选中和错误状态仍保留高对比显示。

润色只使用当前 active session 的有界快照：光标附近草稿、已有摘要、最近六条可见对话、cwd、Git branch、skill 名称和脱敏后的工具元数据。它不会读取 reasoning、完整工具输出、文件正文、diff、bash 命令或其他会话。每次润色后，TUI 会显示 rules、raw ASR、editor、session 和 coding metadata 的估算 token 比例。

macOS 首次录音时，需要在“系统设置 → 隐私与安全性 → 麦克风”中允许当前终端访问麦克风。

## v1 验收

```bash
uv run pytest -q -W error
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv lock --check
uv build
uv pip check
```

发布验收还包括：FakeProvider 跨层 E2E、六种包导入顺序、全新 editable/wheel 安装、console script，以及真实 `deepseek-v4-pro` 流式请求。

## v2 范围

多 Provider 与登录、model picker、session resume/export、context accounting、branching、完整 slash registry、TUI autocomplete/主题/extensions，以及权限沙箱留待 v2。
