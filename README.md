# Axis

Axis 是一个从底层开始构建的个人 Python coding agent。项目采用三层架构，并把每个阶段都当成可运行、可测试、可讲清楚的学习单元。

## 当前状态

阶段 7 已完成：DeepSeek V4 Provider 支持文本、thinking、工具调用、reasoning 回传和可取消重试；`deepseek-v4-pro` 真实流式 smoke test 已通过。

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

## DeepSeek smoke test

密钥只通过环境变量提供，不写入源码或提交到 Git：

```bash
export DEEPSEEK_API_KEY="你的密钥"
uv run python -m axis_ai.smoke
unset DEEPSEEK_API_KEY
```

smoke 默认使用 `deepseek-v4-pro`、thinking 与 128 token 输出上限。可通过 `DEEPSEEK_MODEL`、`DEEPSEEK_REASONING_EFFORT` 和 `DEEPSEEK_MAX_TOKENS` 临时覆盖。
