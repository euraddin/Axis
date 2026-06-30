"""Self-contained HTML and JSONL exports for Axis session trees."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from html import escape
from pathlib import Path

from axis_agent import (
    AssistantMessage,
    BranchSummaryEntry,
    CompactionEntry,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionEntry,
    SessionInfoEntry,
    ThinkingLevelChangeEntry,
    ToolResultMessage,
    UserMessage,
    entry_to_json_line,
    path_to_entry,
)


class SessionExportError(ValueError):
    """A requested session export format or destination is invalid."""


def normalize_export_format(value: str | None) -> str:
    normalized = (value or "html").strip().casefold().removeprefix(".")
    if normalized in {"htm", "html"}:
        return "html"
    if normalized == "jsonl":
        return "jsonl"
    raise SessionExportError(f"Unsupported export format: {value}")


def default_session_export_path(
    *,
    destination_dir: Path,
    session_name: str,
    format: str,
) -> Path:
    suffix = ".jsonl" if normalize_export_format(format) == "jsonl" else ".html"
    return destination_dir / f"{session_name}{suffix}"


def export_session_artifact(
    entries: Sequence[SessionEntry],
    output_path: Path,
    *,
    title: str = "Axis Session Export",
    source: str | None = None,
    format: str | None = None,
) -> Path:
    export_format = normalize_export_format(format or output_path.suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if export_format == "jsonl":
        output_path.write_text(
            "".join(entry_to_json_line(entry) for entry in entries), encoding="utf-8"
        )
    else:
        output_path.write_text(
            render_session_html(entries, title=title, source=source),
            encoding="utf-8",
        )
    return output_path


def render_session_html(
    entries: Sequence[SessionEntry],
    *,
    title: str = "Axis Session Export",
    source: str | None = None,
) -> str:
    """Render the complete branch tree and entry details as standalone HTML."""
    logical = [entry for entry in entries if not isinstance(entry, LeafEntry)]
    active_leaf = _active_leaf_id(entries)
    active_path = (
        {entry.id for entry in path_to_entry(entries, active_leaf)} if active_leaf else set()
    )
    children: dict[str | None, list[SessionEntry]] = defaultdict(list)
    for entry in logical:
        children[entry.parent_id].append(entry)
    tree = "".join(
        _render_tree_node(root, children, active_path, active_leaf) for root in children[None]
    )
    details = "".join(_render_entry(entry, active_path, active_leaf) for entry in logical)
    source_row = f'<p class="muted">Source: <code>{escape(source)}</code></p>' if source else ""
    generated = datetime.now(UTC).replace(microsecond=0).isoformat()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; background: #111; color: #e5e7eb; line-height: 1.5; }}
    header, main {{ padding: 20px clamp(16px, 4vw, 44px); }}
    main {{ display: grid; grid-template-columns: minmax(240px, 340px) 1fr; gap: 20px; }}
    aside, article {{ border: 1px solid #374151; border-radius: 8px; padding: 14px; }}
    aside {{ position: sticky; top: 16px; align-self: start; max-height: 90vh; overflow: auto; }}
    ul {{ list-style: none; padding-left: 14px; }}
    a {{ color: #93c5fd; text-decoration: none; }}
    .node {{ display: block; border-left: 3px solid #4b5563; padding: 4px 8px; margin: 6px 0; }}
    .active-path > .node {{ border-color: #2dd4bf; }}
    .active-leaf > .node {{ background: #153f38; }}
    article {{ margin-bottom: 12px; overflow-wrap: anywhere; }}
    pre {{ white-space: pre-wrap; background: #161b22; padding: 10px; border-radius: 6px; }}
    .muted {{ color: #9ca3af; }}
    @media (max-width: 760px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ position: static; }}
    }}
  </style>
</head>
<body>
<header><h1>{escape(title)}</h1>{source_row}<p class="muted">Generated: {generated}</p></header>
<main>
  <aside><h2>Session tree</h2><ul>{tree}</ul></aside>
  <section>{details or "<p>No session entries.</p>"}</section>
</main>
</body>
</html>
"""


def _active_leaf_id(entries: Sequence[SessionEntry]) -> str | None:
    return next(
        (entry.entry_id for entry in reversed(entries) if isinstance(entry, LeafEntry)),
        None,
    )


def _render_tree_node(
    entry: SessionEntry,
    children: dict[str | None, list[SessionEntry]],
    active_path: set[str],
    active_leaf: str | None,
) -> str:
    classes = []
    if entry.id in active_path:
        classes.append("active-path")
    if entry.id == active_leaf:
        classes.append("active-leaf")
    nested = "".join(
        _render_tree_node(child, children, active_path, active_leaf)
        for child in children.get(entry.id, [])
    )
    class_attr = f' class="{" ".join(classes)}"' if classes else ""
    return (
        f'<li{class_attr}><a class="node" href="#entry-{escape(entry.id)}">'
        f"{escape(_entry_label(entry))}</a>"
        f"{'<ul>' + nested + '</ul>' if nested else ''}</li>"
    )


def _render_entry(entry: SessionEntry, active_path: set[str], active_leaf: str | None) -> str:
    classes = ["active-path"] if entry.id in active_path else []
    if entry.id == active_leaf:
        classes.append("active-leaf")
    class_attr = f' class="{" ".join(classes)}"' if classes else ""
    body = escape(_entry_body(entry))
    return (
        f'<article id="entry-{escape(entry.id)}"{class_attr}>'
        f"<h3>{escape(_entry_label(entry))}</h3>"
        f'<p class="muted">{escape(entry.type)} · {escape(entry.id)}</p>'
        f"<pre>{body}</pre></article>"
    )


def _entry_label(entry: SessionEntry) -> str:
    if isinstance(entry, SessionInfoEntry):
        return f"session: {entry.title or 'Untitled'}"
    if isinstance(entry, ModelChangeEntry):
        return f"model: {entry.model}"
    if isinstance(entry, ThinkingLevelChangeEntry):
        return f"thinking: {entry.thinking_level}"
    if isinstance(entry, MessageEntry):
        message = entry.message
        tools = ""
        if isinstance(message, AssistantMessage) and message.tool_calls:
            tools = " [" + ", ".join(call.name for call in message.tool_calls) + "]"
        return f"{message.role}: {_preview(message.content)}{tools}"
    if isinstance(entry, CompactionEntry):
        return f"compaction: {_preview(entry.summary)}"
    if isinstance(entry, BranchSummaryEntry):
        return f"branch summary: {_preview(entry.summary)}"
    return entry.type


def _entry_body(entry: SessionEntry) -> str:
    if isinstance(entry, MessageEntry):
        message = entry.message
        if isinstance(message, ToolResultMessage):
            return f"{message.name} ({'ok' if message.ok else 'failed'})\n{message.content}"
        if isinstance(message, UserMessage | AssistantMessage):
            return message.content
    if isinstance(entry, CompactionEntry):
        replaced = ", ".join(entry.replaces_entry_ids) or "none"
        return f"{entry.summary}\n\nReplaces entries: {replaced}"
    if isinstance(entry, BranchSummaryEntry):
        return entry.summary
    if isinstance(entry, ModelChangeEntry):
        return entry.model
    if isinstance(entry, ThinkingLevelChangeEntry):
        return entry.thinking_level
    if isinstance(entry, SessionInfoEntry):
        return f"cwd: {entry.cwd or ''}\ntitle: {entry.title or ''}"
    return entry.model_dump_json(indent=2)


def _preview(text: str, *, limit: int = 72) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1]}…"
