"""Selectable themed transcript widgets for Axis's Textual frontend."""

from dataclasses import dataclass
from typing import Any

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.containers import Horizontal, VerticalScroll
from textual.geometry import Offset
from textual.selection import Selection
from textual.widgets import Markdown as TextualMarkdown
from textual.widgets import Static

from axis_coding.tui.autocomplete import CompletionState
from axis_coding.tui.config import AXIS_DARK_THEME, TuiRoleStyle, TuiTheme
from axis_coding.tui.state import ChatItem, TuiState, visible_chat_text


@dataclass(frozen=True, slots=True)
class TranscriptLine:
    """Plain compatibility view of one rendered transcript line."""

    text: str


class NonSelectableStatic(Static):
    """A visual gutter that must not become clipboard text."""

    ALLOW_SELECT = False


def render_completion_suggestions(
    state: CompletionState,
    *,
    theme: TuiTheme = AXIS_DARK_THEME,
) -> RenderableType:
    """Render aligned, categorized completion rows."""
    table = Table.grid(expand=True)
    table.add_column(no_wrap=True)
    table.add_column(ratio=1)
    previous_category: str | None = None
    for index, item in enumerate(state.items):
        if item.category != previous_category:
            if index:
                table.add_row(Text(""), Text(""))
            if item.category:
                table.add_row(
                    Text(item.category, style=theme.completion_description),
                    Text(""),
                )
            previous_category = item.category
        selected = index == state.selected_index
        row_style = theme.completion_selected if selected else theme.prompt_text
        description_style = (
            theme.completion_selected_description if selected else theme.completion_description
        )
        label = Text("› " if selected else "  ", style=row_style)
        label.append(item.display, style=row_style)
        label.append("  ", style=row_style)
        table.add_row(label, Text(item.description or "", style=description_style))
    return table


class ThemedMarkdownWidget(TextualMarkdown):
    """Native Textual Markdown using Axis theme variables."""

    DEFAULT_CSS = """
    ThemedMarkdownWidget MarkdownH1,
    ThemedMarkdownWidget MarkdownH2,
    ThemedMarkdownWidget MarkdownH3,
    ThemedMarkdownWidget MarkdownH4,
    ThemedMarkdownWidget MarkdownH5,
    ThemedMarkdownWidget MarkdownH6 {
        color: $axis-markdown-highlight;
        content-align: left middle;
        text-style: bold;
    }

    ThemedMarkdownWidget MarkdownBlock > .code_inline {
        color: $axis-markdown-inline-code !important;
        background: transparent !important;
    }

    ThemedMarkdownWidget MarkdownBullet {
        color: $axis-markdown-bullet;
    }

    ThemedMarkdownWidget MarkdownFence {
        background: $axis-markdown-code-block-background;
    }

    ThemedMarkdownWidget MarkdownTableContent {
        keyline: thin $axis-markdown-table-border;
    }

    ThemedMarkdownWidget MarkdownTableContent > .header {
        color: $axis-markdown-table-header;
        text-style: bold;
    }
    """

    def __init__(
        self,
        markdown: str | None = None,
        *,
        theme: TuiTheme,
        classes: str | None = None,
    ) -> None:
        self.axis_link_style = theme.markdown_link
        super().__init__(markdown, classes=classes)


class TranscriptMessageWidget(Horizontal):
    """One selectable message with a non-selectable colored gutter."""

    DEFAULT_CSS = """
    TranscriptMessageWidget {
        width: 1fr;
        height: auto;
        margin: 1 1 2 0;
    }

    TranscriptMessageWidget > .transcript-message-gutter {
        width: 1;
        height: auto;
    }

    TranscriptMessageWidget > .transcript-message-body {
        width: 1fr;
        height: auto;
        padding: 0 1 0 1;
    }

    TranscriptMessageWidget > .transcript-markdown-body > MarkdownParagraph {
        margin: 0 0 1 0;
    }
    """

    def __init__(
        self,
        item: ChatItem,
        *,
        theme: TuiTheme,
        show_tool_results: bool,
    ) -> None:
        self.item = item
        self.selection_text = transcript_item_selection_text(
            item,
            show_tool_results=show_tool_results,
        )
        self._theme = theme
        self._role_style = _role_style(item, theme)
        super().__init__(classes="transcript-message")

    def compose(self) -> Any:
        gutter = NonSelectableStatic("▌", classes="transcript-message-gutter")
        gutter.styles.color = self._role_style.border
        yield gutter
        yield self._body_widget()

    def _body_widget(self) -> Static | ThemedMarkdownWidget:
        if self.item.role in {"user", "tool", "skill", "error"}:
            return Static(
                _plain_body_renderable(
                    self.item,
                    self.selection_text,
                    style=self._role_style.body,
                    theme=self._theme,
                ),
                expand=True,
                shrink=True,
                markup=False,
                classes="transcript-message-body transcript-plain-body",
            )

        body = ThemedMarkdownWidget(
            self.selection_text,
            theme=self._theme,
            classes="transcript-message-body transcript-markdown-body",
        )
        foreground, background = _style_colors(self._role_style.body)
        if foreground:
            body.styles.color = foreground
        if background:
            body.styles.background = background
        return body

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Return selected source text rather than rendered Markdown markup."""
        selected = _extract_text_selection(self.selection_text, selection)
        return (selected, "\n") if selected else None


class StreamingTranscriptMessageWidget(ThemedMarkdownWidget):
    """One live assistant or thinking block updated without rebuilding history."""

    DEFAULT_CSS = """
    StreamingTranscriptMessageWidget {
        width: 1fr;
        height: auto;
        margin: 1 1 2 1;
        padding: 0 1 0 0;
    }

    StreamingTranscriptMessageWidget > MarkdownParagraph {
        margin: 0 0 1 0;
    }
    """

    def __init__(self, item: ChatItem, *, theme: TuiTheme) -> None:
        if item.role not in {"assistant", "thinking"}:
            raise ValueError("Streaming transcript widgets require assistant or thinking items")
        self.item = item
        self.selection_text = item.text
        super().__init__(item.text, theme=theme, classes="transcript-message")

    async def append_fragment(self, fragment: str) -> None:
        """Append one provider fragment using native Markdown rendering."""
        if not fragment:
            return
        self.item.text += fragment
        self.selection_text += fragment
        await self.update(self.item.text)

    async def replace_text(self, text: str) -> None:
        """Replace streamed fragments with the authoritative complete message."""
        self.item.text = text
        self.selection_text = text
        await self.update(text)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        selected = _extract_text_selection(self.selection_text, selection)
        return (selected, "\n") if selected else None


class TranscriptView(VerticalScroll):
    """Scrollable transcript backed by separate selectable message widgets."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        for legacy_option in ("wrap", "highlight", "markup"):
            kwargs.pop(legacy_option, None)
        min_width = kwargs.pop("min_width", None)
        super().__init__(*args, **kwargs)
        if min_width is not None:
            self.styles.min_width = min_width
        self._state: TuiState | None = None
        self._theme = AXIS_DARK_THEME
        self._active_assistant: StreamingTranscriptMessageWidget | None = None
        self._active_thinking: StreamingTranscriptMessageWidget | None = None
        self._hidden_thinking_visible = False

    def on_mount(self) -> None:
        self.anchor()

    def follow_output(self) -> None:
        """Resume automatic scrolling after explicit user submission."""
        self.anchor()

    def update_from_state(
        self,
        state: TuiState,
        *,
        theme: TuiTheme = AXIS_DARK_THEME,
    ) -> None:
        """Rebuild visible widgets from semantic state."""
        should_scroll = self.is_vertical_scroll_end or self.is_anchored
        self._state = state
        self._theme = theme
        self.remove_children(
            [
                child
                for child in self.children
                if isinstance(child, TranscriptMessageWidget | StreamingTranscriptMessageWidget)
            ]
        )
        self._active_assistant = None
        self._active_thinking = None
        self._hidden_thinking_visible = False
        hidden_placeholder = False
        for item in state.items:
            if item.role == "thinking" and not state.show_thinking:
                if not hidden_placeholder:
                    self.mount(
                        TranscriptMessageWidget(
                            ChatItem(
                                role="thinking",
                                text="Thinking… Press Ctrl+T to show thinking tokens.",
                            ),
                            theme=theme,
                            show_tool_results=state.show_tool_results,
                        )
                    )
                    hidden_placeholder = True
                continue
            hidden_placeholder = False
            self.mount(
                _transcript_widget(
                    item,
                    theme=theme,
                    show_tool_results=(state.show_tool_results or item.always_show_tool_result),
                )
            )
        if state.assistant_buffer:
            self.mount(
                TranscriptMessageWidget(
                    ChatItem(role="assistant", text=state.assistant_buffer),
                    theme=theme,
                    show_tool_results=state.show_tool_results,
                )
            )
        self.refresh(layout=True)
        if should_scroll:
            self.scroll_end(animate=False)

    async def append_item(
        self,
        item: ChatItem,
        *,
        theme: TuiTheme = AXIS_DARK_THEME,
        show_tool_results: bool = False,
        scroll_end: bool = False,
    ) -> TranscriptMessageWidget | StreamingTranscriptMessageWidget:
        """Mount one new item without rebuilding previous widgets."""
        self._theme = theme
        widget = _transcript_widget(
            item,
            theme=theme,
            show_tool_results=show_tool_results,
        )
        await self.mount(widget)
        self._active_assistant = None
        self._active_thinking = None
        self._hidden_thinking_visible = False
        if scroll_end:
            self.scroll_end(animate=False)
        return widget

    async def append_assistant_delta(
        self,
        delta: str,
        *,
        theme: TuiTheme = AXIS_DARK_THEME,
        scroll_end: bool = False,
    ) -> None:
        """Append text to one active assistant widget."""
        self._active_thinking = None
        self._hidden_thinking_visible = False
        if self._active_assistant is None:
            self._active_assistant = StreamingTranscriptMessageWidget(
                ChatItem(role="assistant", text=""),
                theme=theme,
            )
            await self.mount(self._active_assistant)
        await self._active_assistant.append_fragment(delta)
        if scroll_end:
            self.scroll_end(animate=False)

    async def append_thinking_delta(
        self,
        delta: str,
        *,
        theme: TuiTheme = AXIS_DARK_THEME,
        show_thinking: bool,
        scroll_end: bool = False,
    ) -> None:
        """Append reasoning or mount one collapsed placeholder."""
        if not show_thinking:
            if self._hidden_thinking_visible:
                return
            await self.append_item(
                ChatItem(
                    role="thinking",
                    text="Thinking… Press Ctrl+T to show thinking tokens.",
                ),
                theme=theme,
                scroll_end=scroll_end,
            )
            self._hidden_thinking_visible = True
            return
        self._hidden_thinking_visible = False
        if self._active_thinking is None:
            self._active_thinking = StreamingTranscriptMessageWidget(
                ChatItem(role="thinking", text=""),
                theme=theme,
            )
            await self.mount(self._active_thinking)
        await self._active_thinking.append_fragment(delta)
        if scroll_end:
            self.scroll_end(animate=False)

    async def finish_assistant_message(self, text: str | None = None) -> None:
        """Finalize the active assistant block using the complete message."""
        if self._active_assistant is None:
            if text:
                await self.append_item(ChatItem(role="assistant", text=text), theme=self._theme)
            return
        if text is not None:
            await self._active_assistant.replace_text(text)
        self._active_assistant = None

    @property
    def lines(self) -> tuple[TranscriptLine, ...]:
        """Return plain rendered lines for inspection and tests."""
        messages = [
            child
            for child in self.children
            if isinstance(child, TranscriptMessageWidget | StreamingTranscriptMessageWidget)
        ]
        return tuple(
            TranscriptLine(line)
            for message in messages
            for line in message.selection_text.splitlines()
        )


def transcript_item_selection_text(
    item: ChatItem,
    *,
    show_tool_results: bool = False,
) -> str:
    """Return selectable plain text after visibility rules."""
    return visible_chat_text(item, show_tool_results=show_tool_results)


def render_chat_item(
    item: ChatItem,
    *,
    theme: TuiTheme = AXIS_DARK_THEME,
    show_tool_results: bool = False,
) -> RenderableType:
    """Render one item for non-Textual inspection helpers."""
    role_style = _role_style(item, theme)
    text = visible_chat_text(item, show_tool_results=show_tool_results)
    body: RenderableType
    if item.role in {"assistant", "thinking", "status", "branch_summary", "compaction_summary"}:
        body = Markdown(text, style=role_style.body, code_theme=theme.syntax_theme)
    else:
        body = _plain_body_renderable(item, text, style=role_style.body, theme=theme)
    return Group(Text("▌", style=role_style.border), body)


def _transcript_widget(
    item: ChatItem,
    *,
    theme: TuiTheme,
    show_tool_results: bool,
) -> TranscriptMessageWidget | StreamingTranscriptMessageWidget:
    if item.role in {"assistant", "thinking"}:
        return StreamingTranscriptMessageWidget(item, theme=theme)
    return TranscriptMessageWidget(
        item,
        theme=theme,
        show_tool_results=show_tool_results,
    )


def _role_style(item: ChatItem, theme: TuiTheme) -> TuiRoleStyle:
    return theme.role_styles[item.role]


def _plain_body_renderable(
    item: ChatItem,
    text: str,
    *,
    style: str,
    theme: TuiTheme,
) -> RenderableType:
    marker = "\nPatch:\n"
    if item.role == "tool" and marker in text:
        before, patch = text.split(marker, 1)
        if patch.strip():
            return Group(
                Text(f"{before}{marker.rstrip()}", style=style, overflow="fold"),
                Syntax(
                    patch.rstrip("\n"),
                    "diff",
                    theme=theme.syntax_theme,
                    word_wrap=True,
                    background_color=theme.markdown_code_block_background,
                ),
            )
    return Text(text, style=style, overflow="fold", no_wrap=False)


def _style_colors(style: str) -> tuple[str | None, str | None]:
    parsed = Style.parse(style)
    foreground = parsed.color.name if parsed.color is not None else None
    background = parsed.bgcolor.name if parsed.bgcolor is not None else None
    return foreground, background


def _extract_text_selection(text: str, selection: Selection) -> str:
    lines = text.split("\n")
    if not lines:
        return ""
    start = _clip_offset(selection.start, lines, default=Offset(0, 0))
    end = _clip_offset(
        selection.end,
        lines,
        default=Offset(len(lines[-1]), len(lines) - 1),
    )
    if (start.y, start.x) > (end.y, end.x):
        start, end = end, start
    if start.y == end.y:
        return lines[start.y][start.x : end.x]
    selected = [lines[start.y][start.x :]]
    selected.extend(lines[start.y + 1 : end.y])
    selected.append(lines[end.y][: end.x])
    return "\n".join(selected)


def _clip_offset(offset: Offset | None, lines: list[str], *, default: Offset) -> Offset:
    if offset is None:
        return default
    row = min(max(offset.y, 0), len(lines) - 1)
    column = min(max(offset.x, 0), len(lines[row]))
    return Offset(column, row)
