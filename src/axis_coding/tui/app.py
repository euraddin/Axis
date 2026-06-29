"""Basic Textual application for one Axis coding session."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import ClassVar, Protocol

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Footer, Header, Input, Static

from axis_agent import AgentEndEvent, AgentEvent, ErrorEvent
from axis_coding.tui.adapter import TuiEventAdapter
from axis_coding.tui.rendering import format_tui_status, render_tui_state
from axis_coding.tui.state import TuiNoticeItem, TuiState


class TuiSession(Protocol):
    """Narrow session interface required by the basic Textual frontend."""

    @property
    def cwd(self) -> Path:
        """Return the session working directory."""
        ...

    @property
    def model(self) -> str:
        """Return the active model name."""
        ...

    @property
    def is_running(self) -> bool:
        """Return whether an agent run is active."""
        ...

    def prompt(self, content: str) -> AsyncIterator[AgentEvent]:
        """Start one prompt event stream."""
        ...

    def cancel(self) -> None:
        """Request cancellation of the active run."""
        ...


class AxisTuiApp(App[None]):
    """Minimal interactive frontend over a CodingSession-like object."""

    TITLE = "Axis"
    SUB_TITLE = "Personal coding agent"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel_run", "Cancel", priority=True),
        Binding("ctrl+d", "exit_app", "Quit", priority=True),
    ]
    CSS = """
    Screen {
        background: #080b10;
        color: #d8dee9;
    }

    #main {
        height: 1fr;
    }

    #transcript-scroll {
        height: 1fr;
        padding: 1 2;
        border-bottom: solid #263241;
    }

    #transcript {
        width: 100%;
        height: auto;
    }

    #status {
        height: 1;
        padding: 0 2;
        color: #8fa1b5;
        background: #111720;
    }

    #prompt {
        dock: bottom;
        margin: 0 1 1 1;
        border: tall #4f8f86;
        background: #111720;
    }

    #prompt:focus {
        border: tall #70b7ad;
    }
    """

    def __init__(self, session: TuiSession) -> None:
        super().__init__()
        self.session = session
        self.state = TuiState()
        self.adapter = TuiEventAdapter(self.state)
        self._cancel_requested = False

    def compose(self) -> ComposeResult:
        """Create the compact single-session layout."""
        yield Header()
        with Vertical(id="main"):
            with VerticalScroll(id="transcript-scroll"):
                yield Static(id="transcript")
            yield Static(id="status")
            yield Input(
                placeholder="Ask Axis…  Enter submits · Esc cancels · Ctrl+D quits",
                id="prompt",
            )
        yield Footer()

    def on_mount(self) -> None:
        """Render initial state and focus the prompt."""
        self._render_state()
        self.query_one("#prompt", Input).focus()

    def on_unmount(self) -> None:
        """Do not leave an active provider/tool run behind the UI."""
        if self.session.is_running:
            self.session.cancel()

    @on(Input.Submitted)
    def submit_prompt(self, event: Input.Submitted) -> None:
        """Start a prompt in a Textual worker so the UI remains responsive."""
        content = event.value.strip()
        if not content:
            event.input.clear()
            return
        if self.session.is_running or self.state.running:
            return

        event.input.clear()
        event.input.disabled = True
        self._cancel_requested = False
        self.run_worker(
            self._run_prompt(content),
            name="axis-agent-run",
            group="agent",
            exclusive=True,
            exit_on_error=False,
        )

    async def _run_prompt(self, content: str) -> None:
        try:
            async for event in self.session.prompt(content):
                self.adapter.apply(event)
                if self.state.cancelled:
                    self._cancel_requested = False
                self._render_state()
        except Exception as exc:  # UI boundary: surface unexpected session failures
            self.adapter.apply(ErrorEvent(message=str(exc), recoverable=False))
            self.adapter.apply(AgentEndEvent())
            self._render_state()
        finally:
            try:
                prompt = self.query_one("#prompt", Input)
            except NoMatches:
                pass
            else:
                prompt.disabled = False
                prompt.focus()
                self._render_state()

    def action_cancel_run(self) -> None:
        """Request cooperative cancellation without killing the UI worker."""
        if not (self.session.is_running or self.state.running):
            return
        if not self._cancel_requested:
            self.state.items.append(TuiNoticeItem(level="status", text="Cancellation requested…"))
        self._cancel_requested = True
        self.session.cancel()
        self._render_state()

    def action_exit_app(self) -> None:
        """Cancel active work and exit the application."""
        if self.session.is_running or self.state.running:
            self.session.cancel()
        self.exit()

    def _render_state(self) -> None:
        try:
            transcript = self.query_one("#transcript", Static)
        except NoMatches:
            return
        transcript.update(render_tui_state(self.state))
        status = format_tui_status(
            self.state,
            model=self.session.model,
            cwd=str(self.session.cwd),
        )
        if self._cancel_requested and self.state.running:
            status = f"{status} · Cancelling"
        self.query_one("#status", Static).update(status)
        self.query_one("#transcript-scroll", VerticalScroll).scroll_end(animate=False)


async def run_tui_app(session: TuiSession) -> None:
    """Run the basic Axis TUI in the caller's current async loop."""
    await AxisTuiApp(session).run_async()
