"""Lifecycle owner for recording, streaming ASR, and one-shot polishing."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from axis_coding.context_window import RequestContextBreakdown
from axis_coding.voice.asr import AsrTranscriptEvent, StreamingAsrProvider
from axis_coding.voice.audio import AudioSource
from axis_coding.voice.config import VoiceInputConfig
from axis_coding.voice.context import VoiceContextSnapshot
from axis_coding.voice.polisher import VoicePolisher, estimate_voice_polish_breakdown

type VoiceState = Literal[
    "idle",
    "connecting",
    "recording",
    "finalizing",
    "polishing",
    "completed",
    "error",
    "cancelled",
]


@dataclass(frozen=True, slots=True)
class VoiceInputEvent:
    type: VoiceState | Literal["partial"]
    text: str = ""
    raw_text: str = ""
    message: str = ""
    used_fallback: bool = False
    breakdown: RequestContextBreakdown | None = None


type VoiceEventListener = Callable[[VoiceInputEvent], None]
type VoiceContextProvider = Callable[[], VoiceContextSnapshot]


class VoiceInputController:
    """Coordinate voice resources without owning any TUI widget."""

    def __init__(
        self,
        *,
        config: VoiceInputConfig,
        asr: StreamingAsrProvider,
        audio: AudioSource,
        polisher: VoicePolisher,
        listener: VoiceEventListener | None = None,
    ) -> None:
        self.config = config
        self.asr = asr
        self.audio = audio
        self.polisher = polisher
        self.listener = listener
        self.state: VoiceState = "idle"
        self._sender_task: asyncio.Task[None] | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._deadline_task: asyncio.Task[None] | None = None
        self._final_event = asyncio.Event()
        self._final_text = ""
        self._failure: Exception | None = None
        self._context_provider: VoiceContextProvider | None = None
        self._operation_lock = asyncio.Lock()

    @property
    def active(self) -> bool:
        return self.state in {"connecting", "recording", "finalizing", "polishing"}

    async def start(self, context_provider: VoiceContextProvider) -> None:
        async with self._operation_lock:
            if self.active:
                raise RuntimeError("Voice input is already active")
            self._reset(context_provider)
            self._set_state("connecting")
            try:
                await self.asr.connect()
                await self.audio.start()
            except Exception as exc:
                await self._cleanup()
                self._set_error(exc)
                return
            self._sender_task = asyncio.create_task(self._send_audio(), name="axis-voice-audio")
            self._receiver_task = asyncio.create_task(
                self._receive_transcripts(), name="axis-voice-asr"
            )
            self._deadline_task = asyncio.create_task(self._auto_stop(), name="axis-voice-deadline")
            self._set_state("recording")

    async def stop(self) -> None:
        async with self._operation_lock:
            if self.state != "recording":
                return
            self._set_state("finalizing")
            if self._deadline_task is not None:
                if self._deadline_task is not asyncio.current_task():
                    self._deadline_task.cancel()
                self._deadline_task = None
            try:
                await self.audio.stop()
                if self._sender_task is not None:
                    await self._sender_task
                await self.asr.finalize()
                await asyncio.wait_for(
                    self._final_event.wait(),
                    timeout=self.config.final_timeout_seconds,
                )
                if self._failure is not None:
                    raise self._failure
                raw_text = self._final_text.strip()
                if not raw_text:
                    raise RuntimeError("ASR returned no speech")
                if self._context_provider is None:
                    raise RuntimeError("Voice context is unavailable")
                context = self._context_provider()
                self._set_state("polishing")
                used_fallback = False
                breakdown: RequestContextBreakdown | None = estimate_voice_polish_breakdown(
                    raw_text, context
                )
                try:
                    result = await self.polisher.polish(raw_text, context)
                    text = result.text
                    breakdown = result.breakdown
                except Exception as exc:
                    text = raw_text
                    used_fallback = True
                    self._emit(
                        VoiceInputEvent(
                            "partial",
                            text=raw_text,
                            message=f"Polishing failed; using raw transcript: {exc}",
                        )
                    )
                self.state = "completed"
                self._emit(
                    VoiceInputEvent(
                        "completed",
                        text=text,
                        raw_text=raw_text,
                        used_fallback=used_fallback,
                        breakdown=breakdown,
                    )
                )
            except Exception as exc:
                self._set_error(exc)
            finally:
                await self._cleanup()

    async def cancel(self) -> None:
        async with self._operation_lock:
            if not self.active:
                return
            await self._cleanup()
            self.state = "cancelled"
            self._emit(VoiceInputEvent("cancelled", message="Voice input cancelled"))

    async def aclose(self) -> None:
        if self.active:
            await self.cancel()
        close = getattr(self.polisher, "aclose", None)
        if callable(close):
            await close()

    async def _send_audio(self) -> None:
        try:
            async for chunk in self.audio.chunks():
                await self.asr.send_audio(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failure = exc
            self._final_event.set()
            asyncio.create_task(self._fail_from_background(exc))

    async def _receive_transcripts(self) -> None:
        try:
            async for event in self.asr.events():
                self._accept_asr_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failure = exc
            self._final_event.set()
            asyncio.create_task(self._fail_from_background(exc))

    def _accept_asr_event(self, event: AsrTranscriptEvent) -> None:
        if event.type == "partial":
            self._emit(VoiceInputEvent("partial", text=event.text))
        elif event.type == "final":
            self._final_text = event.text
            self._emit(VoiceInputEvent("partial", text=event.text))
            self._final_event.set()
        else:
            self._failure = RuntimeError(event.text)
            self._final_event.set()
            asyncio.create_task(self._fail_from_background(self._failure))

    async def _fail_from_background(self, error: Exception) -> None:
        async with self._operation_lock:
            if self.state not in {"connecting", "recording"}:
                return
            await self._cleanup()
            self._set_error(error)

    async def _auto_stop(self) -> None:
        try:
            await asyncio.sleep(self.config.max_recording_seconds)
            await self.stop()
        except asyncio.CancelledError:
            return

    async def _cleanup(self) -> None:
        current = asyncio.current_task()
        for task in (self._deadline_task, self._sender_task, self._receiver_task):
            if task is not None and task is not current and not task.done():
                task.cancel()
        tasks = [
            task
            for task in (self._deadline_task, self._sender_task, self._receiver_task)
            if task is not None and task is not current
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._deadline_task = None
        self._sender_task = None
        self._receiver_task = None
        try:
            await self.audio.stop()
        finally:
            await self.asr.cancel()

    def _reset(self, context_provider: VoiceContextProvider) -> None:
        self._context_provider = context_provider
        self._final_event = asyncio.Event()
        self._final_text = ""
        self._failure = None

    def _set_state(self, state: VoiceState) -> None:
        self.state = state
        self._emit(VoiceInputEvent(state))

    def _set_error(self, error: Exception) -> None:
        self.state = "error"
        self._emit(VoiceInputEvent("error", message=str(error)))

    def _emit(self, event: VoiceInputEvent) -> None:
        if self.listener is not None:
            self.listener(event)
