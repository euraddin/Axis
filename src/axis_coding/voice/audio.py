"""Raw microphone capture kept behind an injectable boundary."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol

from axis_coding.voice.config import VoiceInputConfig


class VoiceAudioError(RuntimeError):
    """Microphone capture failed or could not keep up."""


@dataclass(frozen=True, slots=True)
class AudioInputDevice:
    index: int
    name: str
    channels: int
    default: bool = False


class AudioSource(Protocol):
    async def start(self) -> None: ...

    def chunks(self) -> AsyncIterator[bytes]: ...

    async def stop(self) -> None: ...


class SoundDeviceAudioSource:
    """Capture mono int16 PCM without requiring NumPy."""

    def __init__(self, config: VoiceInputConfig) -> None:
        self.config = config
        self._queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue(
            maxsize=config.audio_queue_chunks
        )
        self._stream: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped = False

    async def start(self) -> None:
        try:
            sounddevice: Any = import_module("sounddevice")
        except ImportError as exc:
            raise VoiceAudioError("Voice input requires the sounddevice package") from exc
        self._loop = asyncio.get_running_loop()
        self._stopped = False

        def callback(indata: Any, frames: int, time: object, status: object) -> None:
            del frames, time
            if self._loop is None or self._stopped:
                return
            if status:
                self._loop.call_soon_threadsafe(
                    self._put_error,
                    VoiceAudioError(f"Microphone stream error: {status}"),
                )
                return
            self._loop.call_soon_threadsafe(self._put_chunk, bytes(indata))

        try:
            self._stream = sounddevice.RawInputStream(
                samplerate=self.config.sample_rate,
                blocksize=self.config.blocksize,
                device=self.config.input_device,
                channels=self.config.channels,
                dtype="int16",
                callback=callback,
            )
            await asyncio.to_thread(self._stream.start)
        except Exception as exc:
            self._stream = None
            raise VoiceAudioError(
                "Could not open the microphone. On macOS, allow microphone access for your "
                "terminal in System Settings > Privacy & Security > Microphone. "
                f"Details: {exc}"
            ) from exc

    async def chunks(self) -> AsyncIterator[bytes]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                await asyncio.to_thread(stream.stop)
            finally:
                await asyncio.to_thread(stream.close)
        self._put_terminal()

    def _put_chunk(self, chunk: bytes) -> None:
        if self._stopped:
            return
        if self._queue.full():
            self._put_error(VoiceAudioError("Microphone audio queue overflowed"))
            return
        self._queue.put_nowait(chunk)

    def _put_error(self, error: Exception) -> None:
        if self._stopped:
            return
        while self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(error)

    def _put_terminal(self) -> None:
        while self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(None)


def list_audio_input_devices() -> tuple[AudioInputDevice, ...]:
    """Return available input devices without opening a stream."""
    try:
        sounddevice: Any = import_module("sounddevice")
    except ImportError as exc:
        raise VoiceAudioError("Voice input requires the sounddevice package") from exc
    try:
        devices = sounddevice.query_devices()
        default_input = sounddevice.default.device[0]
    except Exception as exc:
        raise VoiceAudioError(f"Could not query microphone devices: {exc}") from exc
    result: list[AudioInputDevice] = []
    for index, raw in enumerate(devices):
        channels = int(raw.get("max_input_channels", 0))
        if channels <= 0:
            continue
        result.append(
            AudioInputDevice(
                index=index,
                name=str(raw.get("name", f"Input {index}")),
                channels=channels,
                default=index == default_input,
            )
        )
    return tuple(result)
