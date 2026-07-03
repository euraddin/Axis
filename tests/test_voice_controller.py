import asyncio
from collections.abc import AsyncIterator

from axis_coding.context_window import RequestContextBreakdown, RequestContextPart
from axis_coding.voice import (
    AsrTranscriptEvent,
    VoiceContextSnapshot,
    VoiceInputConfig,
    VoiceInputController,
    VoiceInputEvent,
    VoicePolishResult,
)


class FakeAudio:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def chunks(self) -> AsyncIterator[bytes]:
        while (chunk := await self.queue.get()) is not None:
            yield chunk

    async def stop(self) -> None:
        if not self.stopped:
            self.stopped = True
            await self.queue.put(None)


class FakeAsr:
    def __init__(self, final: str = "raw words") -> None:
        self.events_queue: asyncio.Queue[AsrTranscriptEvent | None] = asyncio.Queue()
        self.final = final
        self.audio: list[bytes] = []
        self.cancelled = False

    async def connect(self) -> None:
        return None

    async def send_audio(self, chunk: bytes) -> None:
        self.audio.append(chunk)

    async def events(self) -> AsyncIterator[AsrTranscriptEvent]:
        while (event := await self.events_queue.get()) is not None:
            yield event

    async def finalize(self) -> None:
        await self.events_queue.put(AsrTranscriptEvent("final", self.final))

    async def cancel(self) -> None:
        self.cancelled = True
        await self.events_queue.put(None)


class FakePolisher:
    async def polish(self, raw_text: str, context: VoiceContextSnapshot) -> VoicePolishResult:
        assert context.coding_metadata == "cwd: /repo"
        return VoicePolishResult(
            raw_text.upper(),
            RequestContextBreakdown("Voice polish", (RequestContextPart("raw ASR", 2),)),
        )


class FailedPolisher:
    async def polish(self, raw_text: str, context: VoiceContextSnapshot) -> VoicePolishResult:
        del raw_text, context
        raise RuntimeError("offline")


def test_voice_controller_streams_polishes_and_cleans_up() -> None:
    async def scenario() -> list[VoiceInputEvent]:
        events: list[VoiceInputEvent] = []
        audio = FakeAudio()
        asr = FakeAsr()
        controller = VoiceInputController(
            config=VoiceInputConfig(),
            asr=asr,
            audio=audio,
            polisher=FakePolisher(),
            listener=events.append,
        )
        await controller.start(lambda: VoiceContextSnapshot(coding_metadata="cwd: /repo"))
        await audio.queue.put(b"pcm")
        await asr.events_queue.put(AsrTranscriptEvent("partial", "raw"))
        await asyncio.sleep(0)
        await controller.stop()
        assert asr.audio == [b"pcm"]
        assert asr.cancelled
        return events

    events = asyncio.run(scenario())
    assert [event.type for event in events] == [
        "connecting",
        "recording",
        "partial",
        "finalizing",
        "partial",
        "polishing",
        "completed",
    ]
    assert events[-1].text == "RAW WORDS"
    assert events[-1].used_fallback is False


def test_voice_controller_uses_raw_transcript_when_polishing_fails() -> None:
    async def scenario() -> VoiceInputEvent:
        events: list[VoiceInputEvent] = []
        controller = VoiceInputController(
            config=VoiceInputConfig(),
            asr=FakeAsr("keep this"),
            audio=FakeAudio(),
            polisher=FailedPolisher(),
            listener=events.append,
        )
        await controller.start(VoiceContextSnapshot)
        await controller.stop()
        return events[-1]

    completed = asyncio.run(scenario())
    assert completed.type == "completed"
    assert completed.text == "keep this"
    assert completed.used_fallback is True


def test_voice_controller_auto_stops_at_maximum_duration() -> None:
    async def scenario() -> list[VoiceInputEvent]:
        events: list[VoiceInputEvent] = []
        controller = VoiceInputController(
            config=VoiceInputConfig(max_recording_seconds=0.01),
            asr=FakeAsr("automatic"),
            audio=FakeAudio(),
            polisher=FakePolisher(),
            listener=events.append,
        )
        await controller.start(lambda: VoiceContextSnapshot(coding_metadata="cwd: /repo"))
        for _ in range(100):
            if any(event.type == "completed" for event in events):
                break
            await asyncio.sleep(0.01)
        return events

    events = asyncio.run(scenario())
    assert any(event.type == "completed" and event.text == "AUTOMATIC" for event in events)
