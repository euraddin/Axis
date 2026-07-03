import asyncio
import gzip
from json import dumps

import pytest

from axis_coding.voice import (
    AsrTranscriptEvent,
    VoiceAsrError,
    VoiceInputConfig,
    VolcengineSeedAsrProvider,
    build_audio_request_frame,
    build_frame,
    build_full_client_request_frame,
    parse_response_frame,
)


class FakeWebSocket:
    def __init__(self, responses: list[bytes]) -> None:
        self.responses: asyncio.Queue[bytes] = asyncio.Queue()
        for response in responses:
            self.responses.put_nowait(response)
        self.sent: list[bytes] = []
        self.closed = False

    async def send(self, data: bytes) -> None:
        self.sent.append(data)

    async def recv(self) -> bytes:
        return await self.responses.get()

    async def close(self) -> None:
        self.closed = True


def test_volcengine_client_frames_have_sequence_and_gzip_payload() -> None:
    config = VoiceInputConfig()
    frame = build_full_client_request_frame(config, 1)
    assert frame[:4] == bytes((0x11, 0x11, 0x11, 0))
    assert int.from_bytes(frame[4:8], "big", signed=True) == 1
    size = int.from_bytes(frame[8:12], "big")
    payload = gzip.decompress(frame[12 : 12 + size]).decode()
    assert '"rate":16000' in payload
    assert '"language":"zh-CN"' in payload

    final = build_audio_request_frame(b"", -3)
    assert final[1] & 0x0F == 0x03
    assert int.from_bytes(final[4:8], "big", signed=True) == -3


def test_parse_volcengine_partial_final_and_error_frames() -> None:
    partial = build_frame(
        0x9,
        0x1,
        0x1,
        dumps({"code": 0, "result": {"text": "hello"}}).encode(),
        2,
    )
    final = build_frame(
        0x9,
        0x3,
        0x1,
        dumps({"code": 0, "result": {"text": "hello world"}}).encode(),
        -3,
    )
    assert parse_response_frame(partial) == AsrTranscriptEvent("partial", "hello")
    assert parse_response_frame(final) == AsrTranscriptEvent("final", "hello world")

    error_payload = gzip.compress(dumps({"message": "bad key"}).encode())
    error = b"".join(
        (
            bytes((0x11, 0xF0, 0x11, 0)),
            (401).to_bytes(4, "big", signed=True),
            len(error_payload).to_bytes(4, "big"),
            error_payload,
        )
    )
    assert parse_response_frame(error) == AsrTranscriptEvent("error", "bad key")


def test_parse_volcengine_rejects_truncated_frames() -> None:
    with pytest.raises(VoiceAsrError, match="too short"):
        parse_response_frame(b"123")


def test_volcengine_provider_streams_with_expected_auth_and_final_frame() -> None:
    async def scenario() -> None:
        ack = build_frame(0x9, 0x1, 0x1, dumps({"code": 0, "result": {}}).encode(), 1)
        partial = build_frame(
            0x9,
            0x1,
            0x1,
            dumps({"code": 0, "result": {"text": "正在"}}).encode(),
            2,
        )
        final = build_frame(
            0x9,
            0x3,
            0x1,
            dumps({"code": 0, "result": {"text": "正在测试"}}).encode(),
            -3,
        )
        websocket = FakeWebSocket([ack, partial, final])
        observed_headers: dict[str, str] = {}

        async def connector(uri: str, *, additional_headers: dict[str, str]) -> FakeWebSocket:
            assert uri.endswith("/bigmodel_async")
            observed_headers.update(additional_headers)
            return websocket

        provider = VolcengineSeedAsrProvider(
            VoiceInputConfig(), api_key="speech-key", connector=connector
        )
        await provider.connect()
        await provider.send_audio(b"pcm")
        await provider.finalize()
        events = [event async for event in provider.events()]
        await provider.cancel()

        assert events == [
            AsrTranscriptEvent("partial", "正在"),
            AsrTranscriptEvent("final", "正在测试"),
        ]
        assert observed_headers["X-Api-Key"] == "speech-key"
        assert observed_headers["X-Api-Resource-Id"] == "volc.seedasr.sauc.duration"
        assert observed_headers["X-Api-Connect-Id"].startswith("axis-")
        assert int.from_bytes(websocket.sent[-1][4:8], "big", signed=True) < 0
        assert websocket.closed

    asyncio.run(scenario())


def test_volcengine_provider_retries_before_streaming() -> None:
    async def scenario() -> int:
        ack = build_frame(0x9, 0x1, 0x1, dumps({"code": 0, "result": {}}).encode(), 1)
        attempts = 0

        async def connector(uri: str, **kwargs: object) -> FakeWebSocket:
            nonlocal attempts
            del uri, kwargs
            attempts += 1
            if attempts < 3:
                raise OSError("offline")
            return FakeWebSocket([ack])

        provider = VolcengineSeedAsrProvider(
            VoiceInputConfig(), api_key="speech-key", connector=connector
        )
        await provider.connect()
        await provider.cancel()
        return attempts

    assert asyncio.run(scenario()) == 3
