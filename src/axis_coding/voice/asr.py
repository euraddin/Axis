"""Streaming Volcengine Seed ASR 2.0 protocol adapter."""

from __future__ import annotations

import asyncio
import gzip
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from json import dumps, loads
from typing import Any, Literal, Protocol
from uuid import uuid4

from axis_coding.voice.config import VoiceInputConfig

FULL_CLIENT_REQUEST = 0x1
AUDIO_ONLY_REQUEST = 0x2
FULL_SERVER_RESPONSE = 0x9
ERROR_RESPONSE = 0xF
POS_SEQUENCE = 0x1
NEG_SEQUENCE = 0x2
NEG_WITH_SEQUENCE = 0x3
JSON_SERIALIZATION = 0x1
NO_SERIALIZATION = 0x0
GZIP_COMPRESSION = 0x1


class VoiceAsrError(RuntimeError):
    """The streaming ASR transport or protocol failed."""


@dataclass(frozen=True, slots=True)
class AsrTranscriptEvent:
    type: Literal["partial", "final", "error"]
    text: str


class StreamingAsrProvider(Protocol):
    async def connect(self) -> None: ...

    async def send_audio(self, chunk: bytes) -> None: ...

    def events(self) -> AsyncIterator[AsrTranscriptEvent]: ...

    async def finalize(self) -> None: ...

    async def cancel(self) -> None: ...


type WebSocketConnector = Callable[..., Awaitable[Any]]


class VolcengineSeedAsrProvider:
    """Translate raw PCM chunks to Volcengine's binary streaming protocol."""

    def __init__(
        self,
        config: VoiceInputConfig,
        *,
        api_key: str,
        connector: WebSocketConnector | None = None,
    ) -> None:
        if not api_key.strip():
            raise VoiceAsrError("Volcengine ASR API key is empty")
        self.config = config
        self.api_key = api_key.strip()
        self._connector = connector
        self._connection: Any | None = None
        self._sequence = 1
        self._sent_audio = False
        self._finalized = False

    async def connect(self) -> None:
        connector = self._connector or _default_connector
        headers = {
            "X-Api-Key": self.api_key,
            "X-Api-Resource-Id": self.config.resource_id,
            "X-Api-Connect-Id": f"axis-{uuid4().hex}",
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                connection = await asyncio.wait_for(
                    connector(self.config.endpoint, additional_headers=headers),
                    timeout=self.config.connect_timeout_seconds,
                )
                self._connection = connection
                self._sequence = 1
                self._sent_audio = False
                self._finalized = False
                await connection.send(build_full_client_request_frame(self.config, 1))
                initial = await asyncio.wait_for(
                    connection.recv(),
                    timeout=self.config.connect_timeout_seconds,
                )
                if isinstance(initial, str):
                    raise VoiceAsrError(initial)
                event = parse_response_frame(bytes(initial))
                if event is not None and event.type == "error":
                    raise VoiceAsrError(event.text)
                return
            except Exception as exc:
                last_error = exc
                await self._close_connection()
                if attempt < 2:
                    await asyncio.sleep(0.25 * (2**attempt))
        raise VoiceAsrError(f"Could not connect to Volcengine ASR: {last_error}")

    async def send_audio(self, chunk: bytes) -> None:
        if self._connection is None:
            raise VoiceAsrError("Volcengine ASR is not connected")
        if self._finalized:
            raise VoiceAsrError("Volcengine ASR is already finalized")
        if not chunk:
            return
        self._sequence += 1
        await self._connection.send(build_audio_request_frame(chunk, self._sequence))
        self._sent_audio = True

    async def finalize(self) -> None:
        if self._connection is None or self._finalized:
            return
        self._finalized = True
        if not self._sent_audio:
            raise VoiceAsrError("No microphone audio was captured")
        self._sequence += 1
        await self._connection.send(build_audio_request_frame(b"", -abs(self._sequence)))

    async def events(self) -> AsyncIterator[AsrTranscriptEvent]:
        if self._connection is None:
            raise VoiceAsrError("Volcengine ASR is not connected")
        while self._connection is not None:
            try:
                message = await self._connection.recv()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                yield AsrTranscriptEvent("error", str(exc))
                return
            if isinstance(message, str):
                yield AsrTranscriptEvent("error", message)
                return
            event = parse_response_frame(bytes(message))
            if event is None:
                continue
            yield event
            if event.type in {"final", "error"}:
                return

    async def cancel(self) -> None:
        await self._close_connection()

    async def _close_connection(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            with suppress(Exception):
                await connection.close()


async def _default_connector(
    uri: str,
    *,
    additional_headers: Mapping[str, str],
) -> Any:
    from websockets.asyncio.client import connect

    return await connect(
        uri, additional_headers=additional_headers, proxy=None
    )  # bypass SOCKS / HTTP proxy for the domestic ASR endpoint


def build_full_client_request_frame(config: VoiceInputConfig, sequence: int) -> bytes:
    payload = dumps(
        {
            "user": {"uid": "axis"},
            "audio": {
                "format": "pcm",
                "codec": "raw",
                "rate": config.sample_rate,
                "bits": config.sample_width_bits,
                "channel": config.channels,
                "language": config.language,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_ddc": False,
                "enable_punc": True,
                "show_utterances": True,
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return build_frame(
        FULL_CLIENT_REQUEST,
        POS_SEQUENCE,
        JSON_SERIALIZATION,
        payload,
        sequence,
    )


def build_audio_request_frame(chunk: bytes, sequence: int) -> bytes:
    flags = NEG_WITH_SEQUENCE if sequence < 0 else POS_SEQUENCE
    return build_frame(AUDIO_ONLY_REQUEST, flags, NO_SERIALIZATION, chunk, sequence)


def build_frame(
    message_type: int,
    flags: int,
    serialization: int,
    payload: bytes,
    sequence: int,
) -> bytes:
    compressed = gzip.compress(payload)
    header = bytes(
        (
            0x11,
            (message_type << 4) | flags,
            (serialization << 4) | GZIP_COMPRESSION,
            0,
        )
    )
    return b"".join(
        (
            header,
            sequence.to_bytes(4, "big", signed=True),
            len(compressed).to_bytes(4, "big"),
            compressed,
        )
    )


def parse_response_frame(frame: bytes) -> AsrTranscriptEvent | None:
    if len(frame) < 8:
        raise VoiceAsrError("Volcengine ASR frame is too short")
    header_size = (frame[0] & 0x0F) * 4
    if header_size < 4 or len(frame) < header_size + 4:
        raise VoiceAsrError("Volcengine ASR frame header is incomplete")
    message_type = frame[1] >> 4
    flags = frame[1] & 0x0F
    serialization = frame[2] >> 4
    compression = frame[2] & 0x0F
    offset = header_size
    sequence: int | None = None
    if flags & POS_SEQUENCE:
        if len(frame) < offset + 4:
            raise VoiceAsrError("Volcengine ASR frame sequence is incomplete")
        sequence = int.from_bytes(frame[offset : offset + 4], "big", signed=True)
        offset += 4
    is_last = bool(flags & NEG_SEQUENCE)
    if flags & 0x04:
        if len(frame) < offset + 4:
            raise VoiceAsrError("Volcengine ASR frame event is incomplete")
        offset += 4
    error_code: int | None = None
    if message_type == ERROR_RESPONSE:
        if len(frame) < offset + 8:
            raise VoiceAsrError("Volcengine ASR error frame is incomplete")
        error_code = int.from_bytes(frame[offset : offset + 4], "big", signed=True)
        offset += 4
    if len(frame) < offset + 4:
        raise VoiceAsrError("Volcengine ASR frame payload size is incomplete")
    payload_size = int.from_bytes(frame[offset : offset + 4], "big")
    offset += 4
    if len(frame) < offset + payload_size:
        raise VoiceAsrError("Volcengine ASR frame payload is incomplete")
    payload = frame[offset : offset + payload_size]
    if compression == GZIP_COMPRESSION:
        try:
            payload = gzip.decompress(payload)
        except OSError as exc:
            raise VoiceAsrError("Volcengine ASR payload gzip is invalid") from exc
    if serialization != JSON_SERIALIZATION:
        if message_type == ERROR_RESPONSE:
            return AsrTranscriptEvent("error", f"Volcengine ASR error {error_code}")
        return None
    try:
        value = loads(payload)
    except (UnicodeError, ValueError) as exc:
        raise VoiceAsrError("Volcengine ASR response JSON is invalid") from exc
    if not isinstance(value, Mapping):
        raise VoiceAsrError("Volcengine ASR response must be an object")
    code = value.get("code")
    if message_type == ERROR_RESPONSE or (isinstance(code, int) and code not in {0, 20_000_000}):
        return AsrTranscriptEvent("error", _response_message(value, error_code))
    if message_type != FULL_SERVER_RESPONSE:
        return None
    text = _extract_transcript(value)
    if not text:
        return None
    final = is_last or (sequence is not None and sequence < 0)
    return AsrTranscriptEvent("final" if final else "partial", text)


def _extract_transcript(value: Mapping[str, Any]) -> str:
    result = value.get("result")
    if not isinstance(result, Mapping):
        return ""
    text = result.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    utterances = result.get("utterances")
    if isinstance(utterances, list):
        for item in reversed(utterances):
            if isinstance(item, Mapping):
                utterance = item.get("text")
                if isinstance(utterance, str) and utterance.strip():
                    return utterance.strip()
    return ""


def _response_message(value: Mapping[str, Any], error_code: int | None) -> str:
    for key in ("message", "error"):
        message = value.get(key)
        if isinstance(message, str) and message:
            return message
        if isinstance(message, Mapping):
            nested = message.get("message")
            if isinstance(nested, str) and nested:
                return nested
    return f"Volcengine ASR error {error_code}" if error_code is not None else "Unknown ASR error"
