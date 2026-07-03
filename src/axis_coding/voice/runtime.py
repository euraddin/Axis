"""Application composition helpers for real voice input."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from os import environ

from axis_coding.context_window import RequestContextBreakdown, RequestContextPart
from axis_coding.credentials import FileCredentialStore, credentials_path
from axis_coding.paths import AxisPaths
from axis_coding.provider_config import load_provider_settings
from axis_coding.voice.asr import VolcengineSeedAsrProvider
from axis_coding.voice.audio import SoundDeviceAudioSource
from axis_coding.voice.config import (
    VOLCENGINE_ASR_API_KEY_ENV,
    VOLCENGINE_ASR_CREDENTIAL_NAME,
    VoiceInputConfig,
    load_voice_config,
)
from axis_coding.voice.context import VoiceContextSnapshot
from axis_coding.voice.controller import VoiceEventListener, VoiceInputController, VoiceInputEvent
from axis_coding.voice.polisher import (
    VoicePolisher,
    VoicePolishError,
    VoicePolishResult,
    create_deepseek_voice_polisher,
)


def resolve_voice_api_key(
    paths: AxisPaths | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve a stored ASR key before the environment fallback."""
    values = environ if environment is None else environment
    store = FileCredentialStore(credentials_path(paths))
    return store.get(VOLCENGINE_ASR_CREDENTIAL_NAME) or values.get(VOLCENGINE_ASR_API_KEY_ENV)


class _UnavailablePolisher:
    def __init__(self, message: str) -> None:
        self.message = message

    async def polish(
        self,
        raw_text: str,
        context: VoiceContextSnapshot,
    ) -> VoicePolishResult:
        del raw_text, context
        raise VoicePolishError(self.message)


class _RawPolisher:
    async def polish(
        self,
        raw_text: str,
        context: VoiceContextSnapshot,
    ) -> VoicePolishResult:
        del context
        return VoicePolishResult(
            raw_text,
            RequestContextBreakdown("Voice setup test", (RequestContextPart("raw ASR", 0),)),
        )


def create_voice_input_controller(
    *,
    paths: AxisPaths | None = None,
    listener: VoiceEventListener | None = None,
) -> VoiceInputController:
    """Compose fresh per-recording ASR, microphone, and polishing resources."""
    runtime_paths = paths or AxisPaths()
    config = load_voice_config(runtime_paths)
    api_key = resolve_voice_api_key(runtime_paths)
    if not api_key:
        raise RuntimeError("Voice input is not configured; run /voice setup")
    credential_store = FileCredentialStore(credentials_path(runtime_paths))
    try:
        polisher: VoicePolisher = create_deepseek_voice_polisher(
            load_provider_settings(runtime_paths), credential_store
        )
    except Exception as exc:
        polisher = _UnavailablePolisher(str(exc))
    return VoiceInputController(
        config=config,
        asr=VolcengineSeedAsrProvider(config, api_key=api_key),
        audio=SoundDeviceAudioSource(config),
        polisher=polisher,
        listener=listener,
    )


async def test_voice_configuration(
    config: VoiceInputConfig,
    api_key: str,
    *,
    listener: VoiceEventListener | None = None,
) -> str:
    """Record three seconds through the real ASR without invoking DeepSeek."""
    test_config = replace(config, max_recording_seconds=3.0)
    loop = asyncio.get_running_loop()
    completed: asyncio.Future[str] = loop.create_future()

    def receive(event: VoiceInputEvent) -> None:
        if listener is not None:
            listener(event)
        if event.type == "completed" and not completed.done():
            completed.set_result(event.raw_text)
        elif event.type == "error" and not completed.done():
            completed.set_exception(RuntimeError(event.message))

    controller = VoiceInputController(
        config=test_config,
        asr=VolcengineSeedAsrProvider(test_config, api_key=api_key),
        audio=SoundDeviceAudioSource(test_config),
        polisher=_RawPolisher(),
        listener=receive,
    )
    await controller.start(VoiceContextSnapshot)
    try:
        return await asyncio.wait_for(completed, timeout=20)
    finally:
        await controller.aclose()
