"""Context-aware voice input for Axis's coding frontend."""

from axis_coding.voice.asr import (
    AsrTranscriptEvent,
    StreamingAsrProvider,
    VoiceAsrError,
    VolcengineSeedAsrProvider,
    build_audio_request_frame,
    build_frame,
    build_full_client_request_frame,
    parse_response_frame,
)
from axis_coding.voice.audio import (
    AudioInputDevice,
    AudioSource,
    SoundDeviceAudioSource,
    VoiceAudioError,
    list_audio_input_devices,
)
from axis_coding.voice.config import (
    DEFAULT_VOLCENGINE_ASR_ENDPOINT,
    DEFAULT_VOLCENGINE_RESOURCE_ID,
    VOLCENGINE_ASR_API_KEY_ENV,
    VOLCENGINE_ASR_CREDENTIAL_NAME,
    VoiceConfigError,
    VoiceInputConfig,
    load_voice_config,
    save_voice_config,
    voice_config_from_json,
    voice_settings_path,
)
from axis_coding.voice.context import VoiceContextSnapshot, build_voice_context_snapshot
from axis_coding.voice.controller import VoiceInputController, VoiceInputEvent, VoiceState
from axis_coding.voice.polisher import (
    DeepSeekVoicePolisher,
    VoicePolisher,
    VoicePolishError,
    VoicePolishResult,
    create_deepseek_voice_polisher,
    estimate_voice_polish_breakdown,
)
from axis_coding.voice.runtime import (
    create_voice_input_controller,
    resolve_voice_api_key,
    test_voice_configuration,
)

__all__ = [
    "AudioInputDevice",
    "AudioSource",
    "AsrTranscriptEvent",
    "DEFAULT_VOLCENGINE_ASR_ENDPOINT",
    "DEFAULT_VOLCENGINE_RESOURCE_ID",
    "DeepSeekVoicePolisher",
    "SoundDeviceAudioSource",
    "StreamingAsrProvider",
    "VOLCENGINE_ASR_API_KEY_ENV",
    "VOLCENGINE_ASR_CREDENTIAL_NAME",
    "VoiceAsrError",
    "VoiceAudioError",
    "VoiceConfigError",
    "VoiceContextSnapshot",
    "VoiceInputConfig",
    "VoiceInputController",
    "VoiceInputEvent",
    "VoicePolishError",
    "VoicePolishResult",
    "VoicePolisher",
    "VoiceState",
    "VolcengineSeedAsrProvider",
    "build_audio_request_frame",
    "build_frame",
    "build_full_client_request_frame",
    "build_voice_context_snapshot",
    "create_deepseek_voice_polisher",
    "create_voice_input_controller",
    "estimate_voice_polish_breakdown",
    "list_audio_input_devices",
    "load_voice_config",
    "parse_response_frame",
    "resolve_voice_api_key",
    "save_voice_config",
    "test_voice_configuration",
    "voice_config_from_json",
    "voice_settings_path",
]
