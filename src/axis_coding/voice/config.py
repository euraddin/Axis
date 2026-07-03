"""Strict durable settings for Axis voice input."""

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from json import JSONDecodeError, dumps, loads
from pathlib import Path
from tempfile import NamedTemporaryFile

from axis_coding.paths import AxisPaths

DEFAULT_VOLCENGINE_ASR_ENDPOINT = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
DEFAULT_VOLCENGINE_RESOURCE_ID = "volc.seedasr.sauc.duration"
VOLCENGINE_ASR_CREDENTIAL_NAME = "volcengine-asr"
VOLCENGINE_ASR_API_KEY_ENV = "VOLCENGINE_ASR_API_KEY"


class VoiceConfigError(ValueError):
    """Voice settings are invalid or unavailable."""


@dataclass(frozen=True, slots=True)
class VoiceInputConfig:
    """Runtime and durable settings for one voice-input pipeline."""

    version: int = 1
    asr_provider: str = "volcengine-seed-asr-2"
    endpoint: str = DEFAULT_VOLCENGINE_ASR_ENDPOINT
    resource_id: str = DEFAULT_VOLCENGINE_RESOURCE_ID
    language: str = "zh-CN"
    input_device: int | str | None = None
    sample_rate: int = 16_000
    channels: int = 1
    sample_width_bits: int = 16
    chunk_ms: int = 100
    max_recording_seconds: float = 300.0
    connect_timeout_seconds: float = 5.0
    final_timeout_seconds: float = 8.0
    audio_queue_chunks: int = 50

    def __post_init__(self) -> None:
        if self.version != 1:
            raise VoiceConfigError(f"Unsupported voice config version: {self.version}")
        if self.asr_provider != "volcengine-seed-asr-2":
            raise VoiceConfigError(f"Unsupported ASR provider: {self.asr_provider}")
        for name in ("endpoint", "resource_id", "language"):
            if not getattr(self, name).strip():
                raise VoiceConfigError(f"Voice setting must not be empty: {name}")
        if not self.endpoint.startswith(("ws://", "wss://")):
            raise VoiceConfigError("Voice endpoint must use ws:// or wss://")
        if isinstance(self.input_device, str) and not self.input_device.strip():
            raise VoiceConfigError("Voice input_device must not be empty")
        if self.sample_rate <= 0 or self.channels != 1 or self.sample_width_bits != 16:
            raise VoiceConfigError("Voice audio must use positive-rate mono 16-bit PCM")
        if self.chunk_ms <= 0 or self.audio_queue_chunks <= 0:
            raise VoiceConfigError("Voice chunk settings must be greater than 0")
        if self.max_recording_seconds <= 0:
            raise VoiceConfigError("Voice max_recording_seconds must be greater than 0")
        if self.connect_timeout_seconds <= 0 or self.final_timeout_seconds <= 0:
            raise VoiceConfigError("Voice timeout settings must be greater than 0")

    @property
    def blocksize(self) -> int:
        return max(1, self.sample_rate * self.chunk_ms // 1_000)

    def to_json(self) -> dict[str, object]:
        return {
            "version": self.version,
            "asr_provider": self.asr_provider,
            "endpoint": self.endpoint,
            "resource_id": self.resource_id,
            "language": self.language,
            "input_device": self.input_device,
            "max_recording_seconds": self.max_recording_seconds,
        }


def voice_settings_path(paths: AxisPaths | None = None) -> Path:
    return (paths or AxisPaths()).home / "voice.json"


def load_voice_config(paths: AxisPaths | None = None) -> VoiceInputConfig:
    path = voice_settings_path(paths)
    if not path.exists():
        return VoiceInputConfig()
    try:
        raw = loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, JSONDecodeError) as exc:
        raise VoiceConfigError(f"Could not load voice settings from {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise VoiceConfigError("Voice settings must be a JSON object")
    return voice_config_from_json(raw)


def save_voice_config(
    config: VoiceInputConfig,
    paths: AxisPaths | None = None,
) -> Path:
    path = voice_settings_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp:
            temp_path = Path(temp.name)
            temp.write(dumps(config.to_json(), indent=2, sort_keys=True) + "\n")
            temp.flush()
        temp_path.replace(path)
    except OSError as exc:
        if temp_path is not None:
            with suppress(OSError):
                temp_path.unlink()
        raise VoiceConfigError(f"Could not save voice settings to {path}: {exc}") from exc
    return path


def voice_config_from_json(data: Mapping[str, object]) -> VoiceInputConfig:
    allowed = {
        "version",
        "asr_provider",
        "endpoint",
        "resource_id",
        "language",
        "input_device",
        "max_recording_seconds",
    }
    if unknown := set(data) - allowed:
        raise VoiceConfigError(f"Unknown voice settings field: {sorted(unknown)[0]}")
    defaults = VoiceInputConfig()
    input_device = data.get("input_device", defaults.input_device)
    if input_device is not None and not isinstance(input_device, (int, str)):
        raise VoiceConfigError("Voice input_device must be an integer, string, or null")
    return VoiceInputConfig(
        version=_integer(data.get("version", defaults.version), "version"),
        asr_provider=_string(data.get("asr_provider", defaults.asr_provider), "asr_provider"),
        endpoint=_string(data.get("endpoint", defaults.endpoint), "endpoint"),
        resource_id=_string(data.get("resource_id", defaults.resource_id), "resource_id"),
        language=_string(data.get("language", defaults.language), "language"),
        input_device=input_device,
        max_recording_seconds=_number(
            data.get("max_recording_seconds", defaults.max_recording_seconds),
            "max_recording_seconds",
        ),
    )


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VoiceConfigError(f"Voice setting must be a non-empty string: {name}")
    return value.strip()


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise VoiceConfigError(f"Voice setting must be an integer: {name}")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise VoiceConfigError(f"Voice setting must be a number: {name}")
    return float(value)
