"""Private local API-key storage for Axis providers."""

from contextlib import suppress
from json import JSONDecodeError, dumps, loads
from pathlib import Path
from tempfile import NamedTemporaryFile

from axis_coding.paths import AxisPaths


class CredentialStoreError(ValueError):
    """Axis credential storage is invalid or cannot be updated safely."""


class FileCredentialStore:
    """Small JSON-backed API-key store with owner-only permissions."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or credentials_path()

    def get(self, name: str) -> str | None:
        return self._load().get(_validate_name(name))

    def set(self, name: str, value: str) -> None:
        normalized_name = _validate_name(name)
        normalized_value = value.strip()
        if not normalized_value:
            raise CredentialStoreError("Credential value must not be empty")
        data = self._load()
        data[normalized_name] = normalized_value
        self._save(data)

    def delete(self, name: str) -> bool:
        normalized_name = _validate_name(name)
        data = self._load()
        existed = normalized_name in data
        if existed:
            del data[normalized_name]
            self._save(data)
        return existed

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._load()))

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            raw = loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, JSONDecodeError) as exc:
            raise CredentialStoreError(
                f"Could not load Axis credentials from {self.path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise CredentialStoreError("Axis credentials must be a JSON object")
        result: dict[str, str] = {}
        for name, value in raw.items():
            if not isinstance(name, str) or not name.strip():
                raise CredentialStoreError("Axis credential names must be non-empty strings")
            if not isinstance(value, str) or not value.strip():
                raise CredentialStoreError(
                    f"Axis credential value must be a non-empty string: {name}"
                )
            result[name.strip()] = value.strip()
        return result

    def _save(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(dumps(data, indent=2, sort_keys=True) + "\n")
                temp_file.flush()
            temp_path.chmod(0o600)
            temp_path.replace(self.path)
            self.path.chmod(0o600)
        except OSError as exc:
            if temp_path is not None:
                with suppress(OSError):
                    temp_path.unlink()
            raise CredentialStoreError(
                f"Could not save Axis credentials to {self.path}: {exc}"
            ) from exc


def credentials_path(paths: AxisPaths | None = None) -> Path:
    return (paths or AxisPaths()).home / "credentials.json"


def _validate_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise CredentialStoreError("Credential name must not be empty")
    return normalized
