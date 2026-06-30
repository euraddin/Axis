"""User-home index for durable Axis coding sessions."""

from dataclasses import dataclass
from pathlib import Path
from time import time
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from axis_coding.paths import AxisPaths


class SessionRecordModel(BaseModel):
    """Forward-compatible JSON shape stored in a project index."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    id: str = Field(min_length=1)
    path: str
    cwd: str
    model: str = Field(min_length=1)
    provider_name: str | None = None
    title: str | None = None
    created_at: float = Field(ge=0)
    updated_at: float = Field(ge=0)


@dataclass(frozen=True, slots=True)
class CodingSessionRecord:
    """Typed metadata for one indexed JSONL session."""

    id: str
    path: Path
    cwd: Path
    model: str
    provider_name: str | None
    title: str | None
    created_at: float
    updated_at: float

    @classmethod
    def from_model(cls, model: SessionRecordModel) -> CodingSessionRecord:
        return cls(
            id=model.id,
            path=Path(model.path),
            cwd=Path(model.cwd),
            model=model.model,
            provider_name=model.provider_name,
            title=model.title,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def to_model(self) -> SessionRecordModel:
        return SessionRecordModel(
            id=self.id,
            path=str(self.path),
            cwd=str(self.cwd),
            model=self.model,
            provider_name=self.provider_name,
            title=self.title,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class SessionManager:
    """Create, index, find and update project-scoped sessions."""

    def __init__(self, paths: AxisPaths | None = None) -> None:
        self.paths = paths or AxisPaths()

    @property
    def index_path(self) -> Path:
        """Return the legacy global index path read for compatibility."""
        return self.paths.sessions_dir / "index.jsonl"

    def project_index_path(self, cwd: Path) -> Path:
        return self.paths.project_session_dir(cwd) / "index.jsonl"

    def list_sessions(self, cwd: Path | None = None) -> list[CodingSessionRecord]:
        records = self._read_project_records(cwd) if cwd is not None else self._read_all_records()
        return sorted(records, key=lambda record: record.updated_at, reverse=True)

    def get_session(self, session_id: str) -> CodingSessionRecord | None:
        return next(
            (record for record in self._read_all_records() if record.id == session_id),
            None,
        )

    def latest_session_for_cwd(self, cwd: Path) -> CodingSessionRecord | None:
        records = self.list_sessions(cwd)
        return records[0] if records else None

    def create_session(
        self,
        *,
        cwd: Path,
        model: str,
        provider_name: str | None = None,
        title: str | None = None,
        session_id: str | None = None,
    ) -> CodingSessionRecord:
        now = time()
        resolved = cwd.expanduser().resolve()
        record_id = session_id or uuid4().hex
        path = self.paths.session_path(resolved, record_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = CodingSessionRecord(
            id=record_id,
            path=path,
            cwd=resolved,
            model=model,
            provider_name=provider_name,
            title=title,
            created_at=now,
            updated_at=now,
        )
        self._upsert(record)
        return record

    def get_or_create_default_session(
        self,
        *,
        cwd: Path,
        model: str,
        provider_name: str | None = None,
    ) -> CodingSessionRecord:
        resolved = cwd.expanduser().resolve()
        suffix = self.paths.project_session_dir(resolved).name.rsplit("-", maxsplit=1)[-1]
        session_id = f"default-{suffix}"
        existing = self.get_session(session_id)
        if existing is not None:
            return existing
        now = time()
        record = CodingSessionRecord(
            id=session_id,
            path=self.paths.default_session_path(resolved),
            cwd=resolved,
            model=model,
            provider_name=provider_name,
            title="Default session",
            created_at=now,
            updated_at=now,
        )
        record.path.parent.mkdir(parents=True, exist_ok=True)
        self._upsert(record)
        return record

    def touch_session(
        self,
        session_id: str,
        *,
        model: str | None = None,
        provider_name: str | None = None,
        title: str | None = None,
    ) -> CodingSessionRecord | None:
        existing = self.get_session(session_id)
        if existing is None:
            return None
        updated = CodingSessionRecord(
            id=existing.id,
            path=existing.path,
            cwd=existing.cwd,
            model=model or existing.model,
            provider_name=(provider_name if provider_name is not None else existing.provider_name),
            title=title if title is not None else existing.title,
            created_at=existing.created_at,
            updated_at=time(),
        )
        self._upsert(updated)
        return updated

    def _read_index(self, path: Path) -> list[CodingSessionRecord]:
        if not path.exists():
            return []
        records: list[CodingSessionRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(
                    CodingSessionRecord.from_model(SessionRecordModel.model_validate_json(line))
                )
        return records

    def _read_project_records(self, cwd: Path) -> list[CodingSessionRecord]:
        resolved = cwd.expanduser().resolve()
        records = self._read_index(self.project_index_path(resolved))
        records.extend(
            record for record in self._read_index(self.index_path) if record.cwd == resolved
        )
        return _deduplicate_records(records)

    def _read_all_records(self) -> list[CodingSessionRecord]:
        records = self._read_index(self.index_path)
        for index_path in self.paths.sessions_dir.glob("*/index.jsonl"):
            records.extend(self._read_index(index_path))
        return _deduplicate_records(records)

    def _write_index(self, path: Path, records: list[CodingSessionRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(record.to_model().model_dump_json() for record in records)
        path.write_text(f"{content}\n" if content else "", encoding="utf-8")

    def _upsert(self, record: CodingSessionRecord) -> None:
        path = self.project_index_path(record.cwd)
        records = [item for item in self._read_index(path) if item.id != record.id]
        records.append(record)
        self._write_index(path, records)


def _deduplicate_records(records: list[CodingSessionRecord]) -> list[CodingSessionRecord]:
    by_id: dict[str, CodingSessionRecord] = {}
    for record in records:
        current = by_id.get(record.id)
        if current is None or record.updated_at >= current.updated_at:
            by_id[record.id] = record
    return list(by_id.values())
