"""Durable distill state under ``<home>/derived/``, all plain text.

- ``observations/obs-NNNN.jsonl``: append-only shards (rolled at about 5 MB). One line per
  extracted chunk: ``{"chunk_id", "chunk_hash", "prompt_version", "at", "observations"}``.
  The last line for a chunk wins. A torn last line (crash mid-write) is ignored.
- ``consolidated.txt``: sorted ids of the observations already folded into the self-model.
- ``distill-state.json``: fingerprints of the inputs of the last core and gaps runs, so an
  unchanged self-model costs no request.
- ``rectify-state.json``: per page, a fingerprint of its confirmed key facts and the claims
  already checked against them: ``{"people/noor": {"facts": "<hash>", "checked": [ids]}}``.

These cost money to recompute, which is why they are in the home and not in the cache.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from repartee.schema import Observation

log = logging.getLogger(__name__)

SHARD_LIMIT = 5_000_000  # bytes
SHARD_PREFIX = "obs-"


class ExtractionRecord(BaseModel):
    """One extracted chunk: the observations it gave under one prompt version."""

    chunk_id: str
    chunk_hash: str
    prompt_version: str
    at: str = ""
    observations: list[Observation] = Field(default_factory=list)


class ObservationLog:
    """The append-only observation shards, indexed by chunk id (latest record wins)."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.records: dict[str, ExtractionRecord] = {}
        self._load()

    def _shards(self) -> list[Path]:
        if not self.dir.is_dir():
            return []
        found = [p for p in self.dir.glob(f"{SHARD_PREFIX}*.jsonl")
                 if p.stem.removeprefix(SHARD_PREFIX).isdigit()]
        return sorted(found, key=lambda p: int(p.stem.removeprefix(SHARD_PREFIX)))

    def _load(self) -> None:
        for shard in self._shards():
            with shard.open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = ExtractionRecord.model_validate_json(line)
                    except ValidationError:
                        log.warning("distill: skipping unreadable line %s:%d", shard.name, number)
                        continue
                    self.records[record.chunk_id] = record

    def is_current(self, chunk_id: str, chunk_hash: str | None, version: str) -> bool:
        """True when this chunk text was already extracted with this prompt version."""
        record = self.records.get(chunk_id)
        return bool(record and chunk_hash and record.chunk_hash == chunk_hash
                    and record.prompt_version == version)

    def append(self, records: Iterable[ExtractionRecord]) -> None:
        """Write records durably (one line each) and index them."""
        records = list(records)
        if not records:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        lines = "".join(r.model_dump_json() + "\n" for r in records).encode("utf-8")
        shards = self._shards()
        target = shards[-1] if shards else None
        if target is None:
            target = self.dir / f"{SHARD_PREFIX}0001.jsonl"
        elif target.stat().st_size and target.stat().st_size + len(lines) > SHARD_LIMIT:
            number = int(target.stem.removeprefix(SHARD_PREFIX)) + 1
            target = self.dir / f"{SHARD_PREFIX}{number:04d}.jsonl"
        with target.open("ab") as handle:
            handle.write(lines)
            handle.flush()
            os.fsync(handle.fileno())
        for record in records:
            self.records[record.chunk_id] = record


class ConsumedSet:
    """Observation ids already consolidated, kept in ``consolidated.txt``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.ids: set[str] = set()
        if self.path.is_file():
            text = self.path.read_text(encoding="utf-8")
            self.ids = {line.strip() for line in text.splitlines() if line.strip()}

    def __contains__(self, observation_id: object) -> bool:
        return observation_id in self.ids

    def add(self, observation_ids: Iterable[str]) -> None:
        """Mark ids consumed and write the file."""
        before = len(self.ids)
        self.ids.update(observation_ids)
        if len(self.ids) != before:
            atomic_write(self.path, "".join(f"{i}\n" for i in sorted(self.ids)))


class Fingerprints:
    """Small JSON map of stage name -> hash of that stage's last input."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, str] = {}
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                self.data = {str(k): str(v) for k, v in dict(loaded).items()}
            except (ValueError, TypeError):
                log.warning("distill: ignoring unreadable %s", self.path)

    def get(self, stage: str) -> str | None:
        """The last recorded fingerprint for a stage."""
        return self.data.get(stage)

    def set(self, stage: str, value: str) -> None:
        """Record a fingerprint and write the file."""
        self.data[stage] = value
        atomic_write(self.path, json.dumps(self.data, indent=1, sort_keys=True) + "\n")


class RectifyState:
    """What the rectify stage already checked, per page, kept in ``rectify-state.json``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                self.data = {str(k): {"facts": str(v.get("facts", "")),
                                      "checked": [str(i) for i in v.get("checked", [])]}
                             for k, v in dict(loaded).items() if isinstance(v, dict)}
            except (ValueError, TypeError, AttributeError):
                log.warning("distill: ignoring unreadable %s", self.path)

    def facts(self, page: str) -> str | None:
        """The fingerprint of the page's confirmed facts at its last check."""
        entry = self.data.get(page)
        return entry["facts"] if entry else None

    def checked(self, page: str) -> set[str]:
        """Claim ids already checked against the page's facts as they were then."""
        entry = self.data.get(page)
        return set(entry["checked"]) if entry else set()

    def set(self, page: str, facts: str, checked: Iterable[str]) -> None:
        """Record a page's check (call save() to write)."""
        self.data[page] = {"facts": facts, "checked": sorted(set(checked))}

    def save(self) -> None:
        """Write the file."""
        atomic_write(self.path, json.dumps(self.data, indent=1, sort_keys=True) + "\n")


def now_iso() -> str:
    """Current UTC time, seconds precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def atomic_write(path: Path, text: str) -> None:
    """Replace a file's content in one step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
