"""Durable state files under derived/: shards, torn lines, consumed ids, fingerprints."""

from __future__ import annotations

from repartee.distill import state
from repartee.distill.state import ConsumedSet, ExtractionRecord, Fingerprints, ObservationLog


def _record(n: int, chunk_hash: str = "h") -> ExtractionRecord:
    return ExtractionRecord(chunk_id=f"ch-{n}", chunk_hash=chunk_hash, prompt_version="v")


def test_log_rolls_shards_skips_torn_lines_and_latest_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "SHARD_LIMIT", 150)
    log = ObservationLog(tmp_path)
    log.append([_record(1)])
    log.append([_record(2)])
    log.append([_record(1, "h2")])
    shards = sorted(p.name for p in tmp_path.glob("*.jsonl"))
    assert shards == ["obs-0001.jsonl", "obs-0002.jsonl", "obs-0003.jsonl"]
    with (tmp_path / "obs-0003.jsonl").open("a") as handle:
        handle.write('{"chunk_id": "ch-3", "chunk_ha')  # crash mid-write
    (tmp_path / "obs-notes.jsonl").write_text("not a shard\n")

    reloaded = ObservationLog(tmp_path)
    assert set(reloaded.records) == {"ch-1", "ch-2"}
    assert reloaded.is_current("ch-1", "h2", "v")
    assert not reloaded.is_current("ch-1", "h", "v")
    assert not reloaded.is_current("ch-2", "h", "other-version")
    assert not reloaded.is_current("ch-2", None, "v")


def test_consumed_and_fingerprints_persist(tmp_path):
    consumed = ConsumedSet(tmp_path / "consolidated.txt")
    consumed.add(["ob-b", "ob-a"])
    assert (tmp_path / "consolidated.txt").read_text() == "ob-a\nob-b\n"
    assert "ob-a" in ConsumedSet(tmp_path / "consolidated.txt")

    marks = Fingerprints(tmp_path / "distill-state.json")
    marks.set("core", "abc")
    assert Fingerprints(tmp_path / "distill-state.json").get("core") == "abc"
    (tmp_path / "broken.json").write_text("{")
    assert Fingerprints(tmp_path / "broken.json").get("core") is None
