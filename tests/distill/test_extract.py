"""Extraction: observations with real evidence, stored durably, extracted once per text."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from repartee.distill.extract import ExtractionResult
from repartee.distill.prompts import EXTRACT_VERSION
from repartee.distill.state import ObservationLog
from repartee.schema import Kind
from tests.distill.helpers import OWNER, entry, ids_in, msg, obs, one_per_chunk


def _two_sources():
    return [
        msg("wa-1", "Ana Ruiz", "Tu viens samedi ?", datetime(2021, 3, 12, 14, 5), seq=0),
        msg("wa-2", OWNER, "oui je prends le vélo", datetime(2021, 3, 12, 14, 6), seq=1),
        entry("nt-1", "Rode to work every day this week. I love it.",
              ts=datetime(2021, 3, 14)),
    ]


async def test_extract_writes_observations_and_is_incremental(env):
    env.index(_two_sources())
    env.fake.on("extract", one_per_chunk())

    report = await env.distiller().run(stages=["extract"])

    assert report.errors == []
    assert report.chunks_extracted == 2
    assert report.observations == 2
    assert len(env.fake.calls("extract")) == 2
    shards = sorted((env.home.derived_dir / "observations").glob("*.jsonl"))
    lines = [json.loads(line) for s in shards for line in s.read_text().splitlines()]
    assert {line["chunk_id"] for line in lines} == {c.id for c in env.corpus.iter_chunks()}
    assert all(line["prompt_version"] == EXTRACT_VERSION and line["chunk_hash"]
               for line in lines)
    assert all(o["id"].startswith("ob-") for line in lines for o in line["observations"])
    # nothing that cost money lives in the cache
    assert not list(env.home.cache_dir.rglob("*.jsonl"))

    again = await env.distiller().run(stages=["extract"])
    assert again.chunks_extracted == 0
    assert len(env.fake.calls("extract")) == 2

    edited = _two_sources()
    edited[2] = entry("nt-1", "Rode to work every day this week. I love it, even in rain.",
                      ts=datetime(2021, 3, 14))
    env.index(edited)
    third = await env.distiller().run(stages=["extract"])
    assert third.chunks_extracted == 1
    assert len(env.fake.calls("extract")) == 3
    assert "even in rain" in env.fake.calls("extract")[-1].messages[-1].text


async def test_extract_request_shape(env):
    env.index(_two_sources())
    env.fake.on("extract", ExtractionResult())
    await env.distiller().run(stages=["extract"])

    request = next(r for r in env.fake.calls("extract") if "wa-1" in r.messages[-1].text)
    assert request.tier == "bulk" and request.output_schema is ExtractionResult
    assert len(request.system) == 1 and request.system[0].cache
    assert OWNER in request.system[0].text
    assert "Tu viens" not in request.system[0].text  # the chunk is only in the user turn
    turn = request.messages[-1].text
    assert "[wa-1] 2021-03-12 14:05 Ana Ruiz: Tu viens samedi ?" in turn
    assert "[wa-2] 2021-03-12 14:06 me: oui je prends le vélo" in turn
    assert ids_in(request) == ["wa-1", "wa-2"]


async def test_empty_extraction_is_recorded_and_not_retried(env):
    env.index(_two_sources())
    env.fake.on("extract", ExtractionResult())
    first = await env.distiller().run(stages=["extract"])
    assert first.chunks_extracted == 2 and first.observations == 0
    await env.distiller().run(stages=["extract"])
    assert len(env.fake.calls("extract")) == 2


async def test_extract_drops_mirror_turn_evidence(env):
    transcript = "2025-06-01-childhood.md"
    env.index([
        entry("iv-1", "Where did you grow up? Nantes, maybe?", kind=Kind.INTERVIEW,
              source="interviews", container=transcript, author="mirror", seq=0),
        entry("iv-2", "In Rennes, until I was eleven.", kind=Kind.INTERVIEW,
              source="interviews", container=transcript, seq=1),
    ])
    env.fake.on("extract", ExtractionResult(observations=[
        obs("timeline", "Nantes", "I grew up in Nantes.", ["iv-1"]),
        obs("topic", "childhood", "I grew up in Rennes until I was eleven.", ["iv-1", "iv-2"]),
        obs("topic", "childhood", "I had a dog.", ["nt-404"]),
        obs("topic", "childhood", "I lived in Rennes as a child.", []),
    ]))

    report = await env.distiller().run(stages=["extract"])

    assert report.observations == 1 and report.observations_dropped == 3
    (record,) = ObservationLog(env.home.derived_dir / "observations").records.values()
    (kept,) = record.observations
    assert kept.text == "I grew up in Rennes until I was eleven."
    assert kept.evidence == ["iv-2"]
    turn = env.fake.calls("extract")[0].messages[-1].text
    assert "[iv-1] mirror (context only): Where did you grow up?" in turn
    assert "[iv-2] me: In Rennes" in turn


@pytest.mark.parametrize(("evidence", "expected"), [
    (["nt-1"], 0.8),             # journal prior
    (["wa-2"], 0.6),             # chat, owner's own line
    (["wa-1"], 0.6 * 0.7),       # chat, only someone else's line
    (["wa-1", "wa-2"], 0.6),
])
async def test_confidence_uses_source_prior(env, evidence, expected):
    env.index(_two_sources())

    def reply(request):
        cited = [e for e in evidence if e in ids_in(request)]
        if not cited:
            return ExtractionResult()
        return ExtractionResult(observations=[obs("topic", "cycling", "I cycle.", cited)])

    env.fake.on("extract", reply)
    await env.distiller().run(stages=["extract"])

    records = ObservationLog(env.home.derived_dir / "observations").records.values()
    (found,) = [o for r in records for o in r.observations]
    assert found.confidence == pytest.approx(expected)


async def test_own_health_taste_behaviour_need_a_line_by_the_owner(env):
    env.index(_two_sources())

    def reply(request):
        ids = ids_in(request)
        if "wa-1" not in ids:
            return ExtractionResult()
        return ExtractionResult(observations=[
            obs("health", "migraines", "I get migraines.", ["wa-1"]),     # other's line only
            obs("taste", "jazz", "I like jazz.", ["wa-1"]),
            obs("behaviour", "lateness", "I run late.", ["wa-1"]),
            obs("person", "Ana", "Ana gets migraines.", ["wa-1"]),        # kept: about Ana
            obs("health", "sleep", "I sleep badly.", ["wa-1", "wa-2"]),   # kept: owner's line
        ])

    env.fake.on("extract", reply)
    report = await env.distiller().run(stages=["extract"])

    records = ObservationLog(env.home.derived_dir / "observations").records.values()
    kept = sorted(o.subject for r in records for o in r.observations)
    assert kept == ["Ana", "sleep"]
    assert report.observations_dropped == 3


async def test_model_confidence_scales_and_is_clamped(env):
    env.index(_two_sources()[2:])
    env.fake.on("extract", ExtractionResult(observations=[
        obs("topic", "cycling", "I cycle to work.", ["nt-1"], confidence=0.5),
        obs("topic", "cycling", "I love cycling.", ["nt-1"], confidence=7.0),
    ]))
    await env.distiller().run(stages=["extract"])
    records = ObservationLog(env.home.derived_dir / "observations").records.values()
    values = sorted(o.confidence for r in records for o in r.observations)
    assert values == pytest.approx([0.4, 0.8])


async def test_bad_dates_are_cleared(env):
    env.index(_two_sources()[2:])
    env.fake.on("extract", ExtractionResult(observations=[
        obs("timeline", "2021", "I started cycling to work.", ["nt-1"], when="2021-03"),
        obs("topic", "cycling", "I like it.", ["nt-1"], when="last spring"),
    ]))
    await env.distiller().run(stages=["extract"])
    records = ObservationLog(env.home.derived_dir / "observations").records.values()
    whens = sorted(str(o.when) for r in records for o in r.observations)
    assert whens == ["2021-03", "None"]
