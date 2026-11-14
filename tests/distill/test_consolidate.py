"""Consolidation: flat ops from the model become validated claim operations."""

from __future__ import annotations

from datetime import datetime

from repartee.distill.consolidate import ConsolidationResult, PlannedOp, combined_confidence
from repartee.distill.extract import ExtractionResult
from repartee.schema import AddClaim, ClaimOrigin, ClaimStatus, CreatePage, Kind, PageKind
from tests.distill.helpers import OWNER, entry, ids_in, msg, obs


def _chat():
    return [
        msg("wa-1", "Ana Ruiz", "Tu bosses encore de nuit ?", datetime(2021, 3, 12, 14, 5)),
        msg("wa-2", OWNER, "toi oui, moi non haha", datetime(2021, 3, 12, 14, 6), seq=1),
    ]


def _extract_about_ana(env, *texts: str) -> None:
    env.fake.on("extract", lambda request: ExtractionResult(observations=[
        obs("person", "Ana Ruiz", text, ids_in(request)) for text in texts]))


def _add_who(request) -> ConsolidationResult:
    return ConsolidationResult(ops=[PlannedOp(op="add", section="Who", text="Ana works nights.",
                                              observations=["o1"])])


async def test_consolidate_applies_ops_and_marks_consumed(env):
    env.index(_chat())
    _extract_about_ana(env, "Ana works night shifts.")
    env.fake.on("consolidate", _add_who)
    env.fake.on("core", {"markdown": "I am Camille. My friend Ana works nights."})
    env.fake.on("gaps", {"questions": [{"text": "How did you meet Ana?",
                                        "page": "people/ana-ruiz"}]})

    report = await env.distiller().run()

    assert report.errors == []
    assert report.pages_created == 1 and report.ops_applied == 1
    (claim,) = env.model.claims(page="people/ana-ruiz")
    assert claim.text == "Ana works nights." and claim.section == "Who"
    assert claim.evidence == ["wa-1", "wa-2"]
    assert claim.origin == ClaimOrigin.PIPELINE
    assert report.claim_ids == [claim.id]
    assert "Ana works nights. ^" in (env.home.self_dir / "people/ana-ruiz.md").read_text()
    consumed = (env.home.derived_dir / "consolidated.txt").read_text().split()
    assert len(consumed) == 1 and consumed[0].startswith("ob-")
    (request,) = env.fake.calls("consolidate")
    assert request.tier == "smart" and request.system[0].cache
    assert "Sections you may add to: Who, Relationship, How we talk, Shared history" in (
        request.messages[-1].text)
    assert "- o1 (from a chat, confidence 0.60; person: Ana Ruiz) Ana works night shifts." in (
        request.messages[-1].text)
    assert report.requests == len(env.fake.requests) == 4

    again = await env.distiller().run()
    assert again.requests == 0 and len(env.fake.requests) == 4
    assert again.errors == []


async def test_consolidate_sees_existing_claims_and_supports(env):
    env.index(_chat())
    _extract_about_ana(env, "Ana works night shifts.")
    env.fake.on("consolidate", _add_who)
    await env.distiller().run(stages=["extract", "consolidate"])
    (claim,) = env.model.claims(page="people/ana-ruiz")

    # a later chunk says it again: the model supports, or re-adds the same text
    env.index(_chat() + [
        msg("wa-3", OWNER, "tu dors le jour du coup", datetime(2021, 6, 1, 10, 0), seq=2)])
    _extract_about_ana(env, "Ana sleeps during the day after night shifts.")
    env.fake.on("consolidate", _add_who)  # same text: a duplicate becomes support
    report = await env.distiller().run(stages=["extract", "consolidate"])

    request = env.fake.calls("consolidate")[-1].messages[-1].text
    assert f"- {claim.id} [Who] (inferred, confidence " in request
    assert report.ops_refused == 0 and report.ops_applied == 1
    (after,) = env.model.claims(page="people/ana-ruiz")
    assert after.id == claim.id and after.confidence > claim.confidence


async def test_consolidate_refused_ops_are_counted_not_fatal(env):
    env.index(_chat())
    _extract_about_ana(env, "Ana works night shifts.")
    env.fake.on("consolidate", ConsolidationResult(ops=[
        PlannedOp(op="add", section="Hobbies", text="Ana paints.", observations=["o1"]),
        PlannedOp(op="support", claim_id="c-00000000", observations=["o1"]),
        PlannedOp(op="add", section="Who", text="Ana is a nurse.", observations=["o9"]),
        PlannedOp(op="revise", claim_id="c-00000000"),
        PlannedOp(op="add", section="Who", text="Ana works nights.", observations=["o1"]),
        PlannedOp(op="question", text="Does Ana still work nights?"),
    ]))

    report = await env.distiller().run(stages=["extract", "consolidate"])

    assert report.ops_refused == 4
    assert report.ops_applied == 2 and report.questions_added == 1
    assert [c.text for c in env.model.claims()] == ["Ana works nights."]
    assert [q.page for q in env.model.questions()] == ["people/ana-ruiz"]
    assert report.errors == []


async def test_consolidate_does_not_touch_confirmed_claims(env):
    env.model.apply([CreatePage(kind=PageKind.PERSON, slug="ana-ruiz", title="Ana Ruiz")])
    env.index(_chat())
    (result,) = env.model.apply([AddClaim(page="people/ana-ruiz", section="Who",
                                          text="Ana is a nurse.", evidence=["wa-1"])])
    env.model.confirm(result.claim_id)
    env.model.save()
    _extract_about_ana(env, "Ana became a midwife.")
    env.fake.on("consolidate", lambda r: ConsolidationResult(ops=[
        PlannedOp(op="revise", claim_id=result.claim_id, text="Ana is a midwife.",
                  observations=["o1"]),
        PlannedOp(op="supersede", claim_id=result.claim_id, text="Ana is a midwife.",
                  observations=["o1"]),
        PlannedOp(op="support", claim_id=result.claim_id, observations=["o1"]),
    ]))

    report = await env.distiller().run(stages=["extract", "consolidate"])

    assert report.ops_refused == 2 and report.ops_applied == 1
    claim = env.model.claim(result.claim_id)
    assert claim.text == "Ana is a nurse." and claim.status == ClaimStatus.CONFIRMED
    assert claim.evidence == ["wa-1", "wa-2"]
    request = env.fake.calls("consolidate")[0].messages[-1].text
    assert f"- {claim.id} [Who] (confirmed by me, confidence" in request


async def test_interview_claims_get_interview_origin(env):
    transcript = "2025-06-01-childhood.md"
    env.index([
        entry("iv-1", "Where did you grow up?", kind=Kind.INTERVIEW, source="interviews",
              container=transcript, author="mirror"),
        entry("iv-2", "In Rennes, until I was eleven.", kind=Kind.INTERVIEW,
              source="interviews", container=transcript, seq=1),
    ])
    env.fake.on("extract", ExtractionResult(observations=[
        obs("timeline", "childhood", "I lived in Rennes until I was eleven.", ["iv-2"],
            when="1999")]))
    env.fake.on("consolidate", ConsolidationResult(ops=[
        PlannedOp(op="add", section="Events", text="I lived in Rennes until I was eleven.",
                  observations=["o1"])]))

    report = await env.distiller().run(stages=["extract", "consolidate"])

    (claim,) = env.model.claims(page="timeline/1999")
    assert report.claim_ids == [claim.id]
    assert claim.origin == ClaimOrigin.INTERVIEW and claim.status == ClaimStatus.INFERRED
    assert claim.confidence == 0.9 and claim.valid_from == "1999"
    assert claim.evidence == ["iv-2"]


async def test_consolidate_batches_large_pages_in_order(env):
    env.index(_chat())
    _extract_about_ana(env, *[f"Fact number {n} about Ana." for n in range(45)])
    sizes = []

    def reply(request):
        sizes.append(request.messages[-1].text.count("\n- o"))
        return ConsolidationResult()

    env.fake.on("consolidate", reply)
    report = await env.distiller().run(stages=["extract", "consolidate"])
    assert sizes == [40, 5] and report.errors == []
    consumed = (env.home.derived_dir / "consolidated.txt").read_text().split()
    assert len(consumed) == 45


def test_combined_confidence():
    assert combined_confidence([]) == 0.5
    assert combined_confidence([0.6]) == 0.6
    assert combined_confidence([0.4, 0.6]) == 0.7
    assert combined_confidence([0.9] * 10) == 0.95
