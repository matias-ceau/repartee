"""Key facts and retraction through the pipeline: an interview answer corrects an inference.

Everyone here is invented.
"""

from __future__ import annotations

from datetime import datetime

import regex

from repartee.distill.consolidate import ConsolidationResult, PlannedOp
from repartee.distill.core import CoreResult
from repartee.distill.extract import ExtractionResult
from repartee.distill.gaps import GapsResult
from repartee.distill.rectify import RectifyResult, Verdict
from repartee.llm import LLMRequest
from repartee.schema import AddClaim, ClaimStatus, CreatePage, Kind, PageKind
from tests.distill.helpers import OWNER, entry, ids_in, msg, obs

NOOR = "people/noor-haddad"
OBS_LINE = regex.compile(r"^- (o\d+) \(from an? (\w+).*?\) (.*)$", regex.MULTILINE)
CLAIM_LINE = regex.compile(r"^- (c-[0-9a-f]{8}) \(on [^)]*\) (.*)$", regex.MULTILINE)


def _chat() -> list:
    return [
        msg("wa-1", "Noor Haddad", "miss you already, come home soon", datetime(2024, 2, 3, 21, 0),
            container="Noor Haddad"),
        msg("wa-2", OWNER, "miss you too, back on sunday", datetime(2024, 2, 3, 21, 2),
            container="Noor Haddad", seq=1),
    ]


def _interview() -> list:
    where = "2026-11-15-family.md"
    return [
        entry("iv-0", "Who is Noor to you?", kind=Kind.INTERVIEW, source="interviews",
              container=where, author="mirror", ts=datetime(2026, 11, 15, 10, 0)),
        entry("iv-1", "Noor is one of my sisters, the youngest.", kind=Kind.INTERVIEW,
              source="interviews", container=where, ts=datetime(2026, 11, 15, 10, 1), seq=1),
    ]


def _extract(request: LLMRequest) -> ExtractionResult:
    ids = ids_in(request)
    if "iv-1" in ids:
        return ExtractionResult(observations=[
            obs("person", "Noor Haddad", "Noor is one of my sisters.", ["iv-1"])])
    return ExtractionResult(observations=[
        obs("person", "Noor Haddad", "Noor is my partner.", ids, confidence=0.6)])


def _consolidate(request: LLMRequest) -> ConsolidationResult:
    ops = []
    for ref, source, text in OBS_LINE.findall(request.messages[-1].text):
        if source == "interview":
            ops += [PlannedOp(op="set_fact", slot="relation", value="sister", observations=[ref]),
                    PlannedOp(op="add", section="Who", text="Noor is my youngest sister.",
                              observations=[ref])]
        else:
            ops.append(PlannedOp(op="add", section="Relationship", text=text,
                                 observations=[ref]))
    return ConsolidationResult(ops=ops)


def _rectify(request: LLMRequest) -> RectifyResult:
    return RectifyResult(verdicts=[
        Verdict(claim_id=cid, action="retract", slot="relation", reason="Noor is my sister")
        for cid, text in CLAIM_LINE.findall(request.messages[-1].text) if "partner" in text])


def _script(env) -> None:
    env.fake.on("extract", _extract)
    env.fake.on("consolidate", _consolidate)
    env.fake.on("rectify", _rectify)
    env.fake.on("core", CoreResult(markdown="My youngest sister Noor."))
    env.fake.on("gaps", GapsResult())


async def test_an_interview_answer_confirms_the_fact_and_retracts_the_inference(env):
    _script(env)
    env.index(_chat())
    first = await env.distiller().run()
    assert first.errors == []
    (partner,) = env.model.claims(page=NOOR)
    assert partner.text == "Noor is my partner." and partner.status == ClaimStatus.INFERRED
    assert env.fake.calls("rectify") == []  # nothing confirmed yet: nothing to check against

    env.index(_chat() + _interview())
    report = await env.distiller().run()

    assert report.errors == []
    fact = env.model.fact(NOOR, "relation")
    assert fact.value == "sister" and fact.status == ClaimStatus.CONFIRMED
    assert fact.evidence == ["iv-1"]
    assert report.facts_set == 1 and report.retracted == [partner.id]
    assert env.model.claim(partner.id) is None
    page = (env.home.self_dir / f"{NOOR}.md").read_text()
    assert "relation: sister\n" in page and "partner" not in page
    assert [c.text for c in env.model.claims(page=NOOR)] == ["Noor is my youngest sister."]

    (rectify,) = env.fake.calls("rectify")
    assert rectify.tier == "smart" and rectify.system[0].cache
    turn = rectify.messages[-1].text
    assert "Confirmed key facts:\n- relation: sister" in turn
    assert f"- {partner.id} (on {NOOR}, Relationship) Noor is my partner." in turn

    core = env.fake.calls("core")[-1].messages[-1].text
    stated, _, inferred = core.partition("# What was inferred")
    assert "# What I have stated myself" in stated
    assert "- key facts: relation: sister" in stated
    assert "Noor is my youngest sister." in stated  # said in an interview: first block
    assert "partner" not in core

    # the retracted claim is not inferred again from the same chat
    env.index(_chat() + _interview() + [
        msg("wa-3", "Noor Haddad", "love you, see you sunday", datetime(2024, 2, 4, 9, 0),
            container="Noor Haddad", seq=2)])
    again = await env.distiller().run()
    assert again.errors == [] and "partner" not in (
        env.home.self_dir / f"{NOOR}.md").read_text()
    assert len(env.fake.calls("rectify")) == 1  # facts unchanged, no unchecked claim


def _seed_pet(env) -> tuple[str, str]:
    env.index([entry("nt-1", "Pippa had a hypo this morning, vet says sugar.",
                     ts=datetime(2024, 3, 2))])
    env.model.apply([CreatePage(kind=PageKind.PERSON, slug="pippa", title="Pippa"),
                     CreatePage(kind=PageKind.TIMELINE, slug="2024", title="2024")])
    event, other = (r.claim_id for r in env.model.apply([
        AddClaim(page="timeline/2024", section="Events", evidence=["nt-1"],
                 text="In March 2024 my dog Pippa had a hypoglycemic episode."),
        AddClaim(page="timeline/2024", section="Events", evidence=["nt-1"],
                 text="In March 2024 I started running again."),
    ]))
    env.model.save()
    return event, other


async def test_owner_property_edit_revises_the_detail_and_keeps_the_event(env):
    event, other = _seed_pet(env)
    path = env.home.self_dir / "people/pippa.md"
    path.write_text(path.read_text().replace("updated:", "species: cat\nupdated:"))

    def revise(request: LLMRequest) -> RectifyResult:
        (cid, text), = CLAIM_LINE.findall(request.messages[-1].text)  # only the one naming Pippa
        return RectifyResult(verdicts=[Verdict(
            claim_id=cid, action="revise", slot="species", reason="Pippa is a cat",
            text=text.replace("dog", "cat"))])

    env.fake.on("rectify", revise)
    report = await env.distiller().run(stages=["rectify"])

    assert report.errors == [] and report.claim_ids == [event]
    assert env.model.fact("people/pippa", "species").status == ClaimStatus.CONFIRMED
    claim = env.model.claim(event)
    assert claim.text == "In March 2024 my cat Pippa had a hypoglycemic episode."
    assert claim.status == ClaimStatus.INFERRED and claim.evidence == ["nt-1"]
    assert env.model.claim(other).text == "In March 2024 I started running again."

    # incremental: nothing changed, no request; a changed fact checks the subject again
    await env.distiller().run(stages=["rectify"])
    assert len(env.fake.calls("rectify")) == 1
    path.write_text(path.read_text().replace("species: cat", "species: cat\nborn: 2015"))
    env.fake.on("rectify", RectifyResult())
    await env.distiller().run(stages=["rectify"])
    assert len(env.fake.calls("rectify")) == 2


async def test_rectify_cannot_touch_claims_it_was_not_asked_about(env):
    event, other = _seed_pet(env)
    path = env.home.self_dir / "people/pippa.md"
    path.write_text(path.read_text().replace("updated:", "species: cat\nupdated:"))
    env.fake.on("rectify", RectifyResult(verdicts=[
        Verdict(claim_id=other, action="retract", slot="species", reason="no reason")]))
    report = await env.distiller().run(stages=["rectify"])
    assert report.retracted == [] and env.model.claim(other) is not None


async def test_plan_counts_rectify_requests(env):
    _seed_pet(env)
    path = env.home.self_dir / "people/pippa.md"
    path.write_text(path.read_text().replace("updated:", "species: cat\nupdated:"))
    plan = env.distiller().plan()
    (stage,) = [s for s in plan.stages if s.stage == "rectify"]
    assert stage.requests == 1 and stage.input_tokens > 0


async def test_consolidate_cannot_retract_on_chat_evidence_alone(env):
    _script(env)
    env.index(_chat())
    await env.distiller().run(stages=["extract", "consolidate"])
    (partner,) = env.model.claims(page=NOOR)

    env.index(_chat() + [msg("wa-3", "Noor Haddad", "say hi to mum from your sister",
                             datetime(2024, 5, 1, 9, 0), container="Noor Haddad", seq=2)])
    env.fake.on("consolidate", ConsolidationResult(ops=[
        PlannedOp(op="retract", claim_id=partner.id, text="she calls herself my sister",
                  observations=["o1"]),
        PlannedOp(op="set_fact", slot="relation", value="sister", observations=["o1"])]))
    report = await env.distiller().run(stages=["extract", "consolidate"])

    assert report.retracted == [] and env.model.claim(partner.id) is not None
    assert report.ops_refused == 1 and report.facts_set == 1
    assert env.model.fact(NOOR, "relation").status == ClaimStatus.INFERRED
    turn = env.fake.calls("consolidate")[-1].messages[-1].text
    assert "Key facts:\n(none yet)" in turn
    env.fake.on("consolidate", ConsolidationResult())
    env.index(_chat() + [msg("wa-4", OWNER, "ok", datetime(2024, 6, 1, 9, 0),
                             container="Noor Haddad", seq=3)])
    await env.distiller().run(stages=["extract", "consolidate"])
    assert "- relation: sister (inferred)" in env.fake.calls("consolidate")[-1].messages[-1].text
