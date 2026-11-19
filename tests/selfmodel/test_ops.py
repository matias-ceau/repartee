"""Pipeline operations: validated, refused with a reason, never raising."""

from __future__ import annotations

import pytest

from repartee.schema import (
    AddClaim,
    AskQuestion,
    ClaimOrigin,
    ClaimStatus,
    CreatePage,
    PageKind,
    ReviseClaim,
    SupersedeClaim,
    SupportClaim,
)
from repartee.selfmodel import SelfModel

from .conftest import (
    ANA,
    ANA_KEY,
    C_LYON,
    C_NURSE,
    C_TALK,
    NURSE_LINE,
    NURSE_TEXT,
    evidence_known,
    loaded,
    read_page,
    read_sidecar,
    write_page,
)


def test_apply_add_support_revise_supersede_question_create_page(self_dir):
    model = SelfModel(self_dir)
    model.load()

    (res,) = model.apply([CreatePage(kind=PageKind.PERSON, slug="Bo Lind", title="Bo Lind",
                                     aliases=["Bo"])])
    assert res.ok
    key = "people/bo-lind"
    assert model.page(key).title == "Bo Lind"
    (again,) = model.apply([CreatePage(kind=PageKind.PERSON, slug="bo-lind", title="Other")])
    assert again.ok and model.page(key).title == "Bo Lind"  # existing page: no-op success

    (add,) = model.apply([AddClaim(page=key, section="who", text="My climbing partner.",
                                   evidence=["wa-1", "wa-1"], confidence=0.4)])
    assert add.ok and add.claim_id
    cid = add.claim_id
    model.save()
    assert f"## Who\n- My climbing partner. ^{cid}\n" in read_page(self_dir, key)
    meta = read_sidecar(self_dir, key)["claims"][cid]
    assert meta["status"] == "inferred" and meta["evidence"] == ["wa-1"]
    assert meta["section"] == "Who" and meta["text"] == "My climbing partner."

    (sup,) = model.apply([SupportClaim(claim_id=cid, evidence=["wa-1", "wa-2", "wa-3"])])
    assert sup.ok
    claim = model.claim(cid)
    assert claim.evidence == ["wa-1", "wa-2", "wa-3"]
    expected = 0.4
    for _ in range(2):
        expected = expected + (1 - expected) * 0.25
    assert claim.confidence == pytest.approx(expected)
    page_before = model.read_page(key)

    (rev,) = model.apply([ReviseClaim(claim_id=cid, text="My regular climbing partner.",
                                      evidence=["wa-4"])])
    assert rev.ok and rev.claim_id == cid
    assert model.read_page(key) == page_before.replace("My climbing", "My regular climbing")
    assert model.claim(cid).status is ClaimStatus.INFERRED
    assert "wa-4" in model.claim(cid).evidence

    (sup2,) = model.apply([SupersedeClaim(claim_id=cid, text="We stopped climbing together.",
                                          evidence=["wa-5"], valid_to="2025-06",
                                          valid_from="2025-06")])
    assert sup2.ok and sup2.claim_id != cid
    old, new = model.claim(cid), model.claim(sup2.claim_id)
    assert old.section == "Earlier" and old.valid_to == "2025-06"
    assert new.section == "Who" and new.valid_from == "2025-06"
    text = model.read_page(key)
    assert f"## Who\n- We stopped climbing together. ^{new.id}\n" in text
    assert f"## Earlier\n- My regular climbing partner. ^{cid}\n" in text

    (q,) = model.apply([AskQuestion(text="When did we stop climbing?", page=key)])
    assert q.ok and q.claim_id.startswith("q-")
    (qdup,) = model.apply([AskQuestion(text="when did we stop  climbing")])
    assert not qdup.ok and qdup.reason == "duplicate"
    model.save()
    oq = read_page(self_dir, "open-questions")
    assert f"- [ ] When did we stop climbing? (page: {key}) ^{q.claim_id}\n" in oq

    fresh = loaded(self_dir)
    assert fresh.claim(new.id).text == "We stopped climbing together."
    assert [x.text for x in fresh.questions()] == ["When did we stop climbing?"]


def test_support_confidence_is_capped_and_never_lowered(ana):
    model = loaded(ana)
    model.apply([SupportClaim(claim_id=C_LYON, evidence=[f"e{i}" for i in range(40)])])
    assert model.claim(C_LYON).confidence == pytest.approx(0.95)
    model.confirm(C_TALK)
    model._claims[C_TALK].confidence = 0.99  # e.g. set by review
    (res,) = model.apply([SupportClaim(claim_id=C_TALK, evidence=["new"])])
    assert res.ok and model.claim(C_TALK).confidence == pytest.approx(0.99)
    (res,) = model.apply([SupportClaim(claim_id=C_LYON, evidence=["wa-1"])])
    assert res.ok  # already-known evidence: nothing to do, not an error


REFUSALS = [
    ("unknown_claim", ReviseClaim(claim_id="c-00000000", text="x")),
    ("unknown_claim", SupportClaim(claim_id="c-00000000", evidence=["wa-1"])),
    ("unknown_claim", SupersedeClaim(claim_id="nonsense", text="x", evidence=["wa-1"])),
    ("confirmed", ReviseClaim(claim_id=C_TALK, text="x")),
    ("confirmed", SupersedeClaim(claim_id=C_TALK, text="x", evidence=["wa-1"])),
    ("section", AddClaim(page=ANA_KEY, section="Notes", text="x", evidence=["wa-1"])),
    ("section", AddClaim(page=ANA_KEY, section="Hobbies", text="x", evidence=["wa-1"])),
    ("no_page", AddClaim(page="people/nobody", section="Who", text="x", evidence=["wa-1"])),
    ("no_evidence", AddClaim(page=ANA_KEY, section="Who", text="x", evidence=[])),
    ("no_evidence", AddClaim(page=ANA_KEY, section="Who", text="x", evidence=["", " "])),
    ("no_evidence", SupersedeClaim(claim_id=C_NURSE, text="x", evidence=[])),
    ("no_evidence", SupportClaim(claim_id=C_NURSE, evidence=[])),
    ("unknown_evidence", AddClaim(page=ANA_KEY, section="Who", text="x", evidence=["ghost"])),
    ("unknown_evidence", SupportClaim(claim_id=C_NURSE, evidence=["ghost"])),
    ("unknown_evidence", SupersedeClaim(claim_id=C_NURSE, text="x", evidence=["ghost"])),
    ("duplicate", AddClaim(page=ANA_KEY, section="How we talk",
                           text="works as a nurse — mostly night shifts", evidence=["wa-1"])),
    ("duplicate", ReviseClaim(claim_id=C_LYON, text=NURSE_TEXT)),
    ("duplicate", SupersedeClaim(claim_id=C_NURSE, text="Works as a nurse, mostly night shifts.",
                                 evidence=["wa-1"])),
    ("invalid", AddClaim(page=ANA_KEY, section="Who", text="   ", evidence=["wa-1"])),
    ("invalid", AddClaim(page=ANA_KEY, section="Who", text="sneaky ^c-12345678",
                         evidence=["wa-1"])),
    ("invalid", AddClaim(page=ANA_KEY, section="Who", text="x", evidence=["wa-1"],
                         origin=ClaimOrigin.USER)),
    ("invalid", CreatePage(kind=PageKind.CORE, slug="core", title="Core")),
    ("invalid", CreatePage(kind=PageKind.VOICE, slug="whatsapp.fr", title="x")),
    ("invalid", {"op": "add", "page": ANA_KEY}),
    ("invalid", {"op": "explode"}),
    ("invalid", None),
    ("invalid", "add a claim please"),
    ("invalid", AskQuestion(text="\n\t ")),
]


@pytest.mark.parametrize(
    "reason,op", REFUSALS, ids=[f"{r}-{i}" for i, (r, _) in enumerate(REFUSALS)]
)
def test_apply_refusals(ana, reason, op):
    model = loaded(ana)
    model.confirm(C_TALK)
    model.save()
    before_page = read_page(ana, ANA_KEY)
    before_sidecar = read_sidecar(ana, ANA_KEY)
    (res,) = model.apply([op], existing_evidence=evidence_known("wa-1", "wa-2"))
    assert not res.ok
    assert res.reason == reason
    model.save()
    assert read_page(ana, ANA_KEY) == before_page
    assert read_sidecar(ana, ANA_KEY) == before_sidecar


def test_matches_rejected_and_rejected_reasons(ana):
    write_page(ana, ANA_KEY, ANA.replace(NURSE_LINE, ""))
    model = loaded(ana)
    cases = [
        ("rejected", ReviseClaim(claim_id=C_NURSE, text="x")),
        ("rejected", SupersedeClaim(claim_id=C_NURSE, text="x", evidence=["wa-1"])),
        ("rejected", SupportClaim(claim_id=C_NURSE, evidence=["wa-1"])),
        ("matches_rejected", AddClaim(page=ANA_KEY, section="Earlier",
                                      text="Works as a nurse, mostly night shifts",
                                      evidence=["wa-1"])),
        ("matches_rejected", SupersedeClaim(claim_id=C_LYON,
                                            text="Works as a nurse, mostly night shifts",
                                            evidence=["wa-1"])),
    ]
    for reason, op in cases:
        (res,) = model.apply([op])
        assert (res.ok, res.reason) == (False, reason), op


def test_unknown_evidence_is_dropped_when_some_remain(ana):
    model = loaded(ana)
    (res,) = model.apply([AddClaim(page=ANA_KEY, section="Who", text="Has a cat.",
                                   evidence=["ghost", "wa-2"])],
                         existing_evidence=evidence_known("wa-2"))
    assert res.ok and model.claim(res.claim_id).evidence == ["wa-2"]


def test_duplicate_returns_existing_claim_id(ana):
    model = loaded(ana)
    (res,) = model.apply([AddClaim(page=ANA_KEY, section="Who",
                                   text="Works as a nurse, mostly night shifts", evidence=["x"])])
    assert res.reason == "duplicate" and res.claim_id == C_NURSE


def test_superseded_text_may_come_back(ana):
    model = loaded(ana)
    model.apply([SupersedeClaim(claim_id=C_NURSE, text="Works days now.", evidence=["x"])])
    (res,) = model.apply([AddClaim(page=ANA_KEY, section="Who",
                                   text="Works as a nurse, mostly night shifts.", evidence=["y"])])
    assert res.ok  # a claim in Earlier does not block saying it again


def test_supersede_of_earlier_claim_and_timeline_without_earlier_refused(self_dir):
    model = loaded(self_dir)
    model.apply([CreatePage(kind=PageKind.TIMELINE, slug="2014", title="2014")])
    (add,) = model.apply([AddClaim(page="timeline/2014", section="Events", text="Moved to Lyon.",
                                   evidence=["x"])])
    (res,) = model.apply([SupersedeClaim(claim_id=add.claim_id, text="y", evidence=["x"])])
    assert res.reason == "section"
    model.apply([CreatePage(kind=PageKind.TOPIC, slug="tea", title="Tea")])
    (a,) = model.apply([AddClaim(page="topics/tea", section="What I think", text="Tea is good.",
                                 evidence=["x"])])
    model.apply([SupersedeClaim(claim_id=a.claim_id, text="Tea is fine.", evidence=["x"])])
    (res,) = model.apply([SupersedeClaim(claim_id=a.claim_id, text="z", evidence=["x"])])
    assert res.reason == "section"


def test_pipeline_cannot_change_confirmed(ana):
    write_page(ana, ANA_KEY, ANA.replace("mostly night", "only night").replace(
        "## How we talk\n", "## How we talk\n- Owner wrote this one.\n"))
    model = SelfModel(ana)
    report = model.load()
    owner_added = report.added[0]
    model.confirm(C_LYON)  # through review
    model.save()
    page_before = read_page(ana, ANA_KEY)
    targets = [C_NURSE, C_LYON, owner_added]
    for cid in targets:
        for op in [
            ReviseClaim(claim_id=cid, text="Changed by the pipeline."),
            SupersedeClaim(claim_id=cid, text="Changed by the pipeline.", evidence=["x"]),
        ]:
            (res,) = model.apply([op])
            assert (res.ok, res.reason) == (False, "confirmed")
        (res,) = model.apply([SupportClaim(claim_id=cid, evidence=["wa-77"])])
        assert res.ok
        assert model.claim(cid).status is ClaimStatus.CONFIRMED
    model.save()
    assert read_page(ana, ANA_KEY) == page_before


def test_invalid_ops_never_raise(ana):
    model = loaded(ana)
    mutated = AddClaim(page=ANA_KEY, section="Who", text="ok", evidence=["x"])
    mutated.text = 42  # pydantic does not validate assignment
    weird = SupportClaim(claim_id=C_LYON, evidence=["x"])
    weird.evidence = None
    ops = [mutated, weird, 3.14, {}, [], {"op": "add", "page": None, "section": "Who",
                                          "text": "x", "evidence": ["x"]},
           AddClaim(page="../../etc/passwd", section="Who", text="x", evidence=["x"]),
           CreatePage(kind=PageKind.PERSON, slug="../../escape", title="T\nT")]
    results = model.apply(ops)
    assert len(results) == len(ops)
    assert all(r.reason != "error" for r in results)
    assert [r.ok for r in results[:-1]] == [False] * (len(ops) - 1)
    assert results[-1].ok  # slug is slugified: people/escape, title made single-line
    assert model.page("people/escape").title == "T T"
    model.save()
    assert not (ana.parent / "escape.md").exists()


def test_multiline_text_becomes_one_line(ana):
    model = loaded(ana)
    (res,) = model.apply([AddClaim(page=ANA_KEY, section="Who", text="Has\na cat.\r\n",
                                   evidence=["x"])])
    assert res.ok
    assert f"- Has a cat. ^{res.claim_id}\n" in model.read_page(ANA_KEY)


def test_add_to_catch_all_page(self_dir):
    from repartee.schema import CATCH_ALL_SLUG, page_path

    model = loaded(self_dir)
    key = page_path(PageKind.TOPIC, CATCH_ALL_SLUG)
    model.apply([CreatePage(kind=PageKind.TOPIC, slug=CATCH_ALL_SLUG, title="Misc")])
    (res,) = model.apply([AddClaim(page=key, section="What I know", text="x y", evidence=["e"])])
    assert res.ok and model.claim(res.claim_id).page == "topics/misc"
