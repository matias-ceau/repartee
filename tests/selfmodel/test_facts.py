"""Key facts: frontmatter properties with a status in the sidecar, set and retracted by ops.

Everyone here is invented.
"""

from __future__ import annotations

from datetime import date

from repartee.schema import (
    AddClaim,
    ClaimOrigin,
    ClaimStatus,
    CreatePage,
    PageKind,
    RetractClaim,
    SetFact,
)
from repartee.selfmodel import SelfModel
from repartee.selfmodel.ops import CONFIRMED, MATCHES_REJECTED, UNBACKED
from repartee.selfmodel.pages import Page, property_value, value_key

from .conftest import loaded, read_page, read_sidecar, write_page

NOOR = "people/noor-haddad"
PIPPA = "people/pippa"

NOOR_PAGE = """---
kind: person
title: Noor Haddad
aliases: [Noor]
updated: 2026-11-14
ntype: person
relation: friend
children:
  - "[[Theo Haddad]]"
recommended_by: "[[Lior Benami]]"
---
# Noor Haddad

## Who
- Noor lives in Ghent. ^c-11111111

## Notes
My own words.
"""


def _owner_page(self_dir) -> SelfModel:
    write_page(self_dir, NOOR, NOOR_PAGE)
    model = SelfModel(self_dir)
    model.load()
    model.save()
    return model


def _statements(*ids: str):
    owned = set(ids)
    return lambda evidence: {e for e in evidence if e in owned}


# ---------------------------------------------------------------- page properties


def test_property_values_are_normalized():
    assert property_value(date(2019, 4, 2)) == "2019-04-02"
    assert property_value([["Theo"]]) == "[[Theo]]"  # `key: [[Theo]]` written unquoted
    assert property_value([["Theo"]], multi=True) == ["[[Theo]]"]
    assert property_value("Theo", multi=True) == ["Theo"]
    assert property_value(["cat"], multi=False) == "cat"
    assert property_value("") is None and property_value([]) is None
    assert value_key("[[noor-haddad|Noor Haddad]]") == value_key("noor haddad")


def test_properties_skip_page_metadata_and_keep_unknown_ones():
    page = Page.parse(NOOR, NOOR_PAGE)
    assert page.properties() == {
        "ntype": "person", "relation": "friend", "children": ["[[Theo Haddad]]"],
        "recommended_by": "[[Lior Benami]]"}


def test_set_property_touches_only_its_lines():
    page = Page.parse(NOOR, NOOR_PAGE)
    page.set_property("relation", "sister")
    page.set_property("children", ["[[Theo Haddad]]", "Mila"])
    page.set_property("born", "1990-05-03")
    text = page.render()
    assert "relation: sister\nchildren:\n  - '[[Theo Haddad]]'\n  - Mila\n" in text
    assert "born: 1990-05-03\n---\n" in text
    assert text.replace("relation: sister", "relation: friend").replace(
        "  - '[[Theo Haddad]]'\n  - Mila\n", '  - "[[Theo Haddad]]"\n').replace(
        "born: 1990-05-03\n", "") == NOOR_PAGE
    assert Page.parse(NOOR, text).properties()["born"] == "1990-05-03"


def test_load_and_save_with_properties_is_byte_identical(self_dir):
    _owner_page(self_dir)
    before = {p: p.read_bytes() for p in self_dir.rglob("*") if p.is_file()}
    model = SelfModel(self_dir)
    report = model.load()
    assert report.model_dump() == {k: [] for k in report.model_dump()}
    model.save()
    assert {p: p.read_bytes() for p in self_dir.rglob("*") if p.is_file()} == before
    assert read_page(self_dir, NOOR) == NOOR_PAGE


# ----------------------------------------------------------------- reconciliation


def test_owner_written_properties_are_confirmed_facts(self_dir):
    write_page(self_dir, NOOR, NOOR_PAGE)
    model = SelfModel(self_dir)
    report = model.load()
    assert sorted(report.facts_added) == [f"{NOOR}:{s}" for s in
                                          ("children", "ntype", "recommended_by", "relation")]
    facts = {f.slot: f for f in model.facts(page=NOOR)}
    assert list(facts) == ["ntype", "relation", "children", "recommended_by"]  # registry first
    assert all(f.status == ClaimStatus.CONFIRMED and f.origin == ClaimOrigin.USER
               for f in facts.values())
    assert facts["recommended_by"].value == "[[Lior Benami]]"  # unknown, kept verbatim
    model.save()
    sidecar = read_sidecar(self_dir, NOOR)
    assert sidecar["facts"]["relation"]["value"] == "friend"
    assert "relation" not in read_page(self_dir, NOOR).split("---")[2]  # status not in page


def test_owner_edit_of_an_inferred_property_confirms_it(self_dir):
    model = _owner_page(self_dir)
    model.apply([SetFact(page=NOOR, slot="species", value="human", evidence=["wa-1"])])
    model.save()
    assert loaded(self_dir).fact(NOOR, "species").status == ClaimStatus.INFERRED

    text = read_page(self_dir, NOOR).replace("species: human", "species: cat")
    write_page(self_dir, NOOR, text)
    model = SelfModel(self_dir)
    report = model.load()
    assert report.facts_edited == [f"{NOOR}:species"]
    fact = model.fact(NOOR, "species")
    assert fact.value == "cat" and fact.status == ClaimStatus.CONFIRMED
    assert fact.origin == ClaimOrigin.USER and fact.evidence == ["wa-1"]


def test_a_respelled_value_is_not_an_edit(self_dir):
    _owner_page(self_dir)
    model = loaded(self_dir)
    model.apply([SetFact(page=NOOR, slot="born", value="1990", evidence=["wa-1"])])
    model.save()
    write_page(self_dir, NOOR, read_page(self_dir, NOOR).replace(
        '  - "[[Theo Haddad]]"', "  - Theo Haddad"))
    model = SelfModel(self_dir)
    assert model.load().facts_edited == []


def test_deleting_an_inferred_property_rejects_it_for_good(self_dir):
    model = _owner_page(self_dir)
    model.apply([SetFact(page=NOOR, slot="species", value="dog", evidence=["wa-1"])])
    model.save()
    write_page(self_dir, NOOR, read_page(self_dir, NOOR).replace("species: dog\n", ""))

    model = SelfModel(self_dir)
    report = model.load()
    assert report.facts_rejected == [f"{NOOR}:species"]
    assert model.fact(NOOR, "species") is None
    (result,) = model.apply([SetFact(page=NOOR, slot="species", value="Dog",
                                     evidence=["wa-2"])])
    assert not result.ok and result.reason == MATCHES_REJECTED
    assert "species" not in read_page(self_dir, NOOR)
    model.save()
    assert read_sidecar(self_dir, NOOR)["rejected_facts"][0]["value"] == "dog"
    # another value is still possible
    (result,) = loaded(self_dir).apply([SetFact(page=NOOR, slot="species", value="cat",
                                                evidence=["wa-2"])])
    assert result.ok


def test_broken_frontmatter_keeps_facts_as_they_were(self_dir):
    _owner_page(self_dir)
    write_page(self_dir, NOOR, NOOR_PAGE.replace("relation: friend", "relation: [friend"))
    model = SelfModel(self_dir)
    report = model.load()
    assert report.facts_rejected == [] and model.fact(NOOR, "relation").value == "friend"
    (result,) = model.apply([SetFact(page=NOOR, slot="born", value="1990", evidence=["x"])])
    assert not result.ok


# ------------------------------------------------------------------------- set_fact


def test_set_fact_from_chats_is_inferred_and_from_an_interview_confirmed(self_dir):
    model = SelfModel(self_dir)
    model.load()
    model.apply([CreatePage(kind=PageKind.PERSON, slug="pippa", title="Pippa")])
    results = model.apply(
        [SetFact(page=PIPPA, slot="species", value="dog", evidence=["wa-1"], confidence=0.6),
         SetFact(page=PIPPA, slot="ntype", value="pet", evidence=["iv-1"]),
         SetFact(page=PIPPA, slot="mood", value="grumpy", evidence=["wa-1"])],
        owner_statements=_statements("iv-1"))
    assert [r.ok for r in results] == [True, True, False]  # mood is not a known slot
    species, ntype = model.fact(PIPPA, "species"), model.fact(PIPPA, "ntype")
    assert species.status == ClaimStatus.INFERRED and species.confidence == 0.6
    assert ntype.status == ClaimStatus.CONFIRMED and ntype.origin == ClaimOrigin.INTERVIEW
    assert "species: dog\nntype: pet\n---" in model.read_page(PIPPA)

    # the owner says it in an interview: the inferred value is replaced and confirmed
    (result,) = model.apply([SetFact(page=PIPPA, slot="species", value="cat",
                                     evidence=["iv-2"])], owner_statements=_statements("iv-2"))
    assert result.ok
    assert model.fact(PIPPA, "species").status == ClaimStatus.CONFIRMED
    assert "species: cat\n" in model.read_page(PIPPA)


def test_set_fact_never_overwrites_a_confirmed_fact(self_dir):
    model = _owner_page(self_dir)
    (result,) = model.apply([SetFact(page=NOOR, slot="relation", value="partner",
                                     evidence=["wa-1"])])
    assert not result.ok and result.reason == CONFIRMED and result.claim_id.startswith("q-")
    assert model.fact(NOOR, "relation").value == "friend"
    assert "partner" in model.questions()[0].text
    (same,) = model.apply([SetFact(page=NOOR, slot="relation", value="Friend",
                                   evidence=["wa-1"])])
    assert same.ok and model.fact(NOOR, "relation").evidence == ["wa-1"]


def test_link_slots_name_existing_pages(self_dir):
    model = _owner_page(self_dir)
    model.apply([CreatePage(kind=PageKind.PERSON, slug="pippa", title="Pippa"),
                 CreatePage(kind=PageKind.PERSON, slug="sam-okafor", title="Sam Okafor")])
    model.apply([SetFact(page=PIPPA, slot="owners", value="Noor", evidence=["wa-1"]),
                 SetFact(page=PIPPA, slot="owners", value="Sam Okafor", evidence=["wa-2"]),
                 SetFact(page=PIPPA, slot="owners", value="Ines", evidence=["wa-3"])])
    fact = model.fact(PIPPA, "owners")
    assert fact.value == ["[[noor-haddad|Noor Haddad]]", "[[sam-okafor|Sam Okafor]]", "Ines"]
    assert fact.evidence == ["wa-1", "wa-2", "wa-3"]


# -------------------------------------------------------------------------- retract


def _with_claims(self_dir) -> tuple[SelfModel, str, str]:
    model = _owner_page(self_dir)
    model.apply([CreatePage(kind=PageKind.TIMELINE, slug="2023", title="2023")])
    partner, trip = (r.claim_id for r in model.apply([
        AddClaim(page=NOOR, section="Relationship", text="Noor is my partner.",
                 evidence=["wa-1"]),
        AddClaim(page="timeline/2023", section="Events", text="I went to Ghent with Noor.",
                 evidence=["wa-2"])]))
    return model, partner, trip


def test_retract_needs_owner_backing(self_dir):
    model, partner, _ = _with_claims(self_dir)
    refused = model.apply([
        RetractClaim(claim_id=partner, reason="she is my sister", evidence=["wa-9"]),
        RetractClaim(claim_id=partner, reason="x", fact_page=NOOR, fact_slot="born"),
    ], owner_statements=_statements("iv-1"))
    assert [(r.ok, r.reason) for r in refused] == [(False, UNBACKED), (False, UNBACKED)]
    assert model.claim(partner) is not None

    (ok,) = model.apply([RetractClaim(claim_id=partner, reason="she is my sister",
                                      evidence=["wa-9", "iv-1"])],
                        owner_statements=_statements("iv-1"))
    assert ok.ok and model.claim(partner) is None
    assert "Noor is my partner" not in model.read_page(NOOR)
    model.save()
    (record,) = read_sidecar(self_dir, NOOR)["rejected"]
    assert record["retracted"] and record["reason"] == "she is my sister"
    assert record["retract_evidence"] == ["wa-9", "iv-1"]

    # the same text is not added again
    (again,) = loaded(self_dir).apply([AddClaim(page=NOOR, section="Who",
                                                text="Noor is my partner", evidence=["wa-3"])])
    assert not again.ok and again.reason == MATCHES_REJECTED


def test_retract_backed_by_a_confirmed_fact_must_concern_its_subject(self_dir):
    model, partner, trip = _with_claims(self_dir)
    model.apply([CreatePage(kind=PageKind.PERSON, slug="pippa", title="Pippa")])
    model.apply([SetFact(page=PIPPA, slot="species", value="cat", evidence=["iv-1"])],
                owner_statements=_statements("iv-1"))
    unrelated, by_page, elsewhere = model.apply([
        RetractClaim(claim_id=partner, reason="x", fact_page=PIPPA, fact_slot="species"),
        RetractClaim(claim_id=partner, reason="she is my sister", fact_slot="relation"),
        RetractClaim(claim_id=trip, reason="a test", fact_page=NOOR, fact_slot="relation"),
    ])
    assert (unrelated.ok, unrelated.reason) == (False, UNBACKED)
    assert by_page.ok and elsewhere.ok  # the trip names Noor


def test_retract_on_a_confirmed_claim_becomes_a_question(self_dir):
    model, partner, _ = _with_claims(self_dir)
    model.confirm(partner)
    (result,) = model.apply([RetractClaim(claim_id=partner, reason="she is my sister",
                                          fact_slot="relation")])
    assert not result.ok and result.reason == CONFIRMED and result.claim_id.startswith("q-")
    assert model.claim(partner) is not None
    assert 'You confirmed "Noor is my partner."' in model.questions()[0].text


def test_owner_review_of_facts(self_dir):
    model = _owner_page(self_dir)
    model.apply([SetFact(page=NOOR, slot="born", value="1990", evidence=["wa-1"]),
                 SetFact(page=NOOR, slot="species", value="dog", evidence=["wa-1"])])
    model.confirm_fact(NOOR, "born")
    model.reject_fact(NOOR, "species")
    model.save()
    model = loaded(self_dir)
    assert model.fact(NOOR, "born").status == ClaimStatus.CONFIRMED
    assert model.fact(NOOR, "species") is None and "species" not in read_page(self_dir, NOOR)
    assert model.fact_rejected(NOOR, "species", "dog")
