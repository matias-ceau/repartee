"""Randomised attack: owner clutter plus random operations must never lose owner text."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from repartee.schema import (
    AddClaim,
    AskQuestion,
    CreatePage,
    PageKind,
    ReviseClaim,
    SupersedeClaim,
    SupportClaim,
)
from repartee.selfmodel import ReconcileReport, SelfModel
from repartee.selfmodel.pages import Page

from .conftest import evidence_known, read_page, write_page

OWNER_FRAGMENTS = [
    ["Owner paragraph with *emphasis*."],
    ["  - nested owner note"],
    ["    - deeper owner note  "],
    ["1. numbered owner item", "2. second"],
    ["+ plus item"],
    ["```", "- fake claim ^c-deadbeef", "## fake heading", "```"],
    ["~~~python", "x = '- y ^c-0badf00d'", "~~~"],
    ["| col | col |", "|-----|-----|", "| a ^c-11112222 | b |"],
    ["> quoted - not a claim"],
    ["<!-- owner comment -->"],
    ["%%", "- hidden owner item", "%%"],
    ["", ""],
    ["trailing tab\t"],
    ["Ünïcødé 🙂 שלום"],
    ["- owner item with own ref ^ref-1"],
    ["### owner subheading"],
]
SECTIONS = ["Who", "Relationship", "How we talk", "Shared history", "Earlier", "Notes", "Nope"]
TEXTS = ["Likes hiking.", "Lives in Porto.", "Prefers tea.", "Has two brothers.",
         "Works at a bakery.", "Plays the oud.", "Hates mornings."]


def _owner_lines(text: str) -> Counter:
    return Counter(ln for ln in text.replace("\r\n", "\n").split("\n") if "^c-" not in ln)


@pytest.mark.parametrize("seed", range(40))
def test_random_clutter_and_ops_keep_owner_text(tmp_path, seed):
    rng = random.Random(seed)
    self_dir = tmp_path / "self"
    model = SelfModel(self_dir)
    model.load()
    model.apply([CreatePage(kind=PageKind.PERSON, slug="ana", title="Ana")])
    for text in rng.sample(TEXTS, 4):
        model.apply([AddClaim(page="people/ana", section=rng.choice(SECTIONS[:4]), text=text,
                              evidence=["e1"])])
    model.save()

    # the owner clutters the page in Obsidian
    eol = rng.choice(["\n", "\r\n"])
    lines = read_page(self_dir, "people/ana").split("\n")
    lines = lines[:-1] if lines[-1] == "" else lines
    front_end = lines.index("---", 1) + 1
    positions = sorted((rng.randint(front_end, len(lines)) for _ in range(rng.randint(1, 8))),
                       reverse=True)
    for pos in positions:  # never inside an earlier fragment
        lines[pos:pos] = rng.choice(OWNER_FRAGMENTS)
    text = eol.join(lines) + rng.choice([eol, ""])
    write_page(self_dir, "people/ana", text)
    assert Page.parse("people/ana", text).render() == text

    model = SelfModel(self_dir)
    report = model.load()
    assert report.rejected == [] and report.edited == [] and report.added == []
    owner_before = _owner_lines(text)

    ids = [c.id for c in model.claims()] + ["c-00000000", "junk"]
    ops = []
    for _ in range(rng.randint(5, 25)):
        kind = rng.choice(["add", "support", "revise", "supersede", "question"])
        ev = rng.sample(["e1", "e2", "e3", "ghost"], rng.randint(0, 3))
        text_ = rng.choice(TEXTS + ["New fact.", "Another\nline", ""])
        if kind == "add":
            ops.append(AddClaim(page=rng.choice(["people/ana", "people/x"]),
                                section=rng.choice(SECTIONS), text=text_, evidence=ev))
        elif kind == "support":
            ops.append(SupportClaim(claim_id=rng.choice(ids), evidence=ev))
        elif kind == "revise":
            ops.append(ReviseClaim(claim_id=rng.choice(ids), text=text_ + " (rev)", evidence=ev))
        elif kind == "supersede":
            ops.append(SupersedeClaim(claim_id=rng.choice(ids), text=text_ + " (new)",
                                      evidence=ev))
        else:
            ops.append(AskQuestion(text=text_ or "?", page="people/ana"))
    results = model.apply(ops, existing_evidence=evidence_known("e1", "e2", "e3"))
    assert len(results) == len(ops)
    assert all(r.reason != "error" for r in results), results
    model.save()

    after = read_page(self_dir, "people/ana")
    missing = owner_before - _owner_lines(after)
    assert not missing, missing
    assert Page.parse("people/ana", after).render() == after

    model2 = SelfModel(self_dir)
    report2 = model2.load()
    assert report2 == ReconcileReport(), report2
    for claim in model2.claims():
        assert f"{claim.text} ^{claim.id}" in after or claim.page != "people/ana"
    before_bytes = {p: p.read_bytes() for p in self_dir.rglob("*") if p.is_file()}
    model2.save()
    assert {p: p.read_bytes() for p in self_dir.rglob("*") if p.is_file()} == before_bytes
