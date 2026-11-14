"""Owner edits in the markdown are detected on load and respected (spec section 10.3)."""

from __future__ import annotations

import json

import pytest

from repartee.schema import AddClaim, ClaimOrigin, ClaimStatus, ReviseClaim, SupportClaim
from repartee.selfmodel import SelfModel, SelfModelError

from .conftest import (
    NURSE_LINE,
    NURSE_TEXT,
    ANA,
    ANA_KEY,
    C_LYON,
    C_NURSE,
    C_TALK,
    loaded,
    read_page,
    read_sidecar,
    write_page,
    write_sidecar,
)


def test_clean_load_reports_nothing_and_save_is_identical(ana):
    before = {p: p.read_bytes() for p in ana.rglob("*") if p.is_file()}
    model = SelfModel(ana)
    report = model.load()
    assert report.model_dump() == {k: [] for k in report.model_dump()}
    model.save()
    assert {p: p.read_bytes() for p in ana.rglob("*") if p.is_file()} == before
    assert [c.status for c in model.claims()] == [ClaimStatus.INFERRED] * 3


def test_owner_edit_confirms_claim(ana):
    write_page(ana, ANA_KEY, ANA.replace("mostly night shifts", "only night shifts"))
    model = SelfModel(ana)
    report = model.load()
    assert report.edited == [C_NURSE]
    claim = model.claim(C_NURSE)
    assert claim.status is ClaimStatus.CONFIRMED and claim.origin is ClaimOrigin.USER
    assert claim.text == "Works as a nurse, only night shifts."
    assert claim.evidence == ["wa-1"]  # evidence kept
    model.save()
    assert read_sidecar(ana, ANA_KEY)["claims"][C_NURSE]["status"] == "confirmed"
    assert loaded(ana).claim(C_NURSE).status is ClaimStatus.CONFIRMED
    assert SelfModel(ana).load().edited == []  # saved state is now the baseline


def test_whitespace_only_change_is_not_an_edit(ana):
    write_page(ana, ANA_KEY, ANA.replace("mostly night", "mostly  night").replace(
        "^c-77be01aa", " ^c-77be01aa  "))
    model = SelfModel(ana)
    assert model.load().edited == []
    assert model.claim(C_NURSE).status is ClaimStatus.INFERRED


def test_owner_added_item_gets_id_and_is_confirmed(ana):
    text = ANA.replace(
        "## How we talk\n",
        "## How we talk\n- We switch to Spanish when we argue.  \n",
    ).replace("## Notes\n", "## Notes\n- my own bullet, not a claim\n")
    text = text.replace("\n", "\r\n")
    write_page(ana, ANA_KEY, text)
    model = SelfModel(ana)
    report = model.load()
    assert len(report.added) == 1
    new_id = report.added[0]
    claim = model.claim(new_id)
    assert claim.status is ClaimStatus.CONFIRMED and claim.origin is ClaimOrigin.USER
    assert claim.section == "How we talk" and claim.evidence == []
    model.save()
    out = read_page(ana, ANA_KEY)
    # only change: the id appended (before the trailing spaces); CRLF preserved
    assert out == text.replace(
        "when we argue.  \r\n", f"when we argue. ^{new_id}  \r\n"
    )
    assert "- my own bullet, not a claim\r\n" in out
    assert SelfModel(ana).load().added == []


def test_owner_items_that_are_not_claims_are_untouched(ana):
    extra = (
        "  - nested under a claim\n"
        "1. numbered\n"
        "+ plus item\n"
        "- has my own block ref ^my-ref\n"
        "> - quoted item\n"
    )
    text = ANA.replace("## How we talk\n", extra + "\n## How we talk\n")
    write_page(ana, ANA_KEY, text)
    model = SelfModel(ana)
    assert model.load().added == []
    model.save()
    assert read_page(ana, ANA_KEY) == text


def test_owner_delete_becomes_rejected_and_is_not_readded(ana):
    write_page(ana, ANA_KEY, ANA.replace(NURSE_LINE, ""))
    model = SelfModel(ana)
    report = model.load()
    assert report.rejected == [C_NURSE]
    assert model.claim(C_NURSE) is None
    model.save()
    rej = read_sidecar(ana, ANA_KEY)["rejected"]
    assert rej[0]["text"] == "Works as a nurse, mostly night shifts."
    assert rej[0]["evidence"] == ["wa-1"] and rej[0]["id"] == C_NURSE and rej[0]["norm_hash"]

    model = loaded(ana)
    for text in ["Works as a nurse, mostly night shifts.", "works as a NURSE mostly night shifts!"]:
        (res,) = model.apply([AddClaim(page=ANA_KEY, section="Who", text=text, evidence=["x"])])
        assert not res.ok and res.reason == "matches_rejected"
    (res,) = model.apply([ReviseClaim(claim_id=C_LYON, text=NURSE_TEXT)])
    assert res.reason == "matches_rejected"
    (res,) = model.apply([SupportClaim(claim_id=C_NURSE, evidence=["x"])])
    assert res.reason == "rejected"
    # deleting again on a later load does not duplicate the record
    model.save()
    assert len(read_sidecar(ana, ANA_KEY)["rejected"]) == 1


def test_cut_then_paste_across_loads_restores_claim(ana):
    """Obsidian autosaves mid cut-and-paste; a load in between must not lose the claim."""
    line = NURSE_LINE
    write_page(ana, ANA_KEY, ANA.replace(line, ""))
    model = SelfModel(ana)
    model.load()
    model.save()
    write_page(ana, ANA_KEY, ANA.replace(line, "").replace("## Notes\n", "## Notes\n" + line))
    model = SelfModel(ana)
    report = model.load()
    assert report.restored == [C_NURSE]
    claim = model.claim(C_NURSE)
    assert claim.status is ClaimStatus.INFERRED and claim.evidence == ["wa-1"]
    assert claim.section == "Notes"
    model.save()
    assert read_sidecar(ana, ANA_KEY)["rejected"] == []


def test_owner_move_between_pages(ana):
    line = NURSE_LINE
    write_page(ana, ANA_KEY, ANA.replace(line, ""))
    other = "# Work\n\n## What I know\n" + line
    write_page(ana, "topics/work", other)
    model = SelfModel(ana)
    report = model.load()
    assert report.moved == [C_NURSE] and report.rejected == [] and report.edited == []
    claim = model.claim(C_NURSE)
    assert claim.page == "topics/work" and claim.section == "What I know"
    assert claim.status is ClaimStatus.INFERRED and claim.evidence == ["wa-1"]
    model.save()
    assert read_page(ana, "topics/work") == other  # the moved line is not rewritten
    assert C_NURSE not in read_sidecar(ana, ANA_KEY)["claims"]
    assert C_NURSE in read_sidecar(ana, "topics/work")["claims"]
    assert SelfModel(ana).load().moved == []


def test_move_within_page_updates_section(ana):
    line = NURSE_LINE
    moved = ANA.replace(line, "").replace("teasing. ^c-0d4e9a13\n", "teasing. ^c-0d4e9a13\n" + line)
    write_page(ana, ANA_KEY, moved)
    model = SelfModel(ana)
    report = model.load()
    assert report.moved == [C_NURSE]
    assert model.claim(C_NURSE).section == "How we talk"
    assert model.claim(C_NURSE).status is ClaimStatus.INFERRED


def test_copy_paste_duplicate_gets_new_id(ana):
    line = NURSE_LINE
    write_page(ana, ANA_KEY, ANA.replace("## Notes\n", "## Notes\n" + line))
    write_page(ana, "topics/work", "## What I know\n" + line)
    model = SelfModel(ana)
    report = model.load()
    assert len(report.added) == 2 and report.moved == []
    assert model.claim(C_NURSE).section == "Who"  # the original keeps its place
    copies = [model.claim(c) for c in report.added]
    assert all(c.status is ClaimStatus.CONFIRMED and c.evidence == ["wa-1"] for c in copies)
    model.save()
    assert read_page(ana, ANA_KEY).count("^c-77be01aa") == 1
    assert "^c-77be01aa" not in read_page(ana, "topics/work")
    assert SelfModel(ana).load().model_dump() == loaded_empty_report()


def loaded_empty_report():
    from repartee.selfmodel import ReconcileReport

    return ReconcileReport().model_dump()


def test_unknown_id_adopted_as_owner_claim(self_dir):
    write_page(self_dir, "topics/tea", "## What I think\n- Green tea is best cold. ^c-abcdef12\n")
    model = SelfModel(self_dir)
    report = model.load()
    assert report.added == ["c-abcdef12"]
    claim = model.claim("c-abcdef12")
    assert claim.status is ClaimStatus.CONFIRMED and claim.origin is ClaimOrigin.USER
    model.save()
    assert read_page(self_dir, "topics/tea").endswith("^c-abcdef12\n")


def test_deleted_page_rejects_its_claims(ana):
    (ana / f"{ANA_KEY}.md").unlink()
    model = SelfModel(ana)
    assert sorted(model.load().rejected) == sorted([C_LYON, C_NURSE, C_TALK])
    model.save()
    model = loaded(ana)
    model.apply([{"op": "create_page", "kind": "person", "slug": "ana-ruiz", "title": "Ana Ruiz"}])
    (res,) = model.apply([AddClaim(page=ANA_KEY, section="Who", text="Works as a nurse, mostly "
                                   "night shifts.", evidence=["x"])])
    assert res.reason == "matches_rejected"


def test_renamed_page_keeps_claims_and_rejections(ana):
    line = NURSE_LINE
    write_page(ana, ANA_KEY, ANA.replace(line, ""))
    loaded(ana).save()  # nurse claim rejected on the old page
    (ana / f"{ANA_KEY}.md").rename(ana / "people" / "ana.md")
    model = SelfModel(ana)
    report = model.load()
    assert sorted(report.moved) == sorted([C_LYON, C_TALK]) and report.rejected == []
    model.save()
    assert not (ana / ".claims" / f"{ANA_KEY}.json").exists()
    (res,) = loaded(ana).apply([AddClaim(page="people/ana", section="Who", text="Works as a "
                                         "nurse, mostly night shifts.", evidence=["x"])])
    assert res.reason == "matches_rejected"


def test_nested_claim_is_kept_but_pipeline_cannot_edit_it(ana):
    write_page(ana, ANA_KEY, ANA.replace("- Works as", "  - Works as"))
    model = SelfModel(ana)
    report = model.load()
    assert report.rejected == [] and report.edited == []
    (res,) = model.apply([ReviseClaim(claim_id=C_NURSE, text="Something else.")])
    assert not res.ok and res.reason == "section"
    (res,) = model.apply([SupportClaim(claim_id=C_NURSE, evidence=["wa-2"])])
    assert res.ok


def test_emptied_claim_text_counts_as_deleted(ana):
    write_page(ana, ANA_KEY, ANA.replace("- Works as a nurse, mostly night shifts. ^c", "- ^c"))
    model = SelfModel(ana)
    assert model.load().rejected == [C_NURSE]
    model.save()
    assert "- ^c-77be01aa" in read_page(ana, ANA_KEY)  # the owner's line itself is left alone


def test_claim_moved_into_code_block_is_rejected_then_restored(ana):
    fenced = ANA.replace(
        NURSE_LINE,
        "```\n- Works as a nurse, mostly night shifts. ^c-77be01aa\n```\n",
    )
    write_page(ana, ANA_KEY, fenced)
    model = SelfModel(ana)
    assert model.load().rejected == [C_NURSE]
    model.save()
    assert read_page(ana, ANA_KEY) == fenced
    write_page(ana, ANA_KEY, ANA)
    assert SelfModel(ana).load().restored == [C_NURSE]


def test_corrupt_sidecar_raises_clear_error(ana):
    (ana / ".claims" / f"{ANA_KEY}.json").write_text("<<<<<<< HEAD\n{", encoding="utf-8")
    with pytest.raises(SelfModelError, match="ana-ruiz.json"):
        SelfModel(ana).load()


def test_sidecar_with_bad_field_raises(ana):
    write_sidecar(ana, ANA_KEY, {C_LYON: ("Who", "x")})
    data = json.loads((ana / ".claims" / f"{ANA_KEY}.json").read_text())
    data["claims"][C_LYON]["confidence"] = "very"
    (ana / ".claims" / f"{ANA_KEY}.json").write_text(json.dumps(data))
    with pytest.raises(SelfModelError):
        SelfModel(ana).load()


def test_owner_text_outside_claims_survives_pipeline_ops(ana):
    text = ANA.replace(
        NURSE_LINE,
        "Owner paragraph between claims.\n\n" + NURSE_LINE
        + "  - owner nested note under a claim\n",
    ).replace("## Notes\n", "## Notes\n- list item in Notes\n  - nested in Notes\n| t | t |\n")
    write_page(ana, ANA_KEY, text)
    model = SelfModel(ana)
    model.load()
    results = model.apply([
        AddClaim(page=ANA_KEY, section="Who", text="Has a cat.", evidence=["wa-9"]),
        ReviseClaim(claim_id=C_LYON, text="Friend from university; we met in Lyon."),
        {"op": "supersede", "claim_id": C_NURSE, "text": "Works days now.", "evidence": ["wa-8"]},
        AddClaim(page=ANA_KEY, section="Relationship", text="Close friend.", evidence=["wa-7"]),
    ])
    assert all(r.ok for r in results), results
    model.save()
    out = read_page(ana, ANA_KEY)
    owner_lines = [
        ln for ln in text.splitlines()
        if "^c-" not in ln
    ]
    remaining = out.splitlines()
    for ln in owner_lines:
        assert ln in remaining, ln
    # nested note travelled with its claim to Earlier
    earlier = out[out.index("## Earlier"):]
    assert earlier.startswith(
        "## Earlier\n- Works as a nurse, mostly night shifts. ^c-77be01aa\n"
        "  - owner nested note under a claim\n"
    )
    assert out.endswith("## Notes\n- list item in Notes\n  - nested in Notes\n| t | t |\n"
                        "Anything I write here myself. The pipeline never touches it.\n")
    assert SelfModel(ana).load().model_dump() == loaded_empty_report()


def test_non_utf8_bytes_round_trip(ana):
    raw = ANA.encode("utf-8").replace(b"Anything I write", b"Anything \xff\xfe I write")
    write_page(ana, ANA_KEY, raw)
    model = SelfModel(ana)
    model.load()
    model.apply([AddClaim(page=ANA_KEY, section="Who", text="Has a cat.", evidence=["wa-9"])])
    model.save()
    assert b"Anything \xff\xfe I write" in (ana / f"{ANA_KEY}.md").read_bytes()


def test_hidden_folders_are_ignored(ana):
    write_page(ana, ".trash/people/old", "## Who\n- a ^c-77be01aa\n")
    write_page(ana, ".obsidian/x", "## Who\n- b\n")
    model = SelfModel(ana)
    assert model.load().model_dump() == loaded_empty_report()
    assert [p.key for p in model.pages()] == [ANA_KEY]
