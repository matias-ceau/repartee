"""SelfModel services: evidence upkeep, questions, core, voice, resolve, saving."""

from __future__ import annotations

import os

import pytest

from repartee.schema import AddClaim, ClaimOrigin, ClaimStatus, CreatePage, PageKind
from repartee.selfmodel import SelfModel, SelfModelConflict

from .conftest import (
    ANA,
    ANA_KEY,
    C_LYON,
    C_NURSE,
    C_TALK,
    NURSE_LINE,
    loaded,
    read_page,
    read_sidecar,
    write_page,
)


def test_empty_or_missing_dir(tmp_path):
    model = SelfModel(tmp_path / "nope")
    assert model.load().model_dump()["added"] == []
    assert model.pages() == [] and model.claims() == [] and model.core_text() == ""
    assert model.questions() == [] and model.voice_registers() == []
    model.save()
    assert not (tmp_path / "nope").exists()


def test_claims_listing_and_filters(ana):
    model = loaded(ana)
    model.confirm(C_TALK)
    assert [c.id for c in model.claims()] == [C_LYON, C_NURSE, C_TALK]
    assert [c.id for c in model.claims(status=ClaimStatus.CONFIRMED)] == [C_TALK]
    assert [c.id for c in model.claims(page="topics/x")] == []
    copy = model.claim(C_LYON)
    copy.text = "mutated"
    assert model.claim(C_LYON).text != "mutated"  # callers get copies


def test_review_actions_confirm_edit_reject(ana):
    model = loaded(ana)
    model.confirm(C_LYON)
    assert model.claim(C_LYON).status is ClaimStatus.CONFIRMED
    assert model.claim(C_LYON).origin is ClaimOrigin.PIPELINE
    model.edit(C_NURSE, "Works as a nurse,\nday shifts now.")
    assert model.claim(C_NURSE).origin is ClaimOrigin.USER
    model.reject(C_TALK)
    model.save()
    text = read_page(ana, ANA_KEY)
    assert "- Works as a nurse, day shifts now. ^c-77be01aa\n" in text
    assert C_TALK not in text
    assert read_sidecar(ana, ANA_KEY)["rejected"][0]["id"] == C_TALK
    assert SelfModel(ana).load().model_dump()["rejected"] == []
    with pytest.raises(KeyError):
        model.confirm("c-00000000")


def test_remap_evidence(ana):
    write_page(ana, ANA_KEY, ANA.replace(NURSE_LINE, ""))
    model = loaded(ana)
    model.apply([AddClaim(page=ANA_KEY, section="Who", text="Has a cat.",
                          evidence=["nt-old", "nt-new"])])
    n = model.remap_evidence({"wa-1": "wa-9", "nt-old": "nt-new"})
    # wa-1 on two live claims and on the rejected record; nt-old merged into nt-new
    assert n == 4
    assert model.claim(C_LYON).evidence == ["wa-9"]
    cat = [c for c in model.claims() if c.text == "Has a cat."][0]
    assert cat.evidence == ["nt-new"]
    model.save()
    assert read_sidecar(ana, ANA_KEY)["rejected"][0]["evidence"] == ["wa-9"]


def test_prune_dead_evidence_removes_orphan_inferred_keeps_confirmed(ana):
    model = loaded(ana)
    model.confirm(C_TALK)
    model.apply([AddClaim(page=ANA_KEY, section="Who", text="Has a cat.",
                          evidence=["wa-1", "wa-2"])])
    report = model.prune_dead_evidence(lambda ids: {i for i in ids if i == "wa-2"})
    assert report.evidence_dropped == 4
    assert sorted(report.claims_removed) == sorted([C_LYON, C_NURSE])
    talk = model.claim(C_TALK)
    assert talk.evidence == [] and talk.status is ClaimStatus.CONFIRMED
    model.save()
    text = read_page(ana, ANA_KEY)
    assert C_LYON not in text and C_NURSE not in text and "Has a cat." in text
    assert "Anything I write here myself." in text
    # forgetting is not rejecting: the pipeline may learn it again from new material
    assert read_sidecar(ana, ANA_KEY)["rejected"] == []
    assert SelfModel(ana).load().model_dump()["rejected"] == []


def test_questions_add_dedupe_resolve(self_dir):
    model = loaded(self_dir)
    q1 = model.add_question("Where did we meet?", page="people/ana-ruiz")
    q2 = model.add_question("What music did I like at 15?")
    assert q1.id.startswith("q-") and q1.page == "people/ana-ruiz"
    assert model.add_question("where did we MEET").id == q1.id
    model.resolve_question(q1.id)
    assert [q.id for q in model.questions()] == [q2.id]
    assert [q.resolved for q in model.questions(include_resolved=True)] == [True, False]
    model.save()
    text = read_page(self_dir, "open-questions")
    assert f"- [x] Where did we meet? (page: people/ana-ruiz) ^{q1.id}" in text
    assert f"- [ ] What music did I like at 15? ^{q2.id}" in text
    # owner adds a question by hand, and resolves one by ticking it
    text = text.replace("- [ ] What music", "- [x] What music") + "- [ ] Owner's own question\n"
    write_page(self_dir, "open-questions", text)
    model = loaded(self_dir)
    qs = model.questions(include_resolved=True)
    assert [q.resolved for q in qs] == [True, True, False]
    assert qs[2].text == "Owner's own question" and qs[2].id.startswith("q-")
    with pytest.raises(KeyError):
        model.resolve_question("q-00000000")


def test_core_text_combines_pinned_and_generated(self_dir):
    model = loaded(self_dir)
    assert model.core_text() == ""
    model.set_core_generated("- I am a cartographer.\n- I live near the sea.")
    model.save()
    text = read_page(self_dir, "core").replace("## Pinned\n", "## Pinned\nI never lie.\r\n\n")
    write_page(self_dir, "core", text)
    model = loaded(self_dir)
    assert model.core_text() == "I never lie.\n\n- I am a cartographer.\n- I live near the sea."
    model.set_core_generated("- Short.")
    model.save()
    assert "I never lie." in read_page(self_dir, "core")
    assert loaded(self_dir).core_text() == "I never lie.\n\n- Short."


def test_voice_card_preserves_notes(self_dir):
    model = loaded(self_dir)
    model.set_voice_card("whatsapp.fr", "Short messages, lowercase.")
    model.save()
    path = self_dir / "voice" / "whatsapp.fr.md"
    text = path.read_text(encoding="utf-8").replace("## Notes\n", "## Notes\nNever use '!'.\n")
    path.write_text(text, encoding="utf-8")
    model = loaded(self_dir)
    model.set_voice_card("whatsapp.fr", "Very short messages.")
    model.save()
    card = loaded(self_dir).voice_card("whatsapp.fr")
    assert "Very short messages." in card and "Never use '!'." in card
    assert "Short messages, lowercase." not in card
    assert "---" not in card.splitlines()[0]  # no frontmatter
    assert loaded(self_dir).voice_registers() == ["whatsapp.fr"]
    assert model.voice_card("essay.en") is None
    with pytest.raises(ValueError):
        model.set_voice_card("../evil", "x")


def test_resolve_by_alias_and_close_match(ana):
    model = loaded(ana)
    model.apply([CreatePage(kind=PageKind.PERSON, slug="bo-lind", title="Bo Lind")])
    model.apply([CreatePage(kind=PageKind.TOPIC, slug="urban-cycling", title="Urban cycling")])
    assert model.resolve(PageKind.PERSON, "ana-ruiz") == ANA_KEY
    assert model.resolve(PageKind.PERSON, "ANA RUIZ") == ANA_KEY
    assert model.resolve(PageKind.PERSON, "anita") == ANA_KEY
    assert model.resolve(PageKind.PERSON, "Anna Ruiz") == ANA_KEY
    assert model.resolve(PageKind.TOPIC, "urban cyclin") == "topics/urban-cycling"
    assert model.resolve(PageKind.PERSON, "Bo") is None
    assert model.resolve(PageKind.TOPIC, "Ana Ruiz") is None  # other kind
    assert model.resolve(PageKind.TOPIC, "urban cycling") == "topics/urban-cycling"


def test_resolve_ambiguous_alias_returns_none(ana):
    model = loaded(ana)
    model.apply([CreatePage(kind=PageKind.PERSON, slug="ana-gomez", title="Ana Gomez",
                            aliases=["Ana"])])
    assert model.resolve(PageKind.PERSON, "Ana") is None
    assert model.resolve(PageKind.PERSON, "Ana Gomez") == "people/ana-gomez"


def test_save_writes_only_changed_files(ana):
    write_page(ana, "topics/tea", "# Tea\n\n## What I think\n")
    model = loaded(ana)
    paths = [ana / f"{ANA_KEY}.md", ana / "topics" / "tea.md", ana / ".claims" / f"{ANA_KEY}.json"]
    for p in paths:
        os.utime(p, ns=(1_000_000_000, 1_000_000_000))
    model.apply([AddClaim(page="topics/tea", section="What I think", text="Tea is good.",
                          evidence=["x"])])
    model.save()
    assert paths[0].stat().st_mtime_ns == 1_000_000_000
    assert paths[2].stat().st_mtime_ns == 1_000_000_000
    assert paths[1].stat().st_mtime_ns != 1_000_000_000
    assert (ana / ".claims" / "topics" / "tea.json").exists()
    leftovers = [p for p in ana.rglob("*") if p.name.endswith(".tmp")]
    assert leftovers == []


def test_save_refuses_to_overwrite_concurrent_owner_edit(ana):
    model = loaded(ana)
    model.apply([AddClaim(page=ANA_KEY, section="Who", text="Has a cat.", evidence=["x"])])
    edited = ANA + "Typed in Obsidian meanwhile.\n"
    write_page(ana, ANA_KEY, edited)
    sidecar_before = (ana / ".claims" / f"{ANA_KEY}.json").read_bytes()
    with pytest.raises(SelfModelConflict, match="ana-ruiz"):
        model.save()
    assert read_page(ana, ANA_KEY) == edited
    assert (ana / ".claims" / f"{ANA_KEY}.json").read_bytes() == sidecar_before


def test_concurrent_edit_to_unchanged_page_is_not_overwritten(ana):
    model = loaded(ana)
    write_page(ana, "topics/x", "# X\n")
    model.apply([CreatePage(kind=PageKind.TOPIC, slug="y", title="Y")])
    edited = ANA + "Typed meanwhile.\n"
    write_page(ana, ANA_KEY, edited)
    model.save()  # Ana's page was not changed by us: no conflict, no overwrite
    assert read_page(ana, ANA_KEY) == edited


def test_page_and_read_page(ana):
    model = loaded(ana)
    assert model.page(ANA_KEY).title == "Ana Ruiz"
    assert model.read_page(ANA_KEY) == ANA
    assert model.page("nope") is None and model.read_page("nope") is None
    assert [p.key for p in model.pages(PageKind.PERSON)] == [ANA_KEY]
    assert model.pages(PageKind.TOPIC) == []
