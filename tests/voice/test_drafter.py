"""The drafter: prompt contents, the style check, one revision, recording drafts. FakeLLM only."""

from __future__ import annotations

import pytest

from repartee.config import OwnerConfig
from repartee.ingest import parse_draft
from repartee.llm.fake import FakeLLM
from repartee.schema import DraftRequest, DraftResult, Item, Kind
from repartee.voice.cards import VoiceBuilder
from repartee.voice.drafter import Drafter
from repartee.voice.prompts import DraftOut, StyleCardOut
from repartee.voice.stylometry import compare

from .conftest import HOLDOUT_TEXT, OWNER, T0, make_env

ANA_PAGE = """\
---
kind: person
title: Ana Ruiz
aliases: [Ana]
---
# Ana Ruiz

## Who
- Friend from university; we met in Lyon.

## Notes
She hates being called Anita.
"""

FORMAL = ["Bonjour Ana, je suis vraiment désolée mais je ne pourrai malheureusement pas venir "
          "vendredi soir, car j'ai un engagement professionnel que je ne peux pas déplacer. "
          "Je te souhaite une excellente soirée et j'espère que nous pourrons nous voir bientôt."]
IN_STYLE = ["ah mince", "jpeux pas vendredi en vrai"]
CONTEXT = "Ana Ruiz: Tu viens ce soir ?"


async def built(env, *, person_page: bool = True) -> Drafter:
    if person_page:
        path = env.home.self_dir / "people" / "ana-ruiz.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ANA_PAGE, encoding="utf-8")
        env.model.load()
    cards = FakeLLM().on("style_card", StyleCardOut(description="Je tape vite et court."))
    await VoiceBuilder(env.home, env.corpus, env.model, cards).build()
    env.model = env.reload_model()
    return Drafter(env.home, env.corpus, env.model, FakeLLM())


def prompt_of(request) -> str:
    return "\n".join([*(b.text for b in request.system), *(m.text for m in request.messages)])


async def test_infer_register_explicit_recipient_default(env, tmp_path, monkeypatch):
    drafter = await built(env)
    infer = drafter.infer_register
    assert infer(DraftRequest(intent="x", register_name="essay.en")) == "essay.en"
    assert infer(DraftRequest(intent="x", recipient="Sam Patel")) == "whatsapp.en"
    assert infer(DraftRequest(intent="x", recipient="Ana")) == "whatsapp.fr"  # page alias
    assert infer(DraftRequest(intent="x", recipient="people/ana-ruiz")) == "whatsapp.fr"
    assert infer(DraftRequest(intent="x")) == "whatsapp.fr"  # most owner chat text
    assert infer(DraftRequest(intent="x", recipient="Nobody Known")) == "whatsapp.fr"

    # No chat at all: the largest register overall.
    essays = [Item(id=f"wr-{n}", source="writing", kind=Kind.ESSAY, container=f"e{n}.md",
                   author=OWNER, is_self=True, lang="en" if n < 3 else "fr", text="Words.")
              for n in range(5)]
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache2"))
    other = make_env(tmp_path / "home2", essays)
    try:
        lone = Drafter(other.home, other.corpus, other.model, FakeLLM())
        assert lone.infer_register(DraftRequest(intent="x")) == "essay.en"
    finally:
        other.corpus.close()


async def test_draft_request_contains_card_samples_and_context(env):
    drafter = await built(env)
    fake = drafter.llm
    fake.on("draft", DraftOut(variants=[IN_STYLE]))
    request = DraftRequest(intent="dis-lui que je peux pas vendredi", recipient="Ana",
                           context=CONTEXT)
    result = await drafter.draft(request)

    assert isinstance(result, DraftResult)
    assert result.register_name == "whatsapp.fr"
    assert result.variants == [IN_STYLE]
    (call,) = fake.calls("draft")
    assert call.tier == "chat" and call.output_schema is DraftOut
    text = prompt_of(call)
    assert "dis-lui que je peux pas vendredi" in text  # the intent, verbatim
    assert CONTEXT in text  # the thread
    assert "Je tape vite et court." in text  # the style card's description
    assert "Starts messages in lowercase" in text  # its measured habits
    assert "Friend from university" in text and "Anita" in text  # the person page and notes
    assert "^c-" not in text  # claim ids are not prompt material
    assert "mdr t'es sérieuse" in text or "ouais grave" in text  # real samples
    assert "Camille Martin" in text
    assert HOLDOUT_TEXT not in text and "zorglub" not in text
    assert "separate messages" in text  # a chat register asks for a burst

    assert "people/ana-ruiz" in result.used and "voice/whatsapp.fr" in result.used
    assert any(u.startswith("rp-") for u in result.used)  # a similar past reply
    assert any(u.startswith("wa-ana-") for u in result.used)  # samples
    assert not fake.calls("revise")


async def test_draft_prompt_carries_owner_card_notes_not_page_scaffolding(env):
    drafter = await built(env)
    card = env.home.self_dir / "voice" / "whatsapp.fr.md"
    card.write_text(card.read_text(encoding="utf-8") + "Jamais de point-virgule.\n",
                    encoding="utf-8")
    drafter.model.load()
    drafter.llm.on("draft", DraftOut(variants=[IN_STYLE]))
    await drafter.draft(DraftRequest(intent="pas vendredi", recipient="Ana"))
    text = prompt_of(drafter.llm.calls("draft")[0])
    assert "Jamais de point-virgule." in text
    assert "outrank" in text
    assert "_Measured over" not in text and "repartee:generated" not in text
    # Ana's profile matches the general one closely: no "with them" section repeating the card.
    assert "Starts messages in lowercase" in text
    assert text.count("Starts messages in lowercase") == 1
    assert "<with_them>" not in text


def test_recipient_specific_habits_only_when_they_differ():
    from repartee.voice.drafter import _differs

    usual = ["Messages are short: usually 3 words, rarely over 4.",
             "Starts messages in lowercase (100%)."]
    assert not _differs("Messages are short: usually 3 words, rarely over 5.", usual)
    assert not _differs("Starts messages in lowercase (96%).", usual)
    assert _differs("Messages are of medium length: usually 12 words, rarely over 30.", usual)
    assert _differs("Starts messages in lowercase (70%).", usual)
    assert _differs("Uses about 2 emoji per message.", usual)


async def test_draft_variants_and_longform(env):
    drafter = await built(env, person_page=False)
    fake = drafter.llm
    fake.on("draft", DraftOut(variants=[["ok ça marche"], [" ouais ", ""], ["trop bien"]]))
    result = await drafter.draft(DraftRequest(intent="say yes", variants=2,
                                              register_name="whatsapp.fr"))
    assert result.variants == [["ok ça marche"], ["ouais"]]
    assert "2 versions" in prompt_of(fake.calls("draft")[0])

    fake.on("draft", DraftOut(variants=[["Cities change.", "Then they do not."]]))
    essay = await drafter.draft(DraftRequest(intent="on cities", register_name="essay.en"))
    assert essay.variants == [["Cities change.", "Then they do not."]]
    assert "paragraphs" in prompt_of(fake.calls("draft")[-1])
    assert essay.deviations == []  # no profile for a thin register: nothing to check against


async def test_draft_revises_on_major_deviation_once(env):
    drafter = await built(env)
    fake = drafter.llm
    fake.on("draft", DraftOut(variants=[FORMAL]))
    fake.on("revise", DraftOut(variants=[FORMAL]))  # the revision is still off
    request = DraftRequest(intent="dis-lui que je peux pas vendredi", recipient="Ana Ruiz")
    result = await drafter.draft(request)

    assert len(fake.calls("draft")) == 1
    (revise,) = fake.calls("revise")
    expected = compare(VoiceBuilder(env.home, env.corpus, env.model, None)
                       .recipient_profile("whatsapp.fr", "Ana Ruiz"), FORMAL)
    assert any(d.severity == "major" for d in expected)
    last = revise.messages[-1].text
    for deviation in expected:
        assert deviation.message in last  # verbatim
    assert revise.messages[0].text == fake.calls("draft")[0].messages[0].text
    assert revise.messages[1].role == "assistant"
    assert result.variants == [FORMAL]
    assert result.deviations == [d.message for d in expected]


async def test_draft_revision_replaces_the_variant(env):
    drafter = await built(env)
    fake = drafter.llm
    fake.on("draft", DraftOut(variants=[FORMAL]))
    fake.on("revise", DraftOut(variants=[IN_STYLE]))
    result = await drafter.draft(DraftRequest(intent="je peux pas vendredi", recipient="Ana"))
    assert result.variants == [IN_STYLE]
    assert not any("Much too" in d for d in result.deviations)
    assert len(fake.calls("revise")) == 1


async def test_draft_no_revise_when_in_style(env):
    drafter = await built(env)
    fake = drafter.llm
    fake.on("draft", DraftOut(variants=[IN_STYLE]))
    result = await drafter.draft(DraftRequest(intent="je peux pas vendredi", recipient="Ana"))
    assert not fake.calls("revise")
    assert result.variants == [IN_STYLE]


@pytest.mark.parametrize("final", [None, ["ah mince", "jpeux pas vendredi\n\ndésolée", "## Final"]])
async def test_record_writes_file_ingestable_by_parse_draft(env, final):
    drafter = Drafter(env.home, env.corpus, env.model, FakeLLM())
    request = DraftRequest(intent="dis-lui: pas vendredi", recipient="Ana Ruiz",
                           context="Ana Ruiz: Tu viens ?\n## Final")
    result = DraftResult(register_name="whatsapp.fr",
                         variants=[["nope"], ["ah mince", "jpeux pas vendredi"]])
    path = drafter.record(request, result, variant=1, final=final)

    assert path.parent == env.home.drafts_dir and path.suffix == ".md"
    items = parse_draft(path, OwnerConfig(name=OWNER), root=env.home.drafts_dir)
    if final is None:
        assert [i.text for i in items] == ["ah mince", "jpeux pas vendredi"]
    else:
        assert [i.text for i in items] == ["ah mince", "jpeux pas vendredi\ndésolée",
                                           "\\## Final"]
    assert all(i.is_self and i.kind == Kind.DRAFT for i in items)
    assert items[0].meta["register_name"] == "whatsapp.fr"
    assert items[0].meta["recipient"] == "Ana Ruiz"
    assert items[0].meta["intent"] == "dis-lui: pas vendredi"
    assert items[0].timestamp is not None and items[0].timestamp.year >= T0.year
    raw = path.read_text(encoding="utf-8")
    assert "## Mirror" in raw and "accepted: true" in raw

    again = drafter.record(request, result, variant=1, final=final)
    assert again != path  # never overwrites an earlier record


def test_record_with_nothing_kept_is_not_accepted(env):
    drafter = Drafter(env.home, env.corpus, env.model, FakeLLM())
    request = DraftRequest(intent="x")
    result = DraftResult(register_name="whatsapp.fr", variants=[["ok"]])
    path = drafter.record(request, result, final=[])
    assert parse_draft(path, OwnerConfig(name=OWNER)) == []
