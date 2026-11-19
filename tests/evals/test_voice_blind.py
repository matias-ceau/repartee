"""The blind voice test: fair pairs from held-out texts, reproducible order, honest scores."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from repartee.evals import BlindPair, BlindTest
from repartee.evals.common import Unit, eligible, holdout_units, quotes, units
from repartee.llm import FakeLLM
from repartee.llm.base import LLMError
from repartee.runtime import Runtime
from repartee.schema import Item, Kind
from repartee.store.corpus import Corpus
from repartee.store.embed import HashEmbedder
from repartee.sync import run_voice
from repartee.voice.stylometry import tokenize

from .conftest import TEXT_BLOCK


def _messages(prompt: str) -> set[str]:
    """Every line of a prompt with tags and a leading "Name: " removed, whitespace collapsed."""
    out = set()
    for line in prompt.splitlines():
        line = re.sub(r"<[^>]+>", "", line).strip()
        out.add(" ".join(line.split()))
        out.add(" ".join(line.split(": ", 1)[-1].split()))
    return out


def _msg(text: str, **meta: object) -> Item:
    return Item(id="wa-x", source="whatsapp", kind=Kind.MESSAGE, container="Ana",
                author="Camille", is_self=True, timestamp=datetime(2024, 1, 1), text=text,
                lang="fr", holdout=True, meta=dict(meta))


def test_eligible_skips_short_links_and_captions() -> None:
    assert eligible([_msg("on se voit demain soir alors")])
    assert not eligible([_msg("ok ça marche")])  # three words
    assert not eligible([_msg("regarde ça https://example.org/un/long/lien?x=1 mdr")])
    assert not eligible([_msg("la plage ce matin, incroyable", media=True)])


async def test_pairs_use_only_holdout_items_long_enough(rt: Runtime) -> None:
    test = BlindTest(rt, n=100, seed=0)
    pairs = await test.prepare()
    assert pairs
    for pair in pairs:
        items = rt.corpus.get_items(pair.item_ids)
        assert pair.item_ids[0] == pair.item_id
        assert len(items) == len(pair.item_ids)
        assert all(item.holdout and item.is_self for item in items)
        assert len(tokenize(" ".join(pair.real))) > 3
    shown = {i for p in pairs for i in p.item_ids}
    short = [i for i in rt.corpus.iter_items(holdout=True) if i.is_self
             and len(tokenize(i.text)) <= 3]
    assert short and not shown & {i.id for i in short}


async def test_a_register_can_be_chosen(rt: Runtime) -> None:
    pairs = await BlindTest(rt, register_name="whatsapp.en", n=3, seed=1).prepare()
    assert len(pairs) == 3
    assert {p.register_name for p in pairs} == {"whatsapp.en"}


async def test_brief_forbids_quoting_and_draft_never_sees_the_real_text(
        rt: Runtime, fake: FakeLLM) -> None:
    pairs = await BlindTest(rt, n=100, seed=0).prepare()
    briefs = fake.calls("brief")
    assert briefs and all(r.tier == "smart" and r.output_schema is not None for r in briefs)
    for request in briefs:
        instructions = "\n".join(b.text for b in request.system) + request.messages[-1].text
        assert "Do not quote" in instructions
        assert TEXT_BLOCK.search(request.messages[-1].text)
    drafts = fake.calls("draft")
    assert len(drafts) == len(pairs)
    for request in drafts:
        seen = _messages("\n".join(m.text for m in request.messages))
        for pair in pairs:
            assert not seen & {" ".join(text.split()) for text in pair.real}


async def test_draft_has_the_real_items_recipient_register_and_shape(
        rt: Runtime, fake: FakeLLM) -> None:
    await run_voice(rt, commit=False)  # the style card names the register in the prompt
    pairs = await BlindTest(rt, register_name="whatsapp.fr", n=4, seed=3).prepare()
    assert len(pairs) == 4
    for pair in pairs:
        assert pair.recipient in {"Léa", "Françoise Martin"}
        if len(pair.real) == 1:  # one message against one message
            assert len(pair.generated) == 1
    prompts = [r.messages[-1].text for r in fake.calls("draft")]
    assert all("whatsapp.fr" in p for p in prompts)
    assert all("one single message" in p or "separate messages" in p for p in prompts)


async def test_a_brief_that_quotes_the_original_skips_the_pair(rt: Runtime, fake: FakeLLM) -> None:
    def parrot(request):
        return {"brief": "Say exactly: " + TEXT_BLOCK.search(request.messages[-1].text).group(1)}

    fake.on("brief", parrot)
    test = BlindTest(rt, n=3, seed=0)
    assert await test.prepare() == []
    assert len(test.skipped) == 3
    assert all("brief" in s["reason"] for s in test.skipped)
    assert not fake.calls("draft")


async def test_order_comes_from_the_seed(rt: Runtime) -> None:
    async def order(seed: int) -> list[tuple[str, bool]]:
        return [(p.item_id, p.real_is_a) for p in await BlindTest(rt, n=8, seed=seed).prepare()]

    first = await order(0)
    assert first == await order(0)
    others = [await order(seed) for seed in (1, 2, 3)]
    assert any(o != first for o in others)
    sides = [a for o in [first, *others] for _, a in o]
    assert any(sides) and not all(sides)


async def test_a_failed_draft_becomes_a_skipped_pair(rt: Runtime, fake: FakeLLM) -> None:
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMError("overloaded")
        return {"variants": [["ouais ça me va"]]}

    fake.on("draft", flaky)
    test = BlindTest(rt, n=4, seed=0)
    pairs = await test.prepare()
    assert len(pairs) == 3
    assert len(test.skipped) == 1
    assert "overloaded" in test.skipped[0]["reason"]
    assert test.skipped[0]["item_id"] not in {p.item_id for p in pairs}


def _pair(item_id: str, register: str, real_is_a: bool) -> BlindPair:
    return BlindPair(item_id=item_id, register_name=register, brief="b", real=["r"],
                     generated=["g"], real_is_a=real_is_a)


def test_score_is_the_share_fooled_per_register_and_overall(rt: Runtime) -> None:
    test = BlindTest(rt, n=5)
    test.pairs = [_pair("a", "whatsapp.fr", True), _pair("b", "whatsapp.fr", False),
                  _pair("c", "whatsapp.fr", True), _pair("d", "whatsapp.en", True),
                  _pair("e", "whatsapp.en", False)]
    assert test.score() == {}
    test.record(test.pairs[0], picked_a=True)   # spotted
    test.record(test.pairs[1], picked_a=True)   # fooled
    test.record(test.pairs[2], picked_a=False)  # fooled
    test.record(test.pairs[3], picked_a=False)  # fooled
    # e is never judged
    score = test.score()
    assert score["whatsapp.fr"] == 2 / 3
    assert score["whatsapp.en"] == 1.0
    assert score["overall"] == 3 / 4
    test.record(test.pairs[0], picked_a=False)  # changing one's mind replaces the pick
    assert test.score()["whatsapp.fr"] == 1.0


async def test_saved_file_shape(rt: Runtime) -> None:
    test = BlindTest(rt, n=3, seed=0)
    pairs = await test.prepare()
    test.record(pairs[0], picked_a=pairs[0].real_is_a)
    path = test.save()
    assert path.parent == rt.home.evals_dir
    assert path.name.startswith("voice-") and path.suffix == ".json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["eval"] == "voice_blind"
    datetime.fromisoformat(data["created"])
    assert data["settings"]["n"] == 3 and data["settings"]["seed"] == 0
    assert data["settings"]["register_name"] is None
    assert "50%" in data["meaning"]
    assert data["score"] == test.score()
    assert data["counts"]["overall"] == {"judged": 1, "fooled": 0}
    assert len(data["pairs"]) == 3
    first = data["pairs"][0]
    assert {"item_id", "item_ids", "register_name", "recipient", "brief", "real", "generated",
            "real_is_a", "picked_a", "fooled"} <= set(first)
    assert first["fooled"] is False and data["pairs"][1]["picked_a"] is None
    assert data["skipped"] == []
    assert test.save() != path  # a second save never overwrites the first


def test_holdout_units_group_a_burst_only_when_all_of_it_is_held_out(rt: Runtime) -> None:
    units = holdout_units(rt.corpus)
    assert units
    for unit in units:
        assert all(item.holdout for item in unit.items)
        assert len({item.container for item in unit.items}) == 1


def _thread(holdouts: list[bool]) -> list[Item]:
    """Ana writes, then the owner sends one message a minute (a burst), held out as given."""
    t0 = datetime(2024, 5, 2, 18, 0)
    items = [Item(id="wa-in-0", source="whatsapp", kind=Kind.MESSAGE, container="Ana",
                  author="Ana", is_self=False, timestamp=t0, lang="fr", seq=0,
                  text="tu viens samedi au pique-nique du canal ?")]
    for n, held in enumerate(holdouts, 1):
        items.append(Item(id=f"wa-me-{n}", source="whatsapp", kind=Kind.MESSAGE,
                          container="Ana", author="Camille", is_self=True, lang="fr", seq=n,
                          timestamp=t0 + timedelta(minutes=n), holdout=held,
                          text=f"message numéro {n} de la petite rafale du soir"))
    return items


def test_a_wholly_held_out_burst_is_one_unit(tmp_path) -> None:
    with Corpus(tmp_path / "index.db", HashEmbedder()) as corpus:
        corpus.sync_items(_thread([True, True, True]))
        [unit] = units(corpus, holdout=True)
        assert unit.item_ids == ["wa-me-1", "wa-me-2", "wa-me-3"]
        assert unit.recipient == "Ana" and not unit.group
        assert unit.context == "Ana: tu viens samedi au pique-nique du canal ?"
        assert "3 separate messages" in unit.shape_note()
        assert len(unit.texts()) == 3


def test_a_mixed_burst_gives_single_messages(tmp_path) -> None:
    with Corpus(tmp_path / "index.db", HashEmbedder()) as corpus:
        corpus.sync_items(_thread([False, True, False]))
        [unit] = units(corpus, holdout=True)
        assert unit.item_ids == ["wa-me-2"]
        assert unit.shape_note() == "(Send this as one single message.)"
        assert unit.shape(["un", "deux"]) == ["un\ndeux"]  # one message against one message
        assert "Camille: message numéro 1" in unit.context
        others = units(corpus, holdout=False)
        assert sorted(u.key for u in others) == ["wa-me-1", "wa-me-3"]
        assert all("numéro 2" not in (u.context or "") for u in others)  # never held-out text


def test_longform_is_shown_without_vault_markup() -> None:
    item = Item(id="nt-1", source="vault", kind=Kind.JOURNAL, container="Journal/x.md",
                author="Camille", is_self=True, lang="fr", holdout=True,
                text="# 9 mars\n\n[[Cadence Streets]] m'a fait une offre.\n\n"
                     "J'en parle à [[Léa Fontaine|Léa]] demain matin.")
    unit = Unit("journal.fr", [item], chat=False)
    assert unit.texts() == ["Cadence Streets m'a fait une offre.\n\nJ'en parle à Léa demain matin."]
    assert unit.shape_note() == "(Write this as 2 paragraphs.)"
    assert unit.shape(["Un.", "Deux."]) == ["Un.\n\nDeux."]


def test_quotes_detects_a_run_of_the_original_wording() -> None:
    real = ["je préfère te le dire avant que léa te le raconte"]
    assert quotes("Tell her: je préfère te le dire avant tout", real)
    assert not quotes("Tell your mother you would rather she hear it from you first", real)
    assert quotes("Say Fair enough, I will.", ["Fair enough. I will"])  # a short text whole
