"""Voice on exported AI assistant conversations: owner turns only, pasted text left out."""

from __future__ import annotations

from datetime import datetime, timedelta

from repartee.schema import Item, Kind
from repartee.voice.registers import group_by_register, is_chat_register, register_of
from repartee.voice.samples import pick_samples, register_items
from repartee.voice.stylometry import profile_items

from .conftest import OWNER, make_env

T0 = datetime(2025, 3, 1, 10, 0)
ASSISTANT_MARK = "zorblax"  # appears only in the assistant's turns
CODE_MARK = "frobnicate"  # appears only inside fenced code in an owner turn
PASTE_MARK = "quuxlog"  # appears only in a mostly pasted owner turn

OWN = [
    "I want to plan a quiet weekend in the mountains with two friends, nothing fancy.",
    "Could you suggest a simple soup recipe that keeps well for three days in the fridge?",
    "My neighbour's dog barks every morning at six and I would like to bring it up kindly.",
    "I am trying to read more in the evening instead of scrolling, any gentle tricks?",
    "Which houseplants survive a north-facing room with very little direct light?",
    "I need to write a short thank-you card for my aunt who knitted me a scarf.",
]


def _items() -> list[Item]:
    items: list[Item] = []

    def add(n: int, mine: bool, text: str, **meta) -> None:
        items.append(Item(
            id=f"ai-test-{len(items)}", source="assistant", kind=Kind.ASSISTANT,
            container=f"Chat {n} (abcd{n:04d})", author=OWNER if mine else "Claude",
            is_self=mine, timestamp=T0 + timedelta(days=n, minutes=len(items)), text=text,
            title=f"Chat {n}", lang="en", seq=len(items),
            meta={"role": "me" if mine else "assistant", **meta}))

    for n, text in enumerate(OWN):
        add(n, True, text)
        add(n, False, f"Sure, here is an idea {ASSISTANT_MARK}: " + "lots of advice. " * 40)
    add(10, True, "Here is my script, it should rename my photos by date.\n\n```python\n"
        f"def {CODE_MARK}(path):\n    return path\n```\n\nWhy does it skip some of them?",
        code=True)
    add(11, True, f"what is this\n{PASTE_MARK} 2025-01-01 12:00:00 ERROR something",
        pasted="pasted")
    return items


def test_assistant_register_is_longform_and_owner_only():
    items = _items()
    assert not is_chat_register("assistant.en")
    groups = group_by_register(items)
    assert set(groups) == {"assistant.en"}
    texts = [i.text for i in groups["assistant.en"]]
    assert all(ASSISTANT_MARK not in t for t in texts)
    assert all(PASTE_MARK not in t for t in texts)
    assert all(CODE_MARK not in t for t in texts)
    assert any("Why does it skip some of them?" in t for t in texts)  # prose kept
    assert register_of(items[1]) is None  # an assistant turn
    assert register_of(items[-1]) is None  # mostly pasted


def test_style_profile_ignores_assistant_turns_and_code():
    items = _items()
    profile = profile_items("assistant.en", group_by_register(items)["assistant.en"])
    assert profile.form == "longform"
    assert profile.n_texts == len(OWN) + 1
    assert not any(ASSISTANT_MARK in p for p, _ in profile.distinctive_phrases)


def test_samples_are_owner_turns_without_code_or_pastes(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    env = make_env(tmp_path / "home", _items())
    try:
        pool = register_items(env.corpus, "assistant.en")
        assert len(pool) == len(OWN) + 1
        samples = pick_samples(env.corpus, "assistant.en", topic="photos script", k=20)
        assert samples and all(s.is_self for s in samples)
        joined = "\n".join(s.text for s in samples)
        assert ASSISTANT_MARK not in joined
        assert CODE_MARK not in joined and PASTE_MARK not in joined
        # The stored item keeps its code for search.
        stored = [i for i in env.corpus.iter_items() if i.meta.get("code")]
        assert stored and CODE_MARK in stored[0].text
    finally:
        env.corpus.close()
