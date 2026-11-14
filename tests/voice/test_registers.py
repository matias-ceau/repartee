"""Registers: which kind of text in which language an item belongs to. Fictional fixtures."""

from datetime import datetime, timedelta

from repartee.schema import Item, Kind
from repartee.voice.registers import bursts, group_by_register, is_chat_register, register_of

T0 = datetime(2024, 3, 1, 12, 0)


def item(i, text="x", *, kind=Kind.MESSAGE, source="whatsapp", lang="fr", is_self=True,
         holdout=False, container="Ana", author=None, ts=None, seq=0, meta=None):
    return Item(id=f"it-{i}", source=source, kind=kind, container=container,
                author=author or ("Moi" if is_self else "Ana"), is_self=is_self,
                timestamp=ts, text=text, lang=lang, seq=seq, holdout=holdout, meta=meta or {})


def test_register_of_by_kind_and_lang():
    assert register_of(item(1)) == "whatsapp.fr"
    assert register_of(item(2, kind=Kind.ESSAY, source="writing", lang="en")) == "essay.en"
    assert register_of(item(3, kind=Kind.JOURNAL, source="vault")) == "journal.fr"
    assert register_of(item(4, kind=Kind.NOTE, source="vault", lang="en")) == "note.en"
    assert register_of(item(5, kind=Kind.INTERVIEW, source="interviews")) == "interview.fr"
    draft = item(6, kind=Kind.DRAFT, source="drafts", meta={"register_name": "whatsapp.en"})
    assert register_of(draft) == "whatsapp.en"
    assert register_of(item(7, kind=Kind.DRAFT, source="drafts", lang="en")) == "draft.en"
    # not the owner's, no language, or not writing voice
    assert register_of(item(8, is_self=False)) is None
    assert register_of(item(9, lang=None)) is None
    assert register_of(item(10, kind=Kind.TASTE, source="tastes")) is None


def test_group_excludes_holdout_and_others():
    items = [
        item(1, "a"),
        item(2, "b", holdout=True),
        item(3, "c", is_self=False),
        item(4, "d", lang="en"),
        item(5, "e", kind=Kind.ESSAY, source="writing", lang="en"),
        item(6, "f"),
    ]
    groups = group_by_register(items)
    assert {k: [i.id for i in v] for k, v in groups.items()} == {
        "whatsapp.fr": ["it-1", "it-6"],
        "whatsapp.en": ["it-4"],
        "essay.en": ["it-5"],
    }


def test_is_chat_register():
    assert is_chat_register("whatsapp.fr")
    assert is_chat_register("telegram.pt-br")
    assert not is_chat_register("essay.en")
    assert not is_chat_register("journal.fr")
    assert not is_chat_register("draft.en")
    assert not is_chat_register("note")


def test_bursts_split_on_gap_and_other_author():
    m = timedelta(minutes=1)
    items = [
        item(1, ts=T0, seq=1),
        item(2, ts=T0 + m, seq=2),
        item(3, ts=T0 + 2 * m, seq=3, is_self=False),
        item(4, ts=T0 + 3 * m, seq=4),
        item(5, ts=T0 + 6 * m, seq=5),  # 3 minutes later: new burst
        item(6, ts=T0 + m, seq=1, container="Bob"),
        item(7, ts=None, seq=2, container="Bob"),  # no time: cannot join
    ]
    got = [[i.id for i in b] for b in bursts(reversed(items))]
    assert sorted(got) == sorted([["it-1", "it-2"], ["it-4"], ["it-5"], ["it-6"], ["it-7"]])
