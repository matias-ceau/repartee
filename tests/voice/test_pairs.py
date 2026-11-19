"""Reply pairs mined from chats. Fictional fixtures."""

from datetime import datetime, timedelta

from repartee.schema import Item, Kind, stable_hash
from repartee.voice.pairs import mine_reply_pairs

T0 = datetime(2024, 5, 4, 18, 0)


def msg(i, author, text, minutes, *, container="Ana", holdout=False):
    return Item(id=f"wa-{container}-{i}", source="whatsapp", kind=Kind.MESSAGE,
                container=container, author=author, is_self=author == "Moi",
                timestamp=T0 + timedelta(minutes=minutes), text=text, lang="fr", seq=i,
                holdout=holdout)


def test_reply_pairs_basic():
    items = [
        msg(1, "Moi", "salut", -10),
        msg(2, "Ana", "tu viens ce soir ?", 0),
        msg(3, "Ana", "on sera 5", 1),
        msg(4, "Moi", "oui", 3),
        msg(5, "Moi", "j'arrive vers 9h", 3.5),
    ]
    pairs = mine_reply_pairs(reversed(items))
    assert len(pairs) == 1
    p = pairs[0]
    assert p.recipient == "Ana"
    assert p.container == "Ana"
    assert p.context == "Moi: salut\nAna: tu viens ce soir ?\nAna: on sera 5"
    assert p.reply == ["oui", "j'arrive vers 9h"]
    assert p.context_item_ids == ["wa-Ana-1", "wa-Ana-2", "wa-Ana-3"]
    assert p.reply_item_ids == ["wa-Ana-4", "wa-Ana-5"]
    assert p.id == "rp-" + stable_hash("wa-Ana-4", "wa-Ana-5")
    assert p.timestamp == T0 + timedelta(minutes=3)
    assert p.lang == "fr"
    # context is capped
    capped = mine_reply_pairs(items, max_context_messages=1)
    assert capped[0].context == "Ana: on sera 5"


def test_reply_pairs_skip_holdout_and_stale_context():
    items = [
        msg(1, "Ana", "t'as vu le match ?", 0),
        msg(2, "Moi", "non", 300),                       # 5 h later: no pair
        msg(3, "Ana", "et demain ?", 301),
        msg(4, "Moi", "peut-être", 302, holdout=True),   # holdout reply: skipped
        msg(5, "Ana", "ok dis-moi", 303),
        msg(6, "Moi", "promis", 304),                    # context stops before the holdout
        msg(7, "Ana", "vieux message", 1000),
        msg(8, "Ana", "nouveau message", 1400),
        msg(9, "Moi", "reçu", 1401),                     # context: only the fresh message
    ]
    pairs = mine_reply_pairs(items)
    assert [p.reply for p in pairs] == [["promis"], ["reçu"]]
    assert pairs[0].context == "Ana: ok dis-moi"
    assert pairs[1].context == "Ana: nouveau message"
    for p in pairs:
        assert "wa-Ana-4" not in p.context_item_ids + p.reply_item_ids


def test_reply_pairs_group_recipient():
    items = [
        msg(1, "Ana", "qui grimpe samedi ?", 0, container="Escalade"),
        msg(2, "Bruno", "moi", 1, container="Escalade"),
        msg(3, "Moi", "moi aussi", 2, container="Escalade"),
    ]
    pairs = mine_reply_pairs(items)
    assert len(pairs) == 1
    assert pairs[0].recipient == "Escalade"
    assert pairs[0].context == "Ana: qui grimpe samedi ?\nBruno: moi"


def test_reply_pairs_ignore_non_chat_items():
    note = Item(id="nt-1", source="vault", kind=Kind.NOTE, container="a.md", author="Moi",
                is_self=True, text="une note", lang="fr")
    assert mine_reply_pairs([note]) == []
