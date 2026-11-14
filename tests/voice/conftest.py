"""A small fictional home with chats and essays for the style card and drafter tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from repartee.config import ChunkingConfig
from repartee.home import Home
from repartee.ingest.chunking import chunk_items
from repartee.schema import Item, Kind
from repartee.scaffold import init_home
from repartee.selfmodel.model import SelfModel
from repartee.store.corpus import Corpus
from repartee.store.embed import HashEmbedder
from repartee.voice.pairs import mine_reply_pairs

OWNER = "Camille Martin"
T0 = datetime(2024, 3, 1, 9, 0)
#: Text of the one held-out owner message: must never reach a card, a sample or a prompt.
HOLDOUT_TEXT = "zorglub cette phrase est réservée au test à l'aveugle"
TOPIC_TEXT = "on va faire du vélo au canal dimanche"

FR_SELF = ["ouais grave", "jsuis en route", "mdr t'es sérieuse", "ok ça marche", "trop bien",
           "en vrai je sais pas", "j'arrive dans 10 min", "on se capte demain ?", "mdrr",
           "carrément", "en vrai c'est cool", "jpp de lui"]
FR_OTHER = ["Tu viens ce soir ?", "On sera cinq.", "Tu as vu le message de Paul ?",
            "Je pense qu'il va pleuvoir demain.", "C'est noté, merci !"]
EN_SELF = ["yeah sure", "omw", "lol no way", "sounds good", "haha ok", "wait what",
           "be there in 10", "nah im good"]
EN_OTHER = ["Are you around later?", "The meeting moved to Thursday.", "Did you see that?"]


def _chat(container: str, other: str, rounds: int, mine: list[str], theirs: list[str],
          lang: str, start: datetime, prefix: str) -> list[Item]:
    items: list[Item] = []
    seq = 0

    def add(author: str, text: str, when: datetime) -> Item:
        nonlocal seq
        item = Item(id=f"wa-{prefix}-{seq}", source="whatsapp", kind=Kind.MESSAGE,
                    container=container, author=author, is_self=author == OWNER,
                    timestamp=when, text=text, lang=lang, seq=seq)
        seq += 1
        items.append(item)
        return item

    for r in range(rounds):
        t = start + timedelta(hours=5 * r)
        add(other, theirs[r % len(theirs)], t)
        add(OWNER, mine[(2 * r) % len(mine)], t + timedelta(minutes=2))
        add(OWNER, mine[(2 * r + 1) % len(mine)], t + timedelta(minutes=3))
    return items


def build_items() -> list[Item]:
    """Ana (French, 210 owner messages), Léo (French, 40), Sam (English, 80), five essays."""
    ana = _chat("Ana Ruiz", "Ana Ruiz", 105, FR_SELF, FR_OTHER, "fr", T0, "ana")
    own = [i for i in ana if i.is_self]
    own[14].text, own[14].holdout = HOLDOUT_TEXT, True
    own[100].text = TOPIC_TEXT
    leo = _chat("Léo", "Léo", 20, FR_SELF, FR_OTHER, "fr", T0 + timedelta(minutes=30), "leo")
    sam = _chat("Sam Patel", "Sam Patel", 40, EN_SELF, EN_OTHER, "en",
                T0 + timedelta(minutes=50), "sam")
    essays = [
        Item(id=f"wr-essay-{n}", source="writing", kind=Kind.ESSAY, container=f"essay{n}.md",
             author=OWNER, is_self=True, timestamp=T0 + timedelta(days=n), lang="en", seq=0,
             text="Cities change slowly, and then all at once. I have watched mine do both.")
        for n in range(5)
    ]
    return ana + leo + sam + essays


@dataclass
class VoiceEnv:
    """A home, its corpus and its self-model, filled with the fictional fixture."""

    home: Home
    corpus: Corpus
    model: SelfModel
    items: list[Item]

    def reload_model(self) -> SelfModel:
        """A fresh SelfModel read from disk."""
        model = SelfModel(self.home.self_dir)
        model.load()
        return model


def make_env(root: Path, items: list[Item]) -> VoiceEnv:
    """Home at `root` with `items` indexed (chunks and reply pairs included)."""
    home = init_home(root, OWNER, git=False)
    corpus = Corpus(home.index_path, HashEmbedder())
    corpus.sync_items(items)
    corpus.sync_chunks(chunk_items(items, ChunkingConfig()))
    corpus.sync_reply_pairs(mine_reply_pairs(items))
    model = SelfModel(home.self_dir)
    model.load()
    return VoiceEnv(home, corpus, model, items)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """The standard fictional home."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    voice_env = make_env(tmp_path / "home", build_items())
    yield voice_env
    voice_env.corpus.close()
