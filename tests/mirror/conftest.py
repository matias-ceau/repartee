"""Fixtures for mirror tests: a small home, self-model and corpus. All data is invented."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pytest

from repartee.llm.fake import FakeLLM
from repartee.mirror import Mirror
from repartee.scaffold import init_home
from repartee.schema import (
    AddClaim,
    Chunk,
    CreatePage,
    DraftRequest,
    DraftResult,
    Item,
    Kind,
    PageKind,
    ReplyPair,
)
from repartee.selfmodel import SelfModel
from repartee.store import Corpus, HashEmbedder

OWNER = "Camille Martin"
ANA = "people/ana-ruiz"
MOVING = "topics/moving-abroad"
CORE = "I'm Camille. I draw maps for a living and I live in Lyon."


def _item(id: str, text: str, ts: datetime, *, author: str = "Ana Ruiz", source="whatsapp",
          kind=Kind.MESSAGE, container="Ana Ruiz", seq=0) -> Item:
    return Item(id=id, source=source, kind=kind, container=container, author=author,
                is_self=author == OWNER, timestamp=ts, text=text, seq=seq)


ITEMS = [
    _item("wa-a1", "tu te souviens de Lyon en 2014 ?", datetime(2021, 3, 12, 14, 5)),
    _item("wa-a2", "évidemment, la fac !", datetime(2021, 3, 12, 14, 6), author=OWNER, seq=1),
    _item("wa-b1", "alors, Montréal ?", datetime(2022, 4, 2, 9, 0), seq=2),
    _item("wa-b2", "j'hésite encore, partir me fait peur", datetime(2022, 4, 2, 9, 3),
          author=OWNER, seq=3),
    _item("nt-j1", "I keep thinking about moving to Montréal. Leaving Ana scares me.",
          datetime(2022, 3, 14), author=OWNER, source="vault", kind=Kind.JOURNAL,
          container="Journal/2022-03-14.md"),
]

CHUNKS = [
    Chunk(id="ch-a", source="whatsapp", kind=Kind.MESSAGE, container="Ana Ruiz",
          item_ids=["wa-a1", "wa-a2"], start=ITEMS[0].timestamp, end=ITEMS[1].timestamp,
          participants=["Ana Ruiz", OWNER],
          text="[2021-03-12 14:05] Ana Ruiz: tu te souviens de Lyon en 2014 ?\n"
               "[2021-03-12 14:06] Camille Martin: évidemment, la fac !",
          self_text="évidemment, la fac !"),
    Chunk(id="ch-b", source="whatsapp", kind=Kind.MESSAGE, container="Ana Ruiz",
          item_ids=["wa-b1", "wa-b2"], start=ITEMS[2].timestamp, end=ITEMS[3].timestamp,
          participants=["Ana Ruiz", OWNER],
          text="[2022-04-02 09:00] Ana Ruiz: alors, Montréal ?\n"
               "[2022-04-02 09:03] Camille Martin: j'hésite encore, partir me fait peur",
          self_text="j'hésite encore, partir me fait peur"),
    Chunk(id="ch-j", source="vault", kind=Kind.JOURNAL, container="Journal/2022-03-14.md",
          item_ids=["nt-j1"], start=ITEMS[4].timestamp, end=ITEMS[4].timestamp,
          participants=[OWNER], title="2022-03-14", text=ITEMS[4].text,
          self_text=ITEMS[4].text),
]

PAIRS = [
    ReplyPair(id="rp-1", container="Ana Ruiz", recipient="Ana Ruiz",
              timestamp=datetime(2022, 4, 2, 9, 3), context="Ana Ruiz: alors, Montréal ?",
              reply=["j'hésite encore", "partir me fait peur"],
              context_item_ids=["wa-b1"], reply_item_ids=["wa-b2"]),
]


class Built:
    """Handles to the test world."""

    def __init__(self, home, config, model: SelfModel, corpus: Corpus, claims: dict[str, str]):
        self.home, self.config, self.model, self.corpus, self.claims = (
            home, config, model, corpus, claims)


@pytest.fixture
def world(tmp_path: Path, monkeypatch) -> Built:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    home = init_home(tmp_path / "home", OWNER, git=False)
    config = home.load_config()
    model = SelfModel(home.self_dir)
    model.load()
    model.set_core_generated(CORE)
    results = model.apply([
        CreatePage(kind=PageKind.PERSON, slug="ana-ruiz", title="Ana Ruiz", aliases=["Ana"]),
        CreatePage(kind=PageKind.TOPIC, slug="moving-abroad", title="Moving abroad"),
        AddClaim(page=ANA, section="Who", text="We met at university in Lyon in 2014.",
                 evidence=["wa-a1", "wa-a2"], confidence=0.6),
        AddClaim(page=ANA, section="Who", text="Ana works night shifts as a nurse.",
                 evidence=["wa-a1"], confidence=0.3),
        AddClaim(page=MOVING, section="What I think",
                 text="I want to move to Montréal but leaving my friends scares me.",
                 evidence=["nt-j1", "wa-b2"], confidence=0.7, valid_from="2022"),
        AddClaim(page=MOVING, section="What I think", text="I never wanted to leave Lyon.",
                 evidence=["wa-a2"], confidence=0.45, valid_to="2021"),
    ])
    assert all(r.ok for r in results), results
    ids = [r.claim_id for r in results[2:]]
    claims = {"met": ids[0], "nurse": ids[1], "montreal": ids[2], "lyon": ids[3]}
    model.confirm(claims["met"])
    model.save()
    corpus = Corpus(tmp_path / "index.db", HashEmbedder())
    corpus.sync_items(ITEMS)
    corpus.sync_chunks(CHUNKS)
    corpus.sync_reply_pairs(PAIRS)
    corpus.sync_claims(model.claims())
    yield Built(home, config, model, corpus, claims)
    corpus.close()


class StubDrafter:
    """Stands in for voice.drafter.Drafter."""

    def __init__(self, drafts_dir: Path):
        self.drafts_dir = drafts_dir
        self.requests: list[DraftRequest] = []
        self.recorded: list[tuple[DraftRequest, DraftResult, int, list[str] | None]] = []

    async def draft(self, request: DraftRequest) -> DraftResult:
        self.requests.append(request)
        return DraftResult(register_name=request.register_name or "whatsapp.fr",
                           variants=[["désolée", "je peux pas vendredi"]],
                           used=[ANA, "wa-b2"])

    def record(self, request: DraftRequest, result: DraftResult, *, variant: int = 0,
               final: Sequence[str] | None = None) -> Path:
        self.recorded.append((request, result, variant, list(final) if final else None))
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        path = self.drafts_dir / f"draft-{len(self.recorded)}.md"
        path.write_text("\n\n".join(final or result.variants[variant]), encoding="utf-8")
        return path


class StubVoice:
    """Stands in for the voice lookup a later task adapts."""

    def registers(self) -> list[str]:
        return ["whatsapp.fr", "essay.en"]

    def card(self, register_name: str) -> str | None:
        if register_name == "whatsapp.fr":
            return "Short lowercase bursts, no final period."
        return None

    def samples(self, register_name: str, recipient: str | None, topic: str | None,
                k: int) -> list[Item]:
        return [ITEMS[1], ITEMS[3]][:k]


@pytest.fixture
def fake() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def make_mirror(world: Built, fake: FakeLLM):
    def build(**kwargs) -> Mirror:
        clock = kwargs.pop("clock", lambda: datetime(2026, 11, 14, 10, 32, 5))
        return Mirror(world.home, world.config, world.corpus, world.model, fake,
                      clock=clock, **kwargs)
    return build
