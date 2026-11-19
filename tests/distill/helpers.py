"""Builders and fixtures for distill tests: a temp home, an index without embedder, a FakeLLM.

All people and events are invented.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest
import regex

from repartee.config import HomeConfig
from repartee.distill import Distiller
from repartee.distill.extract import ExtractedObservation, ExtractionResult
from repartee.home import Home
from repartee.ingest.chunking import chunk_items
from repartee.llm import FakeLLM, LLMRequest
from repartee.scaffold import init_home
from repartee.schema import Item, Kind
from repartee.selfmodel import SelfModel
from repartee.store.corpus import Corpus

OWNER = "Camille Martin"
IDS = regex.compile(r"^\[([a-z]{2}-[\w-]+)\]", regex.MULTILINE)


@pytest.fixture(autouse=True)
def _isolated_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        path = tmp_path / "_xdg" / var.lower()
        path.mkdir(parents=True)
        monkeypatch.setenv(var, str(path))


@dataclass
class Env:
    home: Home
    config: HomeConfig
    corpus: Corpus
    model: SelfModel
    fake: FakeLLM
    items: list[Item]

    def distiller(self) -> Distiller:
        """A fresh Distiller over the env (reloads the self-model and derived state)."""
        self.model = SelfModel(self.home.self_dir)
        self.model.load()
        return Distiller(self.home, self.config, self.corpus, self.model, self.fake)

    def index(self, items: Sequence[Item]) -> None:
        """Make the index hold exactly these items and their chunks."""
        self.items = list(items)
        self.corpus.sync_items(self.items)
        self.corpus.sync_chunks(chunk_items(self.items, self.config.chunking))


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    home = init_home(tmp_path / "home", OWNER, git=False)
    config = home.load_config()
    config.owner.aliases = ["Cam"]
    corpus = Corpus(tmp_path / "index.db")
    model = SelfModel(home.self_dir)
    model.load()
    yield Env(home, config, corpus, model, FakeLLM(), [])
    corpus.close()


def msg(id: str, author: str, text: str, ts: datetime, *, container: str = "Ana Ruiz",
        seq: int = 0) -> Item:
    """A chat message."""
    return Item(id=id, source="whatsapp", kind=Kind.MESSAGE, container=container,
                author=author, is_self=author == OWNER, timestamp=ts, text=text, seq=seq)


def entry(id: str, text: str, *, kind: Kind = Kind.JOURNAL, container: str = "Journal/x.md",
          source: str = "vault", ts: datetime | None = None, seq: int = 0,
          author: str = OWNER, title: str | None = None) -> Item:
    """A note, journal entry or interview turn."""
    return Item(id=id, source=source, kind=kind, container=container, author=author,
                is_self=author == OWNER, timestamp=ts, text=text, seq=seq, title=title)


def ids_in(request: LLMRequest) -> list[str]:
    """The item ids cited in the chunk of an extract request, in order."""
    return IDS.findall(request.messages[-1].text)


def obs(subject_kind: str, subject: str, text: str, evidence: list[str], *,
        confidence: float = 1.0, when: str | None = None) -> ExtractedObservation:
    """A model-side observation."""
    return ExtractedObservation(subject_kind=subject_kind, subject=subject, text=text,
                                evidence=evidence, confidence=confidence, when=when)


def one_per_chunk(subject_kind: str = "topic", subject: str = "cycling",
                  confidence: float = 1.0) -> Callable[[LLMRequest], ExtractionResult]:
    """An extract responder: one observation citing the chunk's first id."""
    def reply(request: LLMRequest) -> ExtractionResult:
        first = ids_in(request)[0]
        return ExtractionResult(observations=[
            obs(subject_kind, subject, f"I wrote about it in {first}.", [first],
                confidence=confidence)])
    return reply
