"""Builders and fixtures for store tests. All data is invented."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pytest

from repartee.schema import Chunk, Claim, ClaimStatus, Item, Kind, ReplyPair
from repartee.store.corpus import Corpus
from repartee.store.embed import HashEmbedder


def make_item(
    id: str,
    text: str,
    *,
    source: str = "whatsapp",
    kind: Kind = Kind.MESSAGE,
    container: str = "Ana Ruiz",
    author: str = "Ana Ruiz",
    is_self: bool = False,
    ts: datetime | None = None,
    seq: int = 0,
    holdout: bool = False,
) -> Item:
    """Build an Item with sensible defaults."""
    return Item(
        id=id, source=source, kind=kind, container=container, author=author,
        is_self=is_self, timestamp=ts, text=text, seq=seq, holdout=holdout,
    )


def make_chunk(
    id: str,
    text: str,
    *,
    source: str = "whatsapp",
    kind: Kind = Kind.MESSAGE,
    container: str = "Ana Ruiz",
    item_ids: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    participants: list[str] | None = None,
    self_text: str = "",
    title: str | None = None,
) -> Chunk:
    """Build a Chunk with sensible defaults."""
    return Chunk(
        id=id, source=source, kind=kind, container=container,
        item_ids=item_ids or [f"it-{id}"], start=start, end=end or start,
        participants=participants if participants is not None else ["Ana Ruiz"],
        text=text, self_text=self_text, title=title,
    )


def make_claim(id: str, text: str, *, page: str = "people/ana-ruiz",
               section: str = "Who", status: ClaimStatus = ClaimStatus.INFERRED) -> Claim:
    """Build a Claim."""
    return Claim(id=id, page=page, section=section, text=text, status=status)


def make_pair(id: str, context: str, reply: list[str], *, recipient: str = "Ana Ruiz",
              container: str = "Ana Ruiz") -> ReplyPair:
    """Build a ReplyPair."""
    return ReplyPair(id=id, container=container, recipient=recipient,
                     context=context, reply=reply)


class RecordingEmbedder:
    """HashEmbedder that records every text it embeds."""

    def __init__(self, dim: int = 256, name: str | None = None) -> None:
        self._inner = HashEmbedder(dim)
        self.name = name or self._inner.name
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return self._inner.embed(texts)

    @property
    def embedded(self) -> list[str]:
        return [t for call in self.calls for t in call]


@pytest.fixture
def corpus(tmp_path):
    c = Corpus(tmp_path / "corpus.db", embedder=HashEmbedder())
    yield c
    c.close()


@pytest.fixture
def keyword_corpus(tmp_path):
    c = Corpus(tmp_path / "corpus.db", embedder=None)
    yield c
    c.close()
