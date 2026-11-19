"""Lifetime-scale smoke test: generous time limits, catches accidental O(n^2) or per-row commits."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

from repartee.schema import SearchFilters
from repartee.store.corpus import Corpus
from repartee.store.embed import HashEmbedder

from .conftest import make_chunk, make_item

WORDS = ("river garden train coffee letter winter market violin harbour lantern "
         "orchard meadow pencil thunder cobalt saffron").split()


def test_large_corpus_sync_and_search_are_fast(tmp_path):
    base = datetime(2015, 1, 1)
    n_items, n_chunks = 120_000, 12_000
    items = [
        make_item(f"wa-{n}", f"{WORDS[n % 16]} {WORDS[(n * 7) % 16]} message {n}",
                  container=f"chat{n % 40}", seq=n, ts=base + timedelta(minutes=n),
                  is_self=n % 3 == 0)
        for n in range(n_items)
    ]
    chunks = [
        make_chunk(f"ch-{n}", " ".join(WORDS[(n + j) % 16] for j in range(40)) + f" #{n}",
                   container=f"chat{n % 40}", start=base + timedelta(hours=n),
                   item_ids=[f"wa-{n * 10 + j}" for j in range(10)])
        for n in range(n_chunks)
    ]
    c = Corpus(tmp_path / "big.db", embedder=HashEmbedder())

    t = time.perf_counter()
    c.sync_items(items)
    c.sync_chunks(chunks)
    first = time.perf_counter() - t

    t = time.perf_counter()
    again = c.sync_items(items)
    c.sync_chunks(chunks)
    resync = time.perf_counter() - t
    assert again.unchanged == n_items

    t = time.perf_counter()
    for q in ("river garden", "violin harbour lantern", "saffron"):
        assert c.search(q, k=10)
    c.search("cobalt", SearchFilters(participant="ana", after=base + timedelta(days=100)))
    searched = time.perf_counter() - t

    assert first < 60, first
    assert resync < 20, resync
    assert searched < 5, searched
    c.close()
