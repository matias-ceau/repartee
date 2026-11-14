"""Corpus store: items, chunks, reply pairs and claims in one SQLite file, with search."""

from repartee.store.corpus import Corpus, SyncStats
from repartee.store.embed import (
    Embedder,
    EmbedderUnavailable,
    FastEmbedEmbedder,
    HashEmbedder,
    get_embedder,
)

__all__ = [
    "Corpus",
    "Embedder",
    "EmbedderUnavailable",
    "FastEmbedEmbedder",
    "HashEmbedder",
    "SyncStats",
    "get_embedder",
]
