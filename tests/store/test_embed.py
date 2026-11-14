from __future__ import annotations

import numpy as np
import pytest

from repartee.config import EmbeddingConfig
from repartee.store import embed as embed_mod
from repartee.store.embed import (
    EmbedderUnavailable,
    FastEmbedEmbedder,
    HashEmbedder,
    get_embedder,
)


def test_hash_embedder_shape_norm_and_determinism():
    e = HashEmbedder(dim=64)
    v = e.embed(["bonjour tout le monde", "", "été"])
    assert v.shape == (3, 64) and v.dtype == np.float32
    assert np.isclose(np.linalg.norm(v[0]), 1.0)
    assert np.allclose(v[1], 0.0)  # empty text stays a zero vector, no NaN
    assert np.allclose(HashEmbedder(dim=64).embed(["été"]), v[2:3])
    assert e.embed([]).shape == (0, 64)
    assert e.name == "hash-64"


def test_hash_embedder_similarity_follows_ngrams():
    e = HashEmbedder()
    a, b, c = e.embed(["cycling through the city", "city cyclists", "quarterly tax filing"])
    assert float(a @ b) > float(a @ c)


def test_fastembed_constructor_does_not_load(monkeypatch):
    def boom(self):
        raise AssertionError("loaded eagerly")

    monkeypatch.setattr(FastEmbedEmbedder, "_load", boom)
    e = FastEmbedEmbedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    assert e.name.endswith("paraphrase-multilingual-MiniLM-L12-v2")
    assert e.dim == 384  # known from fastembed's model table, without loading


def test_get_embedder_returns_none_when_model_unavailable(monkeypatch):
    def fail(self):
        raise EmbedderUnavailable("download failed: no network")

    monkeypatch.setattr(FastEmbedEmbedder, "_load", fail)
    reasons: list[str] = []
    assert get_embedder(EmbeddingConfig(), reasons=reasons) is None
    assert "no network" in reasons[0]


def test_get_embedder_wraps_unexpected_loader_errors(monkeypatch):
    def fail(self):
        raise OSError("disk full")

    monkeypatch.setattr(embed_mod, "_new_text_embedding", lambda *a, **k: fail(None))
    assert get_embedder(EmbeddingConfig()) is None


def test_get_embedder_disabled():
    assert get_embedder(EmbeddingConfig(enabled=False)) is None


def test_default_model_is_supported_by_fastembed():
    from fastembed import TextEmbedding

    names = {m["model"] for m in TextEmbedding.list_supported_models()}
    assert EmbeddingConfig().model in names


@pytest.mark.live
def test_fastembed_real_model_live():
    e = get_embedder(EmbeddingConfig())
    assert e is not None
    v = e.embed(["Je vais au travail à vélo.", "I cycle to work.", "The tax form is due."])
    assert v.shape == (3, e.dim)
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])
