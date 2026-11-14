"""Text embedders: a lazy fastembed wrapper and a deterministic hashing one for tests."""

from __future__ import annotations

import logging
import unicodedata
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from repartee.config import APP, EmbeddingConfig, xdg_cache_home

log = logging.getLogger(__name__)


class EmbedderUnavailable(Exception):
    """The embedding model cannot be loaded or run (not installed, not downloadable, ...)."""


class Embedder(Protocol):
    """Turns texts into L2-normalized float32 vectors of a fixed dimension."""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row as float32; all-zero rows stay zero."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


def fold(text: str) -> str:
    """Case- and diacritic-insensitive form of a text (same length is not guaranteed)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


class HashEmbedder:
    """Deterministic character n-gram hashing embedder (3- and 4-grams of each word)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts; identical input gives identical output across processes."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in fold(text).split():
                padded = f"<{word}>"
                for n in (3, 4):
                    for i in range(max(1, len(padded) - n + 1)):
                        h = zlib.crc32(padded[i : i + n].encode("utf-8"))
                        out[row, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        return normalize_rows(out) if len(texts) else out


def _new_text_embedding(model_name: str, cache_dir: Path | None) -> Any:
    from fastembed import TextEmbedding

    kwargs: dict[str, Any] = {"model_name": model_name}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs["cache_dir"] = str(cache_dir)
    return TextEmbedding(**kwargs)


def default_model_cache() -> Path:
    """Where downloaded embedding models live (fastembed defaults to a temp dir)."""
    return xdg_cache_home() / APP / "models"


class FastEmbedEmbedder:
    """Local ONNX embedder via fastembed. The model loads on first use, not on construction."""

    def __init__(self, model_name: str, *, cache_dir: Path | None = None,
                 batch_size: int = 64) -> None:
        self.model_name = model_name
        self.name = f"fastembed:{model_name}"
        self.cache_dir = cache_dir if cache_dir is not None else default_model_cache()
        self.batch_size = batch_size
        self._model: Any = None
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        """Vector size, from fastembed's model table when possible (no load needed)."""
        if self._dim is None:
            try:
                from fastembed import TextEmbedding

                for desc in TextEmbedding.list_supported_models():
                    if desc.get("model") == self.model_name and desc.get("dim"):
                        self._dim = int(desc["dim"])
                        break
            except Exception:  # fall back to probing the model
                pass
        if self._dim is None:
            self._dim = int(self.embed(["dim"]).shape[1])
        return self._dim

    def _load(self) -> Any:
        try:
            return _new_text_embedding(self.model_name, self.cache_dir)
        except Exception as exc:  # any loader failure means "unavailable"
            raise EmbedderUnavailable(
                f"cannot load embedding model {self.model_name}: {exc}"
            ) from exc

    def load(self) -> None:
        """Load the model now; raises EmbedderUnavailable."""
        if self._model is None:
            self._model = self._load()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts; raises EmbedderUnavailable when the model cannot be loaded or run."""
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        self.load()
        try:
            vectors = list(self._model.embed(list(texts), batch_size=self.batch_size))
        except Exception as exc:
            raise EmbedderUnavailable(f"embedding failed with {self.model_name}: {exc}") from exc
        out = normalize_rows(np.vstack(vectors))
        self._dim = out.shape[1]
        return out


def get_embedder(config: EmbeddingConfig, *, reasons: list[str] | None = None
                 ) -> Embedder | None:
    """The configured embedder, loaded; None when disabled or unavailable.

    Loading may download the model on first use. When it fails, the reason is logged and
    appended to `reasons` if given, so callers can show it in a status line.
    """
    if not config.enabled:
        return None
    embedder = FastEmbedEmbedder(config.model)
    try:
        embedder.load()
    except EmbedderUnavailable as exc:
        log.warning("embeddings unavailable, search is keyword only: %s", exc)
        if reasons is not None:
            reasons.append(str(exc))
        return None
    return embedder
