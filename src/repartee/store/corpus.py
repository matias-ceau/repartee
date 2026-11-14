"""The corpus index: one SQLite file with items, chunks, reply pairs and claims.

Keyword search uses FTS5; meaning search uses vectors from an Embedder, kept as float32
blobs and searched by brute force with numpy. Everything here is rebuildable from the
sources and the self-model, so a schema change simply rebuilds the file.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from repartee.schema import (
    PAGE_DIRS,
    Chunk,
    Claim,
    ClaimStatus,
    Item,
    PageKind,
    ReplyPair,
    SearchFilters,
    SearchHit,
    stable_hash,
)
from repartee.store.embed import Embedder, EmbedderUnavailable
from repartee.store.search import (
    chunk_filter_sql,
    fts_query,
    iso,
    item_filter_sql,
    name_key,
    name_param,
    names_column,
    query_terms,
    rrf,
    snippet,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1"
BATCH = 500  # ids per IN (...) query
EMBED_BATCH = 128  # texts per embedder call; each batch is committed
_TOKENIZER = "unicode61 remove_diacritics 2"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY, source TEXT NOT NULL, kind TEXT NOT NULL, container TEXT NOT NULL,
    author_key TEXT NOT NULL, is_self INTEGER NOT NULL, ts TEXT, seq INTEGER NOT NULL,
    holdout INTEGER NOT NULL, hash TEXT NOT NULL, text_hash TEXT NOT NULL, data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS items_order ON items (source, container, seq, id);
CREATE INDEX IF NOT EXISTS items_ts ON items (ts);

CREATE TABLE IF NOT EXISTS vectors (
    tbl TEXT NOT NULL, id TEXT NOT NULL, model TEXT NOT NULL, hash TEXT NOT NULL,
    vec BLOB NOT NULL, PRIMARY KEY (tbl, id, model)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, source TEXT NOT NULL,
    kind TEXT NOT NULL, container TEXT NOT NULL, start_ts TEXT, end_ts TEXT,
    people TEXT NOT NULL, has_self INTEGER NOT NULL, has_other INTEGER NOT NULL,
    title TEXT, text TEXT NOT NULL, pos INTEGER NOT NULL, hash TEXT NOT NULL,
    text_hash TEXT NOT NULL, embed_hash TEXT NOT NULL, data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_order ON chunks (source, container, pos, id);
CREATE TABLE IF NOT EXISTS chunk_items (
    item_id TEXT NOT NULL, chunk_id TEXT NOT NULL, PRIMARY KEY (item_id, chunk_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS chunk_items_chunk ON chunk_items (chunk_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5 (
    title, text, content='chunks', content_rowid='rowid', tokenize='{_TOKENIZER}'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts (rowid, title, text) VALUES (new.rowid, new.title, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, title, text)
        VALUES ('delete', old.rowid, old.title, old.text);
    DELETE FROM vectors WHERE tbl = 'chunk' AND id = old.id;
    DELETE FROM chunk_items WHERE chunk_id = old.id;
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF title, text ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, title, text)
        VALUES ('delete', old.rowid, old.title, old.text);
    INSERT INTO chunks_fts (rowid, title, text) VALUES (new.rowid, new.title, new.text);
END;

CREATE TABLE IF NOT EXISTS claims (
    rowid INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, page TEXT NOT NULL,
    page_kind TEXT NOT NULL, section TEXT NOT NULL, status TEXT NOT NULL, text TEXT NOT NULL,
    hash TEXT NOT NULL, embed_hash TEXT NOT NULL, data TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5 (
    text, section, page, content='claims', content_rowid='rowid', tokenize='{_TOKENIZER}'
);
CREATE TRIGGER IF NOT EXISTS claims_ai AFTER INSERT ON claims BEGIN
    INSERT INTO claims_fts (rowid, text, section, page)
        VALUES (new.rowid, new.text, new.section, new.page);
END;
CREATE TRIGGER IF NOT EXISTS claims_ad AFTER DELETE ON claims BEGIN
    INSERT INTO claims_fts (claims_fts, rowid, text, section, page)
        VALUES ('delete', old.rowid, old.text, old.section, old.page);
    DELETE FROM vectors WHERE tbl = 'claim' AND id = old.id;
END;
CREATE TRIGGER IF NOT EXISTS claims_au AFTER UPDATE OF text, section, page ON claims BEGIN
    INSERT INTO claims_fts (claims_fts, rowid, text, section, page)
        VALUES ('delete', old.rowid, old.text, old.section, old.page);
    INSERT INTO claims_fts (rowid, text, section, page)
        VALUES (new.rowid, new.text, new.section, new.page);
END;

CREATE TABLE IF NOT EXISTS pairs (
    rowid INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, container TEXT NOT NULL,
    recipient_key TEXT NOT NULL, ts TEXT, context TEXT NOT NULL,
    hash TEXT NOT NULL, embed_hash TEXT NOT NULL, data TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS pairs_fts USING fts5 (
    context, content='pairs', content_rowid='rowid', tokenize='{_TOKENIZER}'
);
CREATE TRIGGER IF NOT EXISTS pairs_ai AFTER INSERT ON pairs BEGIN
    INSERT INTO pairs_fts (rowid, context) VALUES (new.rowid, new.context);
END;
CREATE TRIGGER IF NOT EXISTS pairs_ad AFTER DELETE ON pairs BEGIN
    INSERT INTO pairs_fts (pairs_fts, rowid, context) VALUES ('delete', old.rowid, old.context);
    DELETE FROM vectors WHERE tbl = 'pair' AND id = old.id;
END;
CREATE TRIGGER IF NOT EXISTS pairs_au AFTER UPDATE OF context ON pairs BEGIN
    INSERT INTO pairs_fts (pairs_fts, rowid, context) VALUES ('delete', old.rowid, old.context);
    INSERT INTO pairs_fts (rowid, context) VALUES (new.rowid, new.context);
END;
"""

_TABLES = ("items", "vectors", "chunks", "chunk_items", "chunks_fts", "claims", "claims_fts",
           "pairs", "pairs_fts", "meta")

# Embeddable tables: vector tag -> (table, columns of the embedded text, filter alias).
_EMBEDDABLE = {"chunk": ("chunks", "title, text", "c"), "claim": ("claims", "text", "cl"),
               "pair": ("pairs", "context", "p")}

_PAGE_KIND_BY_DIR = {d: k for k, d in PAGE_DIRS.items()}


class SyncStats(BaseModel):
    """What a sync changed."""

    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    remapped: dict[str, str] = Field(default_factory=dict)  # vanished id -> new id, same text


def _batched(values: Sequence[Any], size: int = BATCH) -> Iterator[Sequence[Any]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _text_key(text: str) -> str:
    stripped = text.strip()
    return stable_hash(stripped) if stripped else ""


def _chunk_embed_text(title: str | None, text: str) -> str:
    return f"{title}\n{text}" if title else text


def page_kind_of(page: str) -> PageKind:
    """Page kind from a page key ("people/ana" -> PERSON, "core" -> CORE, else META)."""
    if page == "core":
        return PageKind.CORE
    return _PAGE_KIND_BY_DIR.get(page.split("/", 1)[0], PageKind.META)


class Corpus:
    """SQLite-backed index of items, chunks, reply pairs and self-model claims."""

    def __init__(self, path: Path | str, embedder: Embedder | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder: Embedder | None = embedder
        self.embedder_error: str | None = None
        self._lock = threading.RLock()
        self._matrices: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._init_schema()

    # ------------------------------------------------------------------ plumbing

    @property
    def embedder_name(self) -> str | None:
        """Name of the active embedder, None when search is keyword only."""
        return self.embedder.name if self.embedder is not None else None

    def _init_schema(self) -> None:
        row = None
        try:
            row = self._conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        except sqlite3.OperationalError:
            pass
        if row is not None and row[0] != SCHEMA_VERSION:
            log.info("corpus schema %s is outdated, rebuilding %s", row[0], self.path)
            for name in ("chunks_ai", "chunks_ad", "chunks_au", "claims_ai", "claims_ad",
                         "claims_au", "pairs_ai", "pairs_ad", "pairs_au"):
                self._conn.execute(f"DROP TRIGGER IF EXISTS {name}")
            for name in _TABLES:
                self._conn.execute(f"DROP TABLE IF EXISTS {name}")
            self._conn.commit()
        self._conn.executescript(_SCHEMA)
        with self._conn:
            self._conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema', ?)",
                               (SCHEMA_VERSION,))

    def _disable_embedder(self, reason: str) -> None:
        log.warning("embeddings unavailable, search is keyword only: %s", reason)
        self.embedder = None
        self.embedder_error = reason
        self._matrices.clear()

    def close(self) -> None:
        """Close the database."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Corpus:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------------- sync

    def _diff(self, table: str, rows: dict[str, tuple[str, tuple[Any, ...]]]
              ) -> tuple[list[str], list[str], list[str], int]:
        """(added ids, changed ids, removed ids, unchanged count) against the table."""
        existing = dict(self._conn.execute(f"SELECT id, hash FROM {table}"))
        added = [i for i in rows if i not in existing]
        changed = [i for i in rows if i in existing and existing[i] != rows[i][0]]
        removed = [i for i in existing if i not in rows]
        unchanged = len(rows) - len(added) - len(changed)
        return added, changed, removed, unchanged

    def _write(self, table: str, columns: Sequence[str], ids: list[str], removed: list[str],
               rows: dict[str, tuple[str, tuple[Any, ...]]]) -> None:
        if removed:
            self._conn.executemany(f"DELETE FROM {table} WHERE id = ?",
                                   ((i,) for i in removed))
        if ids:
            cols = ", ".join(columns)
            marks = ", ".join("?" * len(columns))
            updates = ", ".join(f"{c} = excluded.{c}" for c in columns if c != "id")
            self._conn.executemany(
                f"INSERT INTO {table} ({cols}) VALUES ({marks}) "
                f"ON CONFLICT (id) DO UPDATE SET {updates}",
                (rows[i][1] for i in ids),
            )

    _ITEM_COLUMNS = ("id", "source", "kind", "container", "author_key", "is_self", "ts", "seq",
                     "holdout", "hash", "text_hash", "data")

    def sync_items(self, items: Iterable[Item]) -> SyncStats:
        """Make the stored items equal to `items` (the complete current set)."""
        rows: dict[str, tuple[str, tuple[Any, ...]]] = {}
        meta: dict[str, tuple[str, str, str, int]] = {}
        for item in items:
            data = item.model_dump_json()
            h = stable_hash(data)
            tk = _text_key(item.text)
            rows[item.id] = (h, (item.id, item.source, str(item.kind), item.container,
                                 f" {name_key(item.author)} ", int(item.is_self),
                                 iso(item.timestamp), item.seq, int(item.holdout), h, tk, data))
            meta[item.id] = (item.source, tk, item.container, item.seq)
        with self._lock, self._conn:
            added, changed, removed, unchanged = self._diff("items", rows)
            remapped = self._remap(removed, [(i, *meta[i]) for i in added])
            self._write("items", self._ITEM_COLUMNS, added + changed, removed, rows)
        return SyncStats(added=len(added), updated=len(changed), removed=len(removed),
                         unchanged=unchanged, remapped=remapped)

    def _remap(self, removed: list[str], added: list[tuple[str, str, str, str, int]]
               ) -> dict[str, str]:
        """Pair each removed id with an added id of the same source and identical text."""
        if not removed or not added:
            return {}
        gone: dict[tuple[str, str], list[tuple[str, int, str]]] = defaultdict(list)
        for batch in _batched(removed):
            marks = ",".join("?" * len(batch))
            for id_, source, tk, container, seq in self._conn.execute(
                f"SELECT id, source, text_hash, container, seq FROM items WHERE id IN ({marks})",
                batch,
            ):
                if tk:
                    gone[(source, tk)].append((container, seq, id_))
        new: dict[tuple[str, str], list[tuple[str, int, str]]] = defaultdict(list)
        for id_, source, tk, container, seq in added:
            if tk and (source, tk) in gone:
                new[(source, tk)].append((container, seq, id_))
        mapping: dict[str, str] = {}
        for key, candidates in new.items():
            for old, fresh in zip(sorted(gone[key]), sorted(candidates), strict=False):
                mapping[old[2]] = fresh[2]
        return mapping

    _CHUNK_COLUMNS = ("id", "source", "kind", "container", "start_ts", "end_ts", "people",
                      "has_self", "has_other", "title", "text", "pos", "hash", "text_hash",
                      "embed_hash", "data")

    def sync_chunks(self, chunks: Iterable[Chunk]) -> SyncStats:
        """Make the stored chunks equal to `chunks`; embeds only new or changed text."""
        rows: dict[str, tuple[str, tuple[Any, ...]]] = {}
        item_links: dict[str, list[str]] = {}
        for pos, chunk in enumerate(chunks):
            data = chunk.model_dump(mode="json", exclude={"text"})
            data_json = json.dumps(data, ensure_ascii=False, sort_keys=True)
            h = stable_hash(data_json, chunk.text)
            has_self = bool(chunk.self_text.strip())
            rows[chunk.id] = (h, (
                chunk.id, chunk.source, str(chunk.kind), chunk.container, iso(chunk.start),
                iso(chunk.end), names_column(chunk.participants), int(has_self),
                int(len(chunk.participants) > 1 or not has_self), chunk.title, chunk.text, pos,
                h, stable_hash(chunk.text),
                stable_hash(_chunk_embed_text(chunk.title, chunk.text)), data_json,
            ))
            item_links[chunk.id] = list(dict.fromkeys(chunk.item_ids))
        with self._lock:
            with self._conn:
                added, changed, removed, unchanged = self._diff("chunks", rows)
                self._write("chunks", self._CHUNK_COLUMNS, added + changed, removed, rows)
                touched = added + changed
                self._conn.executemany("DELETE FROM chunk_items WHERE chunk_id = ?",
                                       ((i,) for i in changed))
                self._conn.executemany(
                    "INSERT OR IGNORE INTO chunk_items (item_id, chunk_id) VALUES (?, ?)",
                    ((item_id, cid) for cid in touched for item_id in item_links[cid]),
                )
                # Keep input order for undated neighbours without counting it as a change.
                self._conn.executemany(
                    "UPDATE chunks SET pos = ? WHERE id = ? AND pos != ?",
                    ((rows[i][1][11], i, rows[i][1][11]) for i in rows),
                )
            self._after_sync("chunk")
        return SyncStats(added=len(added), updated=len(changed), removed=len(removed),
                         unchanged=unchanged)

    _PAIR_COLUMNS = ("id", "container", "recipient_key", "ts", "context", "hash", "embed_hash",
                     "data")

    def sync_reply_pairs(self, pairs: Iterable[ReplyPair]) -> SyncStats:
        """Make the stored reply pairs equal to `pairs`; embeds new or changed contexts."""
        rows: dict[str, tuple[str, tuple[Any, ...]]] = {}
        for pair in pairs:
            data = pair.model_dump_json()
            h = stable_hash(data)
            rows[pair.id] = (h, (pair.id, pair.container, names_column([pair.recipient]),
                                 iso(pair.timestamp), pair.context, h,
                                 stable_hash(pair.context), data))
        return self._sync_simple("pairs", "pair", self._PAIR_COLUMNS, rows)

    _CLAIM_COLUMNS = ("id", "page", "page_kind", "section", "status", "text", "hash",
                      "embed_hash", "data")

    def sync_claims(self, claims: Iterable[Claim]) -> SyncStats:
        """Make the indexed claims equal to `claims` (all of the self-model's claims)."""
        rows: dict[str, tuple[str, tuple[Any, ...]]] = {}
        for claim in claims:
            data = claim.model_dump_json()
            h = stable_hash(data)
            rows[claim.id] = (h, (claim.id, claim.page, str(page_kind_of(claim.page)),
                                  claim.section, str(claim.status), claim.text, h,
                                  stable_hash(claim.text), data))
        return self._sync_simple("claims", "claim", self._CLAIM_COLUMNS, rows)

    def _sync_simple(self, table: str, tag: str, columns: Sequence[str],
                     rows: dict[str, tuple[str, tuple[Any, ...]]]) -> SyncStats:
        with self._lock:
            with self._conn:
                added, changed, removed, unchanged = self._diff(table, rows)
                self._write(table, columns, added + changed, removed, rows)
            self._after_sync(tag)
        return SyncStats(added=len(added), updated=len(changed), removed=len(removed),
                         unchanged=unchanged)

    def _after_sync(self, tag: str) -> None:
        self._matrices.pop(tag, None)
        self._embed_pending(tag)

    def _embed_pending(self, tag: str) -> None:
        """Embed rows whose vector for the current embedder is missing or stale."""
        if self.embedder is None:
            return
        table, cols, _ = _EMBEDDABLE[tag]
        model = self.embedder.name
        last = 0
        try:
            while True:
                rows = self._conn.execute(
                    f"SELECT t.rowid, t.id, t.embed_hash, {cols} FROM {table} t "
                    "LEFT JOIN vectors v ON v.tbl = ? AND v.id = t.id AND v.model = ? "
                    "WHERE t.rowid > ? AND (v.hash IS NULL OR v.hash != t.embed_hash) "
                    "ORDER BY t.rowid LIMIT ?",
                    (tag, model, last, EMBED_BATCH),
                ).fetchall()
                if not rows:
                    break
                last = rows[-1][0]
                if tag == "chunk":
                    texts = [_chunk_embed_text(r[3], r[4]) for r in rows]
                else:
                    texts = [r[3] for r in rows]
                vectors = np.asarray(self.embedder.embed(texts), dtype=np.float32)
                with self._conn:
                    self._conn.executemany(
                        "INSERT OR REPLACE INTO vectors (tbl, id, model, hash, vec) "
                        "VALUES (?, ?, ?, ?, ?)",
                        ((tag, r[1], model, r[2], vectors[n].tobytes())
                         for n, r in enumerate(rows)),
                    )
        except EmbedderUnavailable as exc:
            self._disable_embedder(str(exc))
            return
        with self._conn:
            self._conn.execute("DELETE FROM vectors WHERE tbl = ? AND model != ?", (tag, model))
        self._matrices.pop(tag, None)

    # --------------------------------------------------------------------- reads

    def get_item(self, item_id: str) -> Item | None:
        """One item by id."""
        found = self.get_items([item_id])
        return found[0] if found else None

    def get_items(self, item_ids: Iterable[str]) -> list[Item]:
        """Items by id, in the order asked, skipping unknown ids."""
        ids = list(dict.fromkeys(item_ids))
        found: dict[str, str] = {}
        with self._lock:
            for batch in _batched(ids):
                marks = ",".join("?" * len(batch))
                found.update(self._conn.execute(
                    f"SELECT id, data FROM items WHERE id IN ({marks})", batch))
        return [Item.model_validate_json(found[i]) for i in ids if i in found]

    def existing_item_ids(self, item_ids: Iterable[str]) -> set[str]:
        """The subset of `item_ids` that the corpus holds."""
        ids = list(dict.fromkeys(item_ids))
        out: set[str] = set()
        with self._lock:
            for batch in _batched(ids):
                marks = ",".join("?" * len(batch))
                out.update(r[0] for r in self._conn.execute(
                    f"SELECT id FROM items WHERE id IN ({marks})", batch))
        return out

    @staticmethod
    def _chunk_from_row(text: str, data: str) -> Chunk:
        return Chunk.model_validate({**json.loads(data), "text": text})

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """One chunk by id."""
        with self._lock:
            row = self._conn.execute("SELECT text, data FROM chunks WHERE id = ?",
                                     (chunk_id,)).fetchone()
        return self._chunk_from_row(*row) if row else None

    def chunk_for_item(self, item_id: str) -> Chunk | None:
        """The (first) chunk that contains an item."""
        with self._lock:
            row = self._conn.execute(
                "SELECT c.text, c.data FROM chunk_items ci JOIN chunks c ON c.id = ci.chunk_id "
                "WHERE ci.item_id = ? ORDER BY c.pos LIMIT 1", (item_id,)).fetchone()
        return self._chunk_from_row(*row) if row else None

    def chunk_hash(self, chunk_id: str) -> str | None:
        """stable_hash of a chunk's text; changes exactly when the text changes."""
        with self._lock:
            row = self._conn.execute("SELECT text_hash FROM chunks WHERE id = ?",
                                     (chunk_id,)).fetchone()
        return row[0] if row else None

    def iter_items(self, filters: SearchFilters | None = None, *,
                   holdout: bool | None = None) -> Iterator[Item]:
        """Stream items by source, container and seq. holdout: None any, True/False only."""
        where, params = item_filter_sql(filters)
        if holdout is not None:
            where += " AND i.holdout = ?"
            params.append(int(holdout))
        key: tuple[Any, ...] = ("", "", -(2**62), "")
        while True:
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT i.source, i.container, i.seq, i.id, i.data FROM items i "
                    f"WHERE {where} AND (i.source, i.container, i.seq, i.id) > (?, ?, ?, ?) "
                    "ORDER BY i.source, i.container, i.seq, i.id LIMIT 1000",
                    [*params, *key]).fetchall()
            if not rows:
                return
            key = rows[-1][:4]
            for row in rows:
                yield Item.model_validate_json(row[4])

    def iter_chunks(self, filters: SearchFilters | None = None) -> Iterator[Chunk]:
        """Stream chunks by source, container and input order."""
        where, params = chunk_filter_sql(filters)
        key: tuple[Any, ...] = ("", "", -1, "")
        while True:
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT c.source, c.container, c.pos, c.id, c.text, c.data FROM chunks c "
                    f"WHERE {where} AND (c.source, c.container, c.pos, c.id) > (?, ?, ?, ?) "
                    "ORDER BY c.source, c.container, c.pos, c.id LIMIT 500",
                    [*params, *key]).fetchall()
            if not rows:
                return
            key = rows[-1][:4]
            for row in rows:
                yield self._chunk_from_row(row[4], row[5])

    def neighbors(self, chunk_id: str, before: int = 1, after: int = 1) -> list[Chunk]:
        """Chunks around one in the same container, ordered by start, itself included."""
        with self._lock:
            row = self._conn.execute("SELECT source, container FROM chunks WHERE id = ?",
                                     (chunk_id,)).fetchone()
            if row is None:
                return []
            ids = [r[0] for r in self._conn.execute(
                "SELECT id FROM chunks WHERE source = ? AND container = ? "
                "ORDER BY start_ts IS NULL, start_ts, pos", row)]
            at = ids.index(chunk_id)
            wanted = ids[max(0, at - max(0, before)) : at + max(0, after) + 1]
            marks = ",".join("?" * len(wanted))
            found = {r[0]: self._chunk_from_row(r[1], r[2]) for r in self._conn.execute(
                f"SELECT id, text, data FROM chunks WHERE id IN ({marks})", wanted)}
        return [found[i] for i in wanted]

    # -------------------------------------------------------------------- search

    def _matrix(self, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
        """(rowids, vectors) of the current embedder for a table, cached until next sync."""
        if self.embedder is None:
            return None
        if tag not in self._matrices:
            table, _, _ = _EMBEDDABLE[tag]
            rows = self._conn.execute(
                f"SELECT t.rowid, v.vec FROM {table} t JOIN vectors v "
                "ON v.tbl = ? AND v.id = t.id AND v.model = ? AND v.hash = t.embed_hash "
                "ORDER BY t.rowid", (tag, self.embedder.name)).fetchall()
            if rows:
                size = len(rows[0][1])
                rows = [r for r in rows if len(r[1]) == size]
                rowids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
                matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32)
                self._matrices[tag] = (rowids, matrix.reshape(len(rows), size // 4))
            else:
                self._matrices[tag] = None
        return self._matrices[tag]

    def _semantic(self, tag: str, query: str, where: str, params: list[Any], n: int
                  ) -> list[tuple[int, float]] | None:
        """Best rowids by cosine similarity; None when meaning search is not possible."""
        loaded = self._matrix(tag)
        if loaded is None or self.embedder is None:
            return None
        if not query.strip():
            return []
        rowids, matrix = loaded
        try:
            q = np.asarray(self.embedder.embed([query]), dtype=np.float32)[0]
        except EmbedderUnavailable as exc:
            self._disable_embedder(str(exc))
            return None
        if q.shape[0] != matrix.shape[1] or not np.any(q):
            return []
        scores = matrix @ q
        if where != "1":
            table, _, alias = _EMBEDDABLE[tag]
            allowed = np.fromiter(
                (r[0] for r in self._conn.execute(
                    f"SELECT {alias}.rowid FROM {table} {alias} WHERE {where}", params)),
                dtype=np.int64)
            scores = np.where(np.isin(rowids, allowed), scores, -np.inf)
        n = min(n, len(scores))
        if n <= 0:
            return []
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(rowids[i]), float(scores[i])) for i in top if np.isfinite(scores[i])]

    def _keyword(self, fts: str, table: str, alias: str, weights: str, query: str,
                 where: str, params: list[Any], n: int) -> list[tuple[int, float]]:
        match = fts_query(query)
        if match is None:
            return []
        try:
            rows = self._conn.execute(
                f"SELECT {alias}.rowid, bm25({fts}{weights}) AS r FROM {fts} "
                f"JOIN {table} {alias} ON {alias}.rowid = {fts}.rowid "
                f"WHERE {fts} MATCH ? AND {where} ORDER BY r LIMIT ?",
                [match, *params, n]).fetchall()
        except sqlite3.OperationalError as exc:  # never let query text break search
            log.debug("fts query failed for %r: %s", query, exc)
            return []
        return [(r[0], -float(r[1])) for r in rows]

    def _rank(self, tag: str, fts: str, table: str, alias: str, weights: str, query: str,
              where: str, params: list[Any], k: int, mode: str) -> list[tuple[int, float]]:
        n = max(4 * k, 50)
        semantic = None
        if mode != "keyword":
            semantic = self._semantic(tag, query, where, params, n)
        if semantic is None or mode == "keyword":
            return self._keyword(fts, table, alias, weights, query, where, params, k)
        if mode == "semantic":
            return semantic[:k]
        keyword = self._keyword(fts, table, alias, weights, query, where, params, n)
        fused = rrf([[r for r, _ in keyword], [r for r, _ in semantic]])
        return fused[:k]

    def search(self, query: str, filters: SearchFilters | None = None, k: int = 10,
               mode: Literal["hybrid", "keyword", "semantic"] = "hybrid") -> list[SearchHit]:
        """Search chunks. Without an embedder every mode behaves as keyword."""
        if k <= 0:
            return []
        where, params = chunk_filter_sql(filters)
        with self._lock:
            ranked = self._rank("chunk", "chunks_fts", "chunks", "c", ", 2.0, 1.0", query,
                                where, params, k, mode)
            if not ranked:
                return []
            marks = ",".join("?" * len(ranked))
            rows = {r[0]: (r[1], r[2]) for r in self._conn.execute(
                f"SELECT rowid, text, data FROM chunks WHERE rowid IN ({marks})",
                [r for r, _ in ranked])}
        terms = query_terms(query)
        hits = []
        for rowid, score in ranked:
            text, data = rows[rowid]
            hits.append(SearchHit(chunk=self._chunk_from_row(text, data), score=score,
                                  snippet=snippet(text, terms)))
        return hits

    def search_claims(self, query: str, kind: PageKind | None = None,
                      k: int = 10) -> list[tuple[Claim, float]]:
        """Search self-model claims (rejected ones never come back)."""
        if k <= 0:
            return []
        where, params = f"cl.status != '{ClaimStatus.REJECTED}'", []
        if kind is not None:
            where += " AND cl.page_kind = ?"
            params.append(str(kind))
        with self._lock:
            ranked = self._rank("claim", "claims_fts", "claims", "cl", ", 2.0, 0.5, 0.5",
                                query, where, params, k, "hybrid")
            if not ranked:
                return []
            marks = ",".join("?" * len(ranked))
            rows = dict(self._conn.execute(
                f"SELECT rowid, data FROM claims WHERE rowid IN ({marks})",
                [r for r, _ in ranked]))
        return [(Claim.model_validate_json(rows[r]), s) for r, s in ranked]

    def search_reply_pairs(self, context: str, recipient: str | None = None,
                           k: int = 5) -> list[ReplyPair]:
        """Reply pairs whose incoming context resembles `context`, optionally for one person."""
        if k <= 0:
            return []
        where, params = "1", []
        needle = name_param(recipient) if recipient else None
        if needle:
            where, params = "instr(p.recipient_key, ?) > 0", [needle]
        with self._lock:
            ranked = self._semantic("pair", context, where, params, k)
            if ranked is None:
                ranked = self._keyword("pairs_fts", "pairs", "p", "", context, where, params, k)
            if not ranked:
                return []
            marks = ",".join("?" * len(ranked))
            rows = dict(self._conn.execute(
                f"SELECT rowid, data FROM pairs WHERE rowid IN ({marks})",
                [r for r, _ in ranked]))
        return [ReplyPair.model_validate_json(rows[r]) for r, _ in ranked]

    # --------------------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        """Counts and coverage, for `status` and the TUI."""
        from repartee.store.search import from_iso

        with self._lock:
            one = lambda sql, *p: self._conn.execute(sql, p).fetchone()[0]  # noqa: E731
            first, last = self._conn.execute("SELECT MIN(ts), MAX(ts) FROM items").fetchone()
            if self.embedder is not None:
                embedded = one(
                    "SELECT COUNT(*) FROM chunks c JOIN vectors v ON v.tbl = 'chunk' "
                    "AND v.id = c.id AND v.model = ? AND v.hash = c.embed_hash",
                    self.embedder.name)
            else:
                embedded = one("SELECT COUNT(DISTINCT id) FROM vectors WHERE tbl = 'chunk'")
            return {
                "items": one("SELECT COUNT(*) FROM items"),
                "chunks": one("SELECT COUNT(*) FROM chunks"),
                "by_source": dict(self._conn.execute(
                    "SELECT source, COUNT(*) FROM items GROUP BY source ORDER BY source")),
                "by_kind": dict(self._conn.execute(
                    "SELECT kind, COUNT(*) FROM items GROUP BY kind ORDER BY kind")),
                "self_items": one("SELECT COUNT(*) FROM items WHERE is_self = 1"),
                "holdout_items": one("SELECT COUNT(*) FROM items WHERE holdout = 1"),
                "embedded_chunks": embedded,
                "embedder": self.embedder_name,
                "embedder_error": self.embedder_error,
                "search": "hybrid" if self.embedder is not None else "keyword",
                "first": from_iso(first),
                "last": from_iso(last),
                "claims": one("SELECT COUNT(*) FROM claims"),
                "reply_pairs": one("SELECT COUNT(*) FROM pairs"),
            }
