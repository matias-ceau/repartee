"""Query helpers for the corpus: safe FTS queries, snippets, rank fusion and filter SQL."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

import regex

from repartee.schema import SearchFilters
from repartee.store.embed import fold

RRF_K = 60
MAX_QUERY_TERMS = 64
SNIPPET_WIDTH = 240

_WORD = regex.compile(r"[\p{L}\p{M}\p{N}]+")
_SPACE = regex.compile(r"\s+")


def fts_query(text: str) -> str | None:
    """An FTS5 MATCH expression for free text: every word quoted, OR-ed. None if no words.

    Only letters, marks and digits survive, so no user input can reach FTS syntax.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for word in _WORD.findall(text):
        key = fold(word)
        if key not in seen:
            seen.add(key)
            terms.append(f'"{word}"')
        if len(terms) >= MAX_QUERY_TERMS:
            break
    return " OR ".join(terms) or None


def query_terms(text: str) -> list[str]:
    """Folded words of a query, for snippet highlighting."""
    terms = {fold(w) for w in _WORD.findall(text)}
    return sorted((t for t in terms if t), key=len, reverse=True)[:MAX_QUERY_TERMS]


def name_key(name: str) -> str:
    """Folded, punctuation-free form of a person's name ("~ Léo" -> "leo")."""
    return " ".join(_WORD.findall(fold(name)))


def names_column(names: Iterable[str]) -> str:
    """Stored form of a list of names, searchable with `name_param` and instr()."""
    return "|" + "|".join(f" {name_key(n)} " for n in names) + "|"


def name_param(query: str) -> str | None:
    """instr() needle matching a name, or a word that starts a name token."""
    key = name_key(query)
    return f" {key}" if key else None


def iso(dt: datetime | None) -> str | None:
    """Fixed-width, lexicographically ordered timestamp text (timezone dropped)."""
    if dt is None:
        return None
    return dt.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S.%f")


def from_iso(text: str | None) -> datetime | None:
    """Inverse of iso()."""
    return datetime.fromisoformat(text) if text else None


def rrf(rankings: Sequence[Sequence[Any]], k: int = RRF_K) -> list[tuple[Any, float]]:
    """Reciprocal rank fusion of several best-first rankings."""
    scores: dict[Any, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def snippet(text: str, terms: Sequence[str], width: int = SNIPPET_WIDTH) -> str:
    """About `width` characters of `text` around the densest cluster of query terms."""
    if len(text) <= width:
        return _SPACE.sub(" ", text).strip()
    folded: list[str] = []
    origin: list[int] = []
    for i, ch in enumerate(text):
        f = fold(ch)
        folded.append(f)
        origin.extend([i] * len(f))
    flat = "".join(folded)
    hits: list[tuple[int, str]] = []
    for term in terms:
        for m in regex.finditer(regex.escape(term), flat):
            hits.append((origin[m.start()], term))
            if len(hits) > 400:
                break
    hits.sort()
    centre = 0
    if hits:
        best = -1
        for i, (pos, _) in enumerate(hits):
            covered = {t for p, t in hits[i:] if p - pos <= width * 2 // 3}
            if len(covered) > best:
                best, centre = len(covered), pos
    start = max(0, min(centre - width // 4, len(text) - width))
    end = min(len(text), start + width)
    if start > 0:
        space = text.find(" ", start, start + 20)
        start = space + 1 if space != -1 else start
    if end < len(text):
        space = text.rfind(" ", end - 20, end)
        end = space if space > start else end
    body = _SPACE.sub(" ", text[start:end]).strip()
    return ("…" if start > 0 else "") + body + ("…" if end < len(text) else "")


def chunk_filter_sql(filters: SearchFilters | None, alias: str = "c") -> tuple[str, list[Any]]:
    """WHERE fragment (always valid, "1" when empty) and parameters for chunk filters."""
    if filters is None:
        return "1", []
    a = alias
    parts, params = _common(filters, a)
    if filters.participant:
        needle = name_param(filters.participant)
        if needle:
            parts.append(f"instr({a}.people, ?) > 0")
            params.append(needle)
    if filters.author == "self":
        parts.append(f"{a}.has_self = 1")
    elif filters.author == "other":
        parts.append(f"{a}.has_other = 1")
    if filters.after:
        parts.append(f"COALESCE({a}.end_ts, {a}.start_ts) >= ?")
        params.append(iso(filters.after))
    if filters.before:
        parts.append(f"COALESCE({a}.start_ts, {a}.end_ts) <= ?")
        params.append(iso(filters.before))
    return " AND ".join(parts) or "1", params


def item_filter_sql(filters: SearchFilters | None, alias: str = "i") -> tuple[str, list[Any]]:
    """WHERE fragment and parameters for item filters (participant matches the author)."""
    if filters is None:
        return "1", []
    a = alias
    parts, params = _common(filters, a)
    if filters.participant:
        needle = name_param(filters.participant)
        if needle:
            parts.append(f"instr({a}.author_key, ?) > 0")
            params.append(needle)
    if filters.author == "self":
        parts.append(f"{a}.is_self = 1")
    elif filters.author == "other":
        parts.append(f"{a}.is_self = 0")
    if filters.after:
        parts.append(f"{a}.ts >= ?")
        params.append(iso(filters.after))
    if filters.before:
        parts.append(f"{a}.ts <= ?")
        params.append(iso(filters.before))
    return " AND ".join(parts) or "1", params


def _common(filters: SearchFilters, a: str) -> tuple[list[str], list[Any]]:
    parts: list[str] = []
    params: list[Any] = []
    for column, values in (("source", filters.sources), ("kind", filters.kinds),
                           ("container", filters.containers)):
        if values is None:
            continue
        if not values:
            parts.append("0")
            continue
        parts.append(f"{a}.{column} IN ({','.join('?' * len(values))})")
        params.extend(str(v) for v in values)
    return parts, params
