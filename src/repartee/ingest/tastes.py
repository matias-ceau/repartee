"""Tastes: CSV and JSON library or listening exports, and markdown lists."""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from repartee.config import OwnerConfig, SourceConfig
from repartee.ingest.base import (
    ParseReport,
    read_text,
    relative,
    split_frontmatter,
    walk_files,
)
from repartee.ingest.markdown import split_sections
from repartee.schema import ID_PREFIX, Item, Kind, make_item_id

#: Most lines of normalized text per taste item.
MAX_LINES = 50

#: Column name (lowercased, `_` as space) -> canonical field.
FIELD_SYNONYMS: dict[str, str] = {
    **dict.fromkeys(
        ["artist", "artist name", "artistname", "artists", "band", "creator", "performer",
         "master metadata album artist name"], "artist"),
    **dict.fromkeys(
        ["track", "track name", "trackname", "song", "song name", "master metadata track name"],
        "track"),
    **dict.fromkeys(["album", "album name", "record", "master metadata album album name"],
                    "album"),
    **dict.fromkeys(["title", "name", "book", "book title", "film", "movie"], "title"),
    **dict.fromkeys(["author", "authors", "writer", "director"], "author"),
    **dict.fromkeys(["plays", "play count", "playcount", "count", "listens", "scrobbles"],
                    "plays"),
    **dict.fromkeys(["year", "release year", "year published", "original publication year"],
                    "year"),
    **dict.fromkeys(["played at", "playedat", "endtime", "end time", "ts", "timestamp", "date",
                     "listened at", "utc time"], "played_at"),
    **dict.fromkeys(["rating", "my rating", "stars", "score"], "rating"),
}

_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?")


def _canonical(record: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in record.items():
        if value is None or isinstance(value, (dict, list)):
            continue
        name = " ".join(str(key).replace("_", " ").replace("-", " ").lower().split())
        text = str(value).strip()
        if text:
            out[FIELD_SYNONYMS.get(name, "other:" + str(key))] = text
    return out


def record_lines(records: list[dict[str, Any]]) -> list[str]:
    """Normalized lines from records: listening events aggregated per artist and year."""
    rows = [_canonical(r) for r in records]
    rows = [r for r in rows if r]
    events = [r for r in rows if "artist" in r and "played_at" in r and "plays" not in r]
    if rows and len(events) == len(rows):
        counts: Counter[tuple[str, str]] = Counter()
        for r in events:
            m = re.search(r"\d{4}", r["played_at"])
            counts[(m.group(0) if m else "unknown year", r["artist"])] += 1
        ordered = sorted(counts.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1]))
        return [f"{year} · {artist}: {n} plays" for (year, artist), n in ordered]
    lines = []
    for r in rows:
        name = r.get("track") or r.get("title")
        if r.get("artist") and name:
            head = f"{r['artist']} — {name}"
        elif name and r.get("author"):
            head = f"{name} — {r['author']}"
        else:
            head = r.get("artist") or name or r.get("author") or ""
        extras = [r[k] for k in ("album", "year") if k in r]
        if "plays" in r:
            extras.append(f"{r['plays']} plays")
        if "rating" in r:
            extras.append(f"rating {r['rating']}")
        if not head:
            extras = [f"{k.removeprefix('other:')}: {v}" for k, v in r.items()]
        lines.append(" · ".join([head, *extras]) if head else " · ".join(extras))
    return lines


class TastesAdapter:
    """Music, books and similar, normalized to `taste` items of at most 50 lines."""

    name = "tastes"

    def discover(self, root: Path, source: SourceConfig) -> list[Path]:
        """CSV, JSON, markdown and text files under root."""
        return walk_files(root, {".csv", ".json", ".md", ".txt"}, source.exclude, source.include)

    def parse(
        self,
        path: Path,
        root: Path,
        source: SourceConfig,
        owner: OwnerConfig,
        report: ParseReport,
    ) -> Iterator[Item]:
        """Normalize one file into taste items."""
        rel = relative(path, root)
        text = read_text(path, report)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            groups = [("", path.stem, self._csv_lines(text))]
        elif suffix == ".json":
            groups = [("", path.stem, self._json_lines(text, rel, report))]
        else:
            groups = self._markdown_groups(text, path.stem, rel, report)
        for key, title, lines in groups:
            for part, start in enumerate(range(0, len(lines), MAX_LINES)):
                block = lines[start : start + MAX_LINES]
                report.items += 1
                yield Item(
                    id=make_item_id(ID_PREFIX["tastes"], rel, key, part),
                    source="tastes",
                    kind=Kind.TASTE,
                    container=rel,
                    author=owner.name,
                    is_self=True,
                    text="\n".join(block),
                    title=title,
                    seq=part,
                    meta={"path": rel, "format": suffix.lstrip("."), "category": path.stem},
                )

    @staticmethod
    def _csv_lines(text: str) -> list[str]:
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        return record_lines(list(csv.DictReader(io.StringIO(text), dialect=dialect)))

    @staticmethod
    def _json_lines(text: str, rel: str, report: ParseReport) -> list[str]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            report.warn(f"{rel}: not valid JSON ({exc.msg}); skipped")
            return []
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [])
        if not isinstance(data, list):
            report.warn(f"{rel}: JSON holds no list of records; skipped")
            return []
        lines = [str(x).strip() for x in data if isinstance(x, (str, int, float))]
        return [line for line in lines if line] + record_lines(
            [x for x in data if isinstance(x, dict)]
        )

    @staticmethod
    def _markdown_groups(text: str, stem: str, rel: str, report: ParseReport):
        _, body = split_frontmatter(text, rel, report)
        groups = []
        for s in split_sections(body):
            lines = [_LIST_MARKER.sub("", line).strip() for line in s.lines]
            lines = [line for line in lines if line]
            if lines:
                groups.append((" > ".join(s.trail), s.title or stem, lines))
        return groups
