"""Markdown sources: an Obsidian vault, a journal, essays, medical notes.

Item ids are keyed by location (path, heading trail, ordinal), never by text,
so editing a section keeps its id.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import regex

from repartee.config import OwnerConfig, SourceConfig
from repartee.ingest.base import (
    ParseReport,
    glob_match,
    read_text,
    relative,
    split_frontmatter,
    to_datetime,
    walk_files,
)
from repartee.schema import ID_PREFIX, Item, Kind, make_item_id

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]{0,3}(```|~~~)")
_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_TAG = regex.compile(r"(?<![\p{L}\p{N}_&#/\]\)\[`])#([\p{L}\p{N}_/-]+)")
_WIKILINK = re.compile(r"\[\[([^\[\]|#]*)(?:#[^\[\]|]*)?(?:\|[^\[\]]*)?\]\]")

DEFAULT_KIND = {"vault": Kind.NOTE, "writing": Kind.ESSAY}
SUFFIXES = {"vault": {".md"}, "writing": {".md", ".markdown", ".txt"}}


@dataclass
class Section:
    """One heading section of a markdown body (level 0 = text before any heading)."""

    level: int
    title: str | None
    trail: list[str]
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def split_sections(body: str) -> list[Section]:
    """Split a body at ATX headings, ignoring headings inside code fences."""
    sections = [Section(0, None, [])]
    stack: list[tuple[int, str]] = []
    fence: str | None = None
    for line in body.split("\n"):
        f = _FENCE.match(line)
        if f:
            fence = None if fence == f.group(1) else (fence or f.group(1))
        elif fence is None:
            h = _HEADING.match(line)
            if h:
                level, title = len(h.group(1)), h.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                sections.append(Section(level, title, [t for _, t in stack], []))
                continue
        sections[-1].lines.append(line)
    return sections


def date_in_name(name: str) -> datetime | None:
    """A YYYY-MM-DD date found in a file name or heading, if valid."""
    m = _ISO_DATE.search(name)
    if not m:
        return None
    try:
        return datetime(int(m[1]), int(m[2]), int(m[3]))
    except ValueError:
        return None


def tags_and_links(text: str) -> tuple[list[str], list[str]]:
    """Inline #tags (not purely numeric) and wikilink targets, in order, deduplicated."""
    tags = [t.strip("/-") for t in _TAG.findall(text) if not t.isdigit()]
    links = [m.strip() for m in _WIKILINK.findall(text) if m.strip()]
    return list(dict.fromkeys(t for t in tags if t)), list(dict.fromkeys(links))


def frontmatter_tags(value: Any) -> list[str]:
    """Frontmatter tags as a list, from a YAML list or a comma/space separated string."""
    if isinstance(value, str):
        value = re.split(r"[,\s]+", value)
    if not isinstance(value, list):
        return []
    return [str(t).strip().lstrip("#") for t in value if str(t).strip()]


class MarkdownAdapter:
    """Markdown notes; `name` is "vault" or "writing" (default kind note or essay)."""

    def __init__(self, name: str):
        self.name = name

    def discover(self, root: Path, source: SourceConfig) -> list[Path]:
        """Markdown files under root, minus hidden folders and excluded globs."""
        return walk_files(root, SUFFIXES[self.name], source.exclude)

    def kind_for(self, rel: str, fm: dict[str, Any], source: SourceConfig,
                 report: ParseReport) -> Kind:
        """Frontmatter `kind:` beats the config globs, which beat the source default."""
        if "kind" in fm:
            try:
                return Kind(str(fm["kind"]).strip().lower())
            except ValueError:
                report.warn(f"{rel}: unknown kind {fm['kind']!r} in frontmatter; ignored")
        for pattern, kind in source.kinds.items():
            if glob_match(rel, pattern):
                return Kind(kind)
        return DEFAULT_KIND[self.name]

    def parse(
        self,
        path: Path,
        root: Path,
        source: SourceConfig,
        owner: OwnerConfig,
        report: ParseReport,
    ) -> Iterator[Item]:
        """One item per heading section; journals one per file or per dated heading."""
        rel = relative(path, root)
        fm, body = split_frontmatter(read_text(path, report), rel, report)
        kind = self.kind_for(rel, fm, source, report)
        stamp = to_datetime(fm.get("date")) or to_datetime(fm.get("created"))
        if stamp is None and ("date" in fm or "created" in fm):
            report.warn(f"{rel}: frontmatter date not understood; using the file name")
        stamp = stamp or date_in_name(path.stem)
        file_title = str(fm.get("title") or path.stem)
        base_tags = frontmatter_tags(fm.get("tags"))
        prefix = ID_PREFIX[self.name]
        sections = split_sections(body)

        entries: list[tuple[str, str | None, list[str], int, str, datetime | None]] = []
        if kind == Kind.JOURNAL:
            entries = self._journal_entries(sections, file_title, stamp)
        else:
            for s in sections:
                entries.append((" > ".join(s.trail), s.title or file_title, s.trail, s.level,
                                s.text, stamp))

        ordinals: dict[str, int] = {}
        seq = 0
        for key, title, trail, level, text, when in entries:
            ordinal = ordinals.get(key, 0)
            ordinals[key] = ordinal + 1
            if not text:
                continue
            tags, links = tags_and_links(text)
            item = Item(
                id=make_item_id(prefix, rel, key, ordinal),
                source=self.name,
                kind=kind,
                container=rel,
                author=owner.name,
                is_self=True,
                timestamp=when,
                text=text,
                title=title,
                seq=seq,
                meta={
                    "path": rel,
                    "tags": list(dict.fromkeys(base_tags + tags)),
                    "links": links,
                    "heading_trail": trail,
                    "level": level,
                },
            )
            seq += 1
            report.items += 1
            yield item

    @staticmethod
    def _journal_entries(sections: list[Section], file_title: str, stamp: datetime | None):
        dated = [s for s in sections if s.title and date_in_name(s.title)]
        if not dated:
            text = "\n".join(_render(s) for s in sections).strip()
            return [("", file_title, [], 0, text, stamp)]
        entries = []
        current: list[Section] = []
        head: Section | None = None

        def flush():
            text = "\n".join(_render(s) for s in current).strip()
            if head is None:
                entries.append(("", file_title, [], 0, text, stamp))
            else:
                entries.append((head.title, head.title, [head.title], head.level, text,
                                date_in_name(head.title)))

        dated_ids = {id(s) for s in dated}
        for s in sections:
            if id(s) in dated_ids:
                flush()
                head, current = s, [Section(0, None, [], list(s.lines))]
            else:
                current.append(s)
        flush()
        return entries


def _render(section: Section) -> str:
    """A section back as markdown, heading line included."""
    lines = list(section.lines)
    if section.title is not None and section.level:
        lines.insert(0, f"{'#' * section.level} {section.title}")
    return "\n".join(lines)
