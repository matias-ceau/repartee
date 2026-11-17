"""Obsidian conventions: properties, wikilinks, typed notes and the entities they name.

A vault note's frontmatter holds facts the owner wrote by hand (`ntype: person`,
`aliases: [Odile]`, `recommended_by: "[[Odile Varenne]]"`). This module keeps them
JSON-safe on the items, pulls the wikilinks out of them, renders them as one
compact line the extractor can read, and indexes the notes as entities so that
`[[Odile Varenne]]` and "Odile" resolve to the same one. Read only, no model.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from repartee.config import SourceConfig
from repartee.ingest.base import ParseReport, read_text, relative, split_frontmatter, walk_files
from repartee.schema import Item

#: `[[target]]`, `[[target|shown]]`, `[[target#heading]]`, `![[embed]]`; group 1 is the target.
WIKILINK = re.compile(r"\[\[([^\[\]|#]*)(?:#[^\[\]|]*)?(?:\|[^\[\]]*)?\]\]")
#: Properties left out of the property block: already on the item, or presentational.
BLOCK_SKIP = {"kind", "title", "date", "created", "tags", "tag", "cssclasses", "cssclass",
              "publish", "permalink"}
#: Fenced blocks that are queries over the vault (Dataview, embedded search), not prose.
QUERY_LANGUAGES = {"dataview", "dataviewjs", "query"}

_FENCE = re.compile(r"^[ \t]{0,3}(```|~~~)[ \t]*([\w-]*)")


def link_target(raw: str) -> str:
    """A link target as a note title or path: no alias, heading, brackets or `.md`."""
    m = WIKILINK.search(raw)
    target = m.group(1) if m else raw.split("|", 1)[0].split("#", 1)[0]
    target = target.strip()
    if target.lower().endswith(".md"):
        target = target[:-3].rstrip()
    return target


def body_links(text: str) -> list[str]:
    """Wikilink targets in a text, in order, deduplicated."""
    return list(dict.fromkeys(t for t in (link_target(m.group(0))
                                         for m in WIKILINK.finditer(text)) if t))


def json_safe(value: Any) -> Any:
    """A frontmatter value as JSON-safe data: dates as ISO strings, unknown types as text.

    An unquoted `[[X]]` in YAML parses as a list holding a list holding "X"; the inner
    one-string list becomes the wikilink "[[X]]" again.
    """
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, list | tuple | set):
        return [_link_or_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _link_or_value(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return f"[[{value[0]}]]"
    return json_safe(value)


def note_properties(fm: dict[str, Any]) -> dict[str, Any]:
    """Every frontmatter property, JSON-safe; a lone unquoted `[[X]]` stays one value."""
    props = {}
    for key, value in fm.items():
        safe = json_safe(value)
        if (isinstance(value, list) and len(value) == 1 and isinstance(value[0], list)
                and isinstance(safe, list) and isinstance(safe[0], str)):
            safe = safe[0]
        props[str(key)] = safe
    return props


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)


def property_links(props: dict[str, Any]) -> list[dict[str, str | None]]:
    """`{"property", "target"}` for each wikilink in a property value, in order."""
    refs = []
    for key, value in props.items():
        for text in _strings(value):
            for target in body_links(text):
                refs.append({"property": key, "target": target})
    return refs


def note_aliases(props: dict[str, Any]) -> list[str]:
    """`aliases` (or `alias`) as a list of names, from a list or a single value."""
    value = props.get("aliases", props.get("alias"))
    names = [link_target(s) if "[[" in s else s.strip() for s in _strings(value)]
    return list(dict.fromkeys(n for n in names if n))


def note_ntype(props: dict[str, Any]) -> str | None:
    """The note type (`ntype`), lower-cased; the first one when it is a list."""
    names = [s.strip() for s in _strings(props.get("ntype")) if s.strip()]
    return link_target(names[0]).lower() if names else None


def _plain(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        text = WIKILINK.sub(lambda m: link_target(m.group(0)), value)
        return " ".join(text.split())
    if isinstance(value, list):
        return ", ".join(p for p in (_plain(v) for v in value) if p)
    if isinstance(value, dict):
        return ", ".join(f"{k}: {p}" for k, v in value.items() if (p := _plain(v)))
    return "" if value is None else str(value)


def property_block(props: dict[str, Any]) -> str | None:
    """`Properties: ntype: person; aliases: Odile; recommended_by: Odile Varenne`, or None.

    Wikilinks are shown as their target; empty values and BLOCK_SKIP keys are left out.
    """
    parts = [f"{key}: {text}" for key, value in props.items()
             if key.lower() not in BLOCK_SKIP and (text := _plain(value))]
    return "Properties: " + "; ".join(parts) if parts else None


def strip_query_blocks(body: str) -> tuple[str, int]:
    """The body without ```dataview / dataviewjs / query blocks, and how many were dropped."""
    out: list[str] = []
    dropped = 0
    fence: str | None = None
    query = False
    for line in body.split("\n"):
        m = _FENCE.match(line)
        if fence is None:
            if m:
                fence, query = m.group(1), m.group(2).lower() in QUERY_LANGUAGES
                if query:
                    dropped += 1
                    continue
            out.append(line)
            continue
        if not query:
            out.append(line)
        if m and m.group(1) == fence and not m.group(2):  # the closing fence
            fence, query = None, False
    return "\n".join(out), dropped


# ------------------------------------------------------------------ entity index


@dataclass(frozen=True)
class NoteEntity:
    """One vault note seen as an entity: its title (file stem), aliases, type and path."""

    title: str
    path: str  # relative to the vault root, posix
    ntype: str | None = None
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.title, *self.aliases)


@dataclass
class EntityIndex:
    """Notes by title, path and alias, for resolving `[[links]]` and plain names."""

    entities: list[NoteEntity] = field(default_factory=list)
    _titles: dict[str, list[NoteEntity]] = field(default_factory=dict, repr=False)
    _aliases: dict[str, list[NoteEntity]] = field(default_factory=dict, repr=False)
    _paths: dict[str, NoteEntity] = field(default_factory=dict, repr=False)

    def add(self, entity: NoteEntity) -> None:
        self.entities.append(entity)
        self._titles.setdefault(entity.title.casefold(), []).append(entity)
        for alias in entity.aliases:
            self._aliases.setdefault(alias.casefold(), []).append(entity)
        self._paths[_strip_md(entity.path).casefold()] = entity

    def candidates(self, name: str) -> list[NoteEntity]:
        """Every note a name or link could mean: a path, else titles, else aliases."""
        target = link_target(name)
        if not target:
            return []
        key = target.casefold()
        if "/" in key:
            if key in self._paths:
                return [self._paths[key]]
            key = key.rsplit("/", 1)[1]
        return list(self._titles.get(key) or self._aliases.get(key) or [])

    def resolve(self, name: str) -> NoteEntity | None:
        """The one note a name or `[[link]]` means; None when unknown or ambiguous."""
        found = self.candidates(name)
        return found[0] if len(found) == 1 else None

    def by_title(self) -> dict[str, NoteEntity]:
        """Title -> entity (the first note when two share a title)."""
        out: dict[str, NoteEntity] = {}
        for e in self.entities:
            out.setdefault(e.title, e)
        return out

    def ambiguous(self) -> dict[str, list[str]]:
        """Names (lower-cased) shared by several notes -> their paths."""
        out: dict[str, list[str]] = {}
        for table in (self._titles, self._aliases):
            for name, ents in table.items():
                paths = list(dict.fromkeys(e.path for e in ents))
                if len(paths) > 1:
                    out[name] = paths
        return out

    @classmethod
    def from_items(cls, items: Iterable[Item]) -> EntityIndex:
        """Index from ingested vault items (one entity per note; meta carries the facts)."""
        index = cls()
        seen: set[str] = set()
        for item in items:
            path = str(item.meta.get("path") or item.container)
            if item.source != "vault" or path in seen:
                continue
            seen.add(path)
            index.add(NoteEntity(_strip_md(Path(path).name), path, item.meta.get("ntype"),
                                 tuple(item.meta.get("aliases") or ())))
        return index


def _strip_md(path: str) -> str:
    return path[:-3] if path.lower().endswith(".md") else path


def build_entity_index(root: Path, source: SourceConfig | None = None,
                       report: ParseReport | None = None) -> EntityIndex:
    """Index every note of a vault source from its frontmatter. Reads files, never writes."""
    exclude = source.exclude if source else []
    include = source.include if source else []
    index = EntityIndex()
    for path in walk_files(root, {".md"}, exclude, include):
        rel = relative(path, root)
        fm, _ = split_frontmatter(read_text(path, report), rel, report)
        props = note_properties(fm)
        index.add(NoteEntity(path.stem, rel, note_ntype(props), tuple(note_aliases(props))))
    return index
