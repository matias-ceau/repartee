"""Obsidian vault conventions: properties, links, typed notes, entities. Invented vault."""

from __future__ import annotations

import os
from datetime import date

from repartee.config import ChunkingConfig, SourceConfig
from repartee.ingest import ingest_source
from repartee.ingest.chunking import chunk_items
from repartee.ingest.markdown import MarkdownAdapter
from repartee.ingest.vault import (
    EntityIndex,
    build_entity_index,
    link_target,
    note_properties,
    property_block,
    strip_query_blocks,
)

from .helpers import parse_with, write

VAULT = SourceConfig(type="vault", path="vault", exclude=["Templates/**"])

PERSON = """---
ntype: person
aliases: [Odile]
met: 2019-04-02
---
Friend from the choir.

## recommendations
- [ ] The salt orchard, Varga
- [x] [[Night ferries]]

## music recommendations
```dataview
TABLE artist WHERE recommended_by = [[Odile Varenne]]
```
"""

BOOK = """---
ntype: book
author: Ilse Varga
recommended_by: [[Odile Varenne]]
status: to-read
---
"""

ALBUM = """---
ntype: album
recommended_by:
  - "[[Odile Varenne|Odile]]"
  - "[[People/Tomas Brehm#2021]]"
rating: 4
---
Heard it twice on the night train.
"""

JOURNAL = """---
ntype: journal
mood: tired
---
Long rehearsal, then soup with [[Odile]].
"""


def _vault(root):
    write(root / "People" / "Odile Varenne.md", PERSON)
    write(root / "People" / "Tomas Brehm.md", "---\nntype: person\nalias: Tom\n---\nNeighbour.\n")
    write(root / "Books" / "The salt orchard.md", BOOK)
    write(root / "Music" / "Night ferries.md", ALBUM)
    write(root / "Daily" / "2024-02-03.md", JOURNAL)
    write(root / "Templates" / "person.md", "---\nntype: person\naliases: [Template]\n---\n")
    return root


def test_link_target_forms():
    assert link_target("[[Odile Varenne]]") == "Odile Varenne"
    assert link_target("[[Odile Varenne|Odile]]") == "Odile Varenne"
    assert link_target("[[Odile Varenne#Choir|x]]") == "Odile Varenne"
    assert link_target("[[People/Tomas Brehm.md]]") == "People/Tomas Brehm"
    assert link_target("Night ferries") == "Night ferries"


def test_properties_are_json_safe_and_keep_link_forms():
    props = note_properties({
        "met": date(2019, 4, 2),
        "by": [["Odile Varenne"]],  # unquoted [[Odile Varenne]] in YAML
        "with": ["[[A]]", "[[B|bee]]"],
        "n": 3,
        "nested": {"when": date(2020, 1, 1)},
    })
    assert props == {"met": "2019-04-02", "by": "[[Odile Varenne]]",
                     "with": ["[[A]]", "[[B|bee]]"], "n": 3,
                     "nested": {"when": "2020-01-01"}}
    assert property_block(props) == ("Properties: met: 2019-04-02; by: Odile Varenne; "
                                     "with: A, B; n: 3; nested: when: 2020-01-01")
    assert property_block({"title": "x", "tags": ["a"], "empty": None}) is None


def test_person_note(tmp_path, owner):
    root = _vault(tmp_path)
    path = root / "People" / "Odile Varenne.md"
    items, report = parse_with(MarkdownAdapter("vault"), path, root, VAULT, owner)
    first = items[0]
    assert first.meta["role"] == "properties"
    assert first.text == "Properties: ntype: person; aliases: Odile; met: 2019-04-02"
    assert first.title == "Odile Varenne" and first.is_self
    for item in items:
        assert item.meta["ntype"] == "person"
        assert item.meta["aliases"] == ["Odile"]
        assert item.meta["properties"]["met"] == "2019-04-02"
        assert item.kind == "note"
    recs = next(i for i in items if i.title == "recommendations")
    assert "- [ ] The salt orchard, Varga" in recs.text  # unchecked: still to read
    assert "- [x] [[Night ferries]]" in recs.text
    assert recs.meta["link_refs"] == [{"property": None, "target": "Night ferries"}]
    # The dataview block is gone; its section has nothing left, so no item.
    assert all("TABLE" not in i.text for i in items)
    assert "music recommendations" not in [i.title for i in items]
    assert report.skipped == {"query block": 1}


def test_property_links(tmp_path, owner):
    root = _vault(tmp_path)
    book, _ = parse_with(MarkdownAdapter("vault"), root / "Books" / "The salt orchard.md",
                         root, VAULT, owner)
    assert len(book) == 1  # no body: the properties alone make an item
    assert book[0].meta["properties"]["recommended_by"] == "[[Odile Varenne]]"
    assert book[0].meta["links"] == ["Odile Varenne"]
    assert book[0].meta["link_refs"] == [{"property": "recommended_by",
                                          "target": "Odile Varenne"}]
    assert "recommended_by: Odile Varenne" in book[0].text
    assert "status: to-read" in book[0].text

    album, _ = parse_with(MarkdownAdapter("vault"), root / "Music" / "Night ferries.md",
                          root, VAULT, owner)
    assert album[0].meta["properties"]["recommended_by"] == [
        "[[Odile Varenne|Odile]]", "[[People/Tomas Brehm#2021]]"]
    body = album[1]
    assert body.meta["link_refs"] == [
        {"property": "recommended_by", "target": "Odile Varenne"},
        {"property": "recommended_by", "target": "People/Tomas Brehm"},
    ]
    assert body.meta["links"] == ["Odile Varenne", "People/Tomas Brehm"]


def test_properties_do_not_change_section_ids(tmp_path, owner):
    plain = PERSON.split("---\n", 2)[2]
    a = write(tmp_path / "a" / "People" / "Odile Varenne.md", plain)
    b = write(tmp_path / "b" / "People" / "Odile Varenne.md", PERSON)
    without, _ = parse_with(MarkdownAdapter("vault"), a, tmp_path / "a", VAULT, owner)
    with_props, _ = parse_with(MarkdownAdapter("vault"), b, tmp_path / "b", VAULT, owner)
    assert [i.id for i in with_props[1:]] == [i.id for i in without]
    assert with_props[0].id not in {i.id for i in without}
    # Editing a property keeps the properties item's id too.
    write(b, PERSON.replace("aliases: [Odile]", "aliases: [Odile, Odi]"))
    again, _ = parse_with(MarkdownAdapter("vault"), b, tmp_path / "b", VAULT, owner)
    assert [i.id for i in again] == [i.id for i in with_props]
    assert again[0].text != with_props[0].text


def test_ntype_maps_kind_but_explicit_kind_wins(tmp_path, owner):
    root = _vault(tmp_path)
    write(root / "Daily" / "essayish.md", "---\nntype: journal\nkind: essay\n---\nText.\n")
    items, _ = ingest_source(VAULT, root, owner)
    kinds = {i.container: i.kind for i in items}
    assert kinds["Daily/2024-02-03.md"] == "journal"
    assert kinds["Daily/essayish.md"] == "essay"
    assert kinds["Books/The salt orchard.md"] == "note"
    assert not any(i.container.startswith("Templates/") for i in items)


def test_chunks_start_with_the_property_block(tmp_path, owner):
    from repartee.distill.extract import render_chunk  # read only: what extraction sees

    items, _ = ingest_source(VAULT, _vault(tmp_path), owner)
    chunks = {c.container: c for c in chunk_items(items, ChunkingConfig())}
    person = chunks["People/Odile Varenne.md"]
    assert person.text.startswith("Properties: ntype: person; aliases: Odile")
    assert "- [ ] The salt orchard" in person.text
    journal = chunks["Daily/2024-02-03.md"]  # per-entry kind: the line joins the entry
    assert journal.text.startswith("Properties: ntype: journal; mood: tired\n\nLong rehearsal")
    assert len(journal.item_ids) == 2
    by_id = {i.id: i for i in items}
    shown = render_chunk(person, [by_id[x] for x in person.item_ids], owner)
    assert "Properties: ntype: person; aliases: Odile" in shown


def test_properties_item_is_not_voice(tmp_path, owner):
    from repartee.voice.registers import register_of

    items, _ = ingest_source(VAULT, _vault(tmp_path), owner)
    props = [i for i in items if i.meta.get("role") == "properties"]
    assert props and all(register_of(i) is None for i in props)


def test_strip_query_blocks_keeps_other_code():
    body = "a\n```python\nx = 1\n```\n```dataviewjs\ndv.list()\n```\n~~~query\nfoo\n~~~\nb"
    text, n = strip_query_blocks(body)
    assert n == 2
    assert text == "a\n```python\nx = 1\n```\nb"


def test_entity_index(tmp_path, owner):
    root = _vault(tmp_path)
    index = build_entity_index(root, VAULT)
    odile = index.resolve("[[Odile Varenne]]")
    assert odile is not None and odile.ntype == "person"
    assert odile.path == "People/Odile Varenne.md"
    assert index.resolve("Odile") == odile
    assert index.resolve("odile varenne") == odile
    assert index.resolve("[[Odile Varenne|O.]]") == odile
    assert index.resolve("[[People/Tomas Brehm#2021]]").aliases == ("Tom",)
    assert index.resolve("Tom").title == "Tomas Brehm"
    assert index.resolve("Template") is None  # excluded folder
    assert index.resolve("Nobody") is None
    assert index.by_title()["The salt orchard"].ntype == "book"

    items, _ = ingest_source(VAULT, root, owner)
    from_items = EntityIndex.from_items(items)
    assert {e.title for e in from_items.entities} == {e.title for e in index.entities}
    assert from_items.resolve("Odile").path == odile.path


def test_entity_index_ambiguity(tmp_path):
    write(tmp_path / "a" / "Sam.md", "---\naliases: [S]\n---\n")
    write(tmp_path / "b" / "Sam.md", "x\n")
    write(tmp_path / "Sasha.md", "---\naliases: [S]\n---\n")
    index = build_entity_index(tmp_path)
    assert index.resolve("Sam") is None
    assert index.resolve("a/Sam").path == "a/Sam.md"
    assert index.resolve("S") is None
    assert set(index.ambiguous()) == {"sam", "s"}


def test_include_globs(tmp_path, owner):
    root = _vault(tmp_path)
    source = VAULT.model_copy(update={"include": ["People/**", "Books/*.md"]})
    items, report = ingest_source(source, root, owner)
    assert {i.container for i in items} == {
        "People/Odile Varenne.md", "People/Tomas Brehm.md", "Books/The salt orchard.md"}
    assert report.files == 3


def _snapshot(root):
    out = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            p = os.path.join(dirpath, name)
            st = os.stat(p)
            with open(p, "rb") as fh:
                out[p] = (fh.read(), st.st_mtime_ns)
    return out


def test_ingest_never_writes_sources(tmp_path, owner):
    root = _vault(tmp_path)
    before = _snapshot(root)
    items, _ = ingest_source(VAULT, root, owner)
    chunk_items(items, ChunkingConfig())
    build_entity_index(root, VAULT)
    assert items
    assert _snapshot(root) == before  # same files, same bytes, same mtimes


def test_ingest_code_has_no_write_paths():
    import re
    from pathlib import Path

    import repartee.ingest as ingest

    writes = re.compile(r"write_text|write_bytes|\.mkdir\(|\.touch\(|\.unlink\(|os\.remove|"
                        r"shutil\.|open\([^)]*['\"][wax+]")
    for path in Path(ingest.__file__).parent.glob("*.py"):
        hits = [line for line in path.read_text(encoding="utf-8").splitlines()
                if writes.search(line)]
        assert not hits, f"{path.name} writes: {hits}"

