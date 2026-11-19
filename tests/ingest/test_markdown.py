"""Markdown sources: vault, journal, essays, medical notes."""

from __future__ import annotations

from datetime import datetime

from repartee.config import SourceConfig
from repartee.ingest import ingest_source
from repartee.ingest.markdown import MarkdownAdapter

from .helpers import parse_with, write

VAULT = SourceConfig(
    type="vault",
    path="vault",
    exclude=["Templates/**"],
    kinds={"Journal/**": "journal", "Health/**": "medical", "Essays/**": "essay"},
)

NOTE = """---
title: Cycling
tags: [bike, city]
---
Intro line with a [[Nantes]] link.

# Cycling in town

I ride every day. #habit

## Gear

A heavy [[Bikes/Dutch bike|old Dutch bike]] and a #rain/jacket.

## Gear

Second gear section.
"""


def test_markdown_sections(tmp_path, owner):
    path = write(tmp_path / "Topics" / "cycling.md", NOTE)
    items, report = parse_with(MarkdownAdapter("vault"), path, tmp_path, VAULT, owner)
    assert [i.title for i in items] == ["Cycling", "Cycling in town", "Gear", "Gear"]
    assert items[0].text == "Intro line with a [[Nantes]] link."
    assert items[2].text.startswith("A heavy")
    assert all(i.container == "Topics/cycling.md" for i in items)
    assert all(i.kind == "note" and i.is_self and i.author == "Camille Martin" for i in items)
    assert [i.seq for i in items] == [0, 1, 2, 3]
    assert items[2].meta["heading_trail"] == ["Cycling in town", "Gear"]
    assert len({i.id for i in items}) == 4
    assert all(i.id.startswith("nt-") for i in items)
    assert report.unparsed_lines == 0


def test_markdown_sections_and_ids_survive_edit(tmp_path, owner):
    path = write(tmp_path / "Topics" / "cycling.md", NOTE)
    before, _ = parse_with(MarkdownAdapter("vault"), path, tmp_path, VAULT, owner)
    write(path, NOTE.replace("I ride every day.", "I ride most days, rain or not."))
    after, _ = parse_with(MarkdownAdapter("vault"), path, tmp_path, VAULT, owner)
    assert [i.id for i in before] == [i.id for i in after]
    assert before[1].text != after[1].text


def test_markdown_frontmatter_tags_links(tmp_path, owner):
    path = write(tmp_path / "Topics" / "cycling.md", NOTE)
    items, _ = parse_with(MarkdownAdapter("vault"), path, tmp_path, VAULT, owner)
    assert items[0].meta["links"] == ["Nantes"]
    assert items[2].meta["links"] == ["Bikes/Dutch bike"]
    assert set(items[1].meta["tags"]) == {"bike", "city", "habit"}
    assert set(items[2].meta["tags"]) == {"bike", "city", "rain/jacket"}
    assert items[0].meta["path"] == "Topics/cycling.md"


def test_markdown_no_headings_is_one_item(tmp_path, owner):
    path = write(tmp_path / "plain.md", "Just text.\n\nTwo paragraphs.\n")
    items, _ = parse_with(MarkdownAdapter("vault"), path, tmp_path, VAULT, owner)
    assert len(items) == 1
    assert items[0].text == "Just text.\n\nTwo paragraphs."
    assert items[0].title == "plain"


def test_markdown_headings_in_code_fences_ignored(tmp_path, owner):
    path = write(tmp_path / "code.md", "# Real\n\n```\n# not a heading\n```\n")
    items, _ = parse_with(MarkdownAdapter("vault"), path, tmp_path, VAULT, owner)
    assert len(items) == 1
    assert "# not a heading" in items[0].text


def test_markdown_kind_precedence(tmp_path, owner):
    write(tmp_path / "Journal" / "2023-05-01.md", "Rainy day.\n")
    write(tmp_path / "Journal" / "2023-05-02.md", "---\nkind: essay\n---\nActually an essay.\n")
    write(tmp_path / "Health" / "knee.md", "Knee hurts after runs.\n")
    write(tmp_path / "Other" / "x.md", "Plain.\n")
    write(tmp_path / "Other" / "bad.md", "---\nkind: nonsense\n---\nPlain too.\n")
    items, report = ingest_source(VAULT, tmp_path, owner)
    kinds = {i.container: i.kind for i in items}
    assert kinds == {
        "Journal/2023-05-01.md": "journal",
        "Journal/2023-05-02.md": "essay",
        "Health/knee.md": "medical",
        "Other/x.md": "note",
        "Other/bad.md": "note",
    }
    assert any("nonsense" in w for w in report.warnings)
    writing = SourceConfig(type="writing", path="w")
    items, _ = ingest_source(writing, tmp_path / "Other", owner)
    assert {i.kind for i in items} == {"essay"}
    assert all(i.id.startswith("wr-") for i in items)


def test_markdown_dates(tmp_path, owner):
    write(tmp_path / "Journal" / "2023-05-01.md", "Rainy day.\n")
    write(tmp_path / "a.md", "---\ndate: 2022-01-02\n---\nx\n")
    write(tmp_path / "b.md", "---\ncreated: 2022-01-03T10:30:00\n---\nx\n")
    write(tmp_path / "c.md", "x\n")
    items, _ = ingest_source(VAULT, tmp_path, owner)
    ts = {i.container: i.timestamp for i in items}
    assert ts["Journal/2023-05-01.md"] == datetime(2023, 5, 1)
    assert ts["a.md"] == datetime(2022, 1, 2)
    assert ts["b.md"] == datetime(2022, 1, 3, 10, 30)
    assert ts["c.md"] is None


def test_journal_one_item_per_file_or_dated_heading(tmp_path, owner):
    write(tmp_path / "Journal" / "2023-05-01.md", "# Morning\n\nCoffee.\n\n# Evening\n\nTired.\n")
    write(
        tmp_path / "Journal" / "journal.md",
        "Preface.\n\n## 2023-06-01\n\nFirst day.\n\n### Detail\n\nMore.\n\n"
        "## 2023-06-02\n\nSecond.\n",
    )
    items, _ = ingest_source(VAULT, tmp_path, owner)
    daily = [i for i in items if i.container == "Journal/2023-05-01.md"]
    assert len(daily) == 1 and "Coffee." in daily[0].text and "Tired." in daily[0].text
    single = [i for i in items if i.container == "Journal/journal.md"]
    assert [i.timestamp for i in single] == [None, datetime(2023, 6, 1), datetime(2023, 6, 2)]
    assert "More." in single[1].text
    assert all(i.kind == "journal" for i in single)


def test_markdown_exclude_and_skip_under(tmp_path, owner):
    vault = tmp_path / "vault"
    home = tmp_path / "home"
    write(vault / "keep.md", "kept\n")
    write(vault / "Templates" / "daily.md", "template\n")
    write(vault / ".obsidian" / "x.md", "config\n")
    write(home / "self" / "people" / "ana.md", "self-model page\n")
    (vault / "self").symlink_to(home / "self", target_is_directory=True)
    write(home / "loose.md", "inside home\n")
    (vault / "linked.md").symlink_to(home / "loose.md")
    items, _ = ingest_source(VAULT, vault, owner, skip_under=home)
    assert [i.container for i in items] == ["keep.md"]
