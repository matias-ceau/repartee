"""The home's own files: transcripts and drafts."""

from __future__ import annotations

from datetime import datetime

from repartee.ingest import parse_draft, parse_transcript

from .helpers import write

TRANSCRIPT = """---
kind: interview
date: 2026-11-14T10:32:00
mode: interview
title: Childhood
---
**mirror:** Where did you grow up?

**me:** In Nantes, until I was eleven.
Then we moved.

**mirror:** Where to?

**me:** Lyon.
"""

DRAFT = """---
kind: draft
date: 2026-11-14T18:02:11
register_name: whatsapp.fr
recipient: Ana Ruiz
intent: say I can't come Friday
accepted: true
---
## Mirror

dsl je peux pas vendredi

## Final

dsl je peux pas vendredi 😕

on se voit samedi ?
"""


def test_transcript_turns(tmp_path, owner):
    path = write(tmp_path / "interviews" / "2026-11-14-childhood.md", TRANSCRIPT)
    items = parse_transcript(path, owner)
    assert [i.is_self for i in items] == [False, True, False, True]
    assert [i.author for i in items] == ["mirror", "Camille Martin", "mirror", "Camille Martin"]
    assert items[1].text == "In Nantes, until I was eleven.\nThen we moved."
    assert items[0].text == "Where did you grow up?"
    assert all(i.kind == "interview" and i.source == "interviews" for i in items)
    assert all(i.id.startswith("iv-") for i in items)
    assert items[0].timestamp == datetime(2026, 11, 14, 10, 32)
    assert items[0].title == "Childhood"
    assert items[0].meta["role"] == "mirror" and items[1].meta["role"] == "me"
    assert items[1].meta["mode"] == "interview"
    assert [i.seq for i in items] == [0, 1, 2, 3]
    assert items[0].container == "2026-11-14-childhood.md"


def test_transcript_ids_stable_when_appended(tmp_path, owner):
    path = write(tmp_path / "s.md", TRANSCRIPT)
    first = parse_transcript(path, owner)
    write(path, TRANSCRIPT + "\n**mirror:** And then?\n")
    second = parse_transcript(path, owner)
    assert [i.id for i in second[:4]] == [i.id for i in first]


def test_transcript_mirror_kind(tmp_path, owner):
    path = write(
        tmp_path / "conversations" / "c.md",
        TRANSCRIPT.replace("kind: interview", "kind: mirror").replace(
            "mode: interview", "mode: chat"
        ),
    )
    items = parse_transcript(path, owner)
    assert {i.kind for i in items} == {"mirror"}
    assert {i.source for i in items} == {"conversations"}
    assert all(i.id.startswith("mr-") for i in items)


def test_draft_only_final_is_ingested(tmp_path, owner):
    path = write(tmp_path / "drafts" / "20261114-180211-ab12.md", DRAFT)
    items = parse_draft(path, owner)
    assert [i.text for i in items] == ["dsl je peux pas vendredi 😕", "on se voit samedi ?"]
    assert all(i.kind == "draft" and i.is_self and i.source == "drafts" for i in items)
    assert items[0].meta["register_name"] == "whatsapp.fr"
    assert items[0].meta["recipient"] == "Ana Ruiz"
    assert items[0].timestamp == datetime(2026, 11, 14, 18, 2, 11)
    assert all(i.id.startswith("dr-") for i in items)
    assert len({i.id for i in items}) == 2


def test_draft_not_accepted_is_skipped(tmp_path, owner):
    path = write(tmp_path / "d.md", DRAFT.replace("accepted: true", "accepted: false"))
    assert parse_draft(path, owner) == []
