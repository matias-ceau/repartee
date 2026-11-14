"""Chunking items for search and extraction."""

from __future__ import annotations

from datetime import datetime, timedelta

from repartee.config import ChunkingConfig
from repartee.ingest.chunking import chunk_items
from repartee.schema import Item, Kind, make_chunk_id


def _msg(n, ts, author="Ana", is_self=False, text=None, container="Ana", lang=None):
    return Item(
        id=f"wa-{n:04d}",
        source="whatsapp",
        kind=Kind.MESSAGE,
        container=container,
        author=author,
        is_self=is_self,
        timestamp=ts,
        text=text or f"message {n}",
        seq=n,
        lang=lang,
    )


def _section(n, text, title=None, container="note.md", kind=Kind.NOTE, level=2):
    return Item(
        id=f"nt-{container}-{n}",
        source="vault",
        kind=kind,
        container=container,
        author="Camille Martin",
        is_self=True,
        text=text,
        title=title,
        seq=n,
        meta={"level": level if title else 0},
    )


def test_chat_chunks_split_on_gap_and_size():
    t0 = datetime(2021, 3, 12, 14, 5)
    items = [_msg(0, t0), _msg(1, t0 + timedelta(minutes=5)), _msg(2, t0 + timedelta(hours=4))]
    chunks = chunk_items(items, ChunkingConfig())
    assert [c.item_ids for c in chunks] == [["wa-0000", "wa-0001"], ["wa-0002"]]
    assert chunks[0].text.splitlines() == ["[2021-03-12 14:05] Ana: message 0",
                                           "[2021-03-12 14:10] Ana: message 1"]
    assert chunks[0].start == t0 and chunks[0].end == t0 + timedelta(minutes=5)
    assert chunks[0].id == make_chunk_id(["wa-0000", "wa-0001"])
    assert chunks[0].kind == Kind.MESSAGE and chunks[0].container == "Ana"

    long = [_msg(n, t0 + timedelta(minutes=n), text="word " * 100) for n in range(20)]
    chunks = chunk_items(long, ChunkingConfig(chat_max_tokens=500))
    assert len(chunks) > 1
    assert sum(len(c.item_ids) for c in chunks) == 20
    assert all(len(c.text) / 3.5 <= 500 for c in chunks)


def test_chunk_self_text_and_participants():
    t0 = datetime(2021, 3, 12, 14, 5)
    items = [
        _msg(0, t0, "Ana", lang="fr"),
        _msg(1, t0, "Cam", True, "moi", lang="fr"),
        _msg(2, t0, "Léo", lang="en"),
        _msg(3, t0, "Ana", lang="fr"),
        _msg(4, t0, "Camille", True, "encore moi"),
    ]
    (chunk,) = chunk_items(items, ChunkingConfig())
    assert chunk.participants == ["Ana", "Cam", "Léo", "Camille"]
    assert chunk.self_text == "moi\nencore moi"
    assert chunk.lang == "fr"


def test_note_chunks_merge_small_sections():
    items = [
        _section(0, "Intro.", None),
        _section(1, "Small one.", "A"),
        _section(2, "Small two.", "B"),
    ]
    (chunk,) = chunk_items(items, ChunkingConfig())
    assert chunk.item_ids == [i.id for i in items]
    assert chunk.text == "Intro.\n\n## A\n\nSmall one.\n\n## B\n\nSmall two."
    assert chunk.self_text == "Intro.\nSmall one.\nSmall two."


def test_note_long_section_split_on_paragraphs():
    para = "sentence " * 200  # ~ 500 tokens
    items = [_section(0, "\n\n".join([para] * 4), "Long"), _section(1, "after", "Next")]
    chunks = chunk_items(items, ChunkingConfig(note_target_tokens=800))
    assert len(chunks) >= 3
    assert chunks[0].item_ids == [items[0].id]
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks[-1].item_ids[-1] == items[1].id


def test_journal_one_chunk_per_entry_and_files_kept_apart():
    items = [
        _section(0, "Day one.", None, container="J/2023-01-01.md", kind=Kind.JOURNAL),
        _section(0, "Day two.", None, container="J/2023-01-02.md", kind=Kind.JOURNAL),
        _section(0, "Other note.", None, container="other.md"),
    ]
    chunks = chunk_items(items, ChunkingConfig())
    assert [c.item_ids for c in chunks] == [[items[0].id], [items[1].id], [items[2].id]]


def test_transcript_one_chunk_per_container():
    turns = [
        Item(id=f"iv-{n}", source="interviews", kind=Kind.INTERVIEW, container="t.md",
             author="mirror" if n % 2 == 0 else "Camille Martin", is_self=n % 2 == 1,
             text=f"turn {n}", seq=n)
        for n in range(4)
    ]
    (chunk,) = chunk_items(turns, ChunkingConfig())
    assert chunk.participants == ["mirror", "Camille Martin"]
    assert chunk.self_text == "turn 1\nturn 3"
    assert "mirror: turn 0" in chunk.text
