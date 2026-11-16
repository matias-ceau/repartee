"""Group items into chunks: the unit for search and for extraction."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta

from repartee.config import ChunkingConfig, estimate_tokens
from repartee.ingest.assistant import strip_code_blocks
from repartee.schema import Chunk, Item, Kind, make_chunk_id

#: Kinds whose sections of one file are merged up to the note target.
SECTIONED = {Kind.NOTE, Kind.ESSAY}
#: Kinds chunked one item (entry) at a time.
PER_ITEM = {Kind.JOURNAL, Kind.TASTE}
#: Kinds chunked one container (file) at a time.
PER_CONTAINER = {Kind.INTERVIEW, Kind.MIRROR, Kind.DRAFT, Kind.MEDICAL, Kind.ASSISTANT}
#: Kinds where each item is one speaker's turn, rendered "Author: text".
TURNS = {Kind.INTERVIEW, Kind.MIRROR, Kind.ASSISTANT}
#: Characters of an AI assistant's turn kept in a chunk: it is context for the owner's
#: turns, and its long replies would otherwise fill the chunk.
ASSISTANT_CONTEXT_CHARS = 600


@dataclass
class _Piece:
    item: Item
    text: str  # rendered for reading
    own: str  # the item's own words in this piece (for self_text)


def _tokens(text: str) -> int:
    return estimate_tokens(text) if text else 0


def render_message(item: Item) -> str:
    """`[YYYY-MM-DD HH:MM] Author: text`, without the time when there is none."""
    if item.timestamp is None:
        return f"{item.author}: {item.text}"
    return f"[{item.timestamp:%Y-%m-%d %H:%M}] {item.author}: {item.text}"


def context_text(item: Item) -> str:
    """What a chunk shows of an item: all of it, except an AI assistant's turn, clipped.

    In an assistant's turn, fenced code becomes "[code]" and the rest is cut after
    ASSISTANT_CONTEXT_CHARS characters, at a word boundary, ending with "[…]".
    """
    if item.kind != Kind.ASSISTANT or item.is_self:
        return item.text
    text = strip_code_blocks(item.text, "[code]")
    if len(text) <= ASSISTANT_CONTEXT_CHARS:
        return text
    cut = text[:ASSISTANT_CONTEXT_CHARS]
    space = max(cut.rfind(" "), cut.rfind("\n"))
    if space > ASSISTANT_CONTEXT_CHARS // 2:
        cut = cut[:space]
    return cut.rstrip() + " […]"


def _heading(item: Item) -> str | None:
    level = int(item.meta.get("level") or 0)
    if item.title and level > 0:
        return f"{'#' * min(level, 6)} {item.title}"
    return None


def _split_paragraphs(text: str, limit: int) -> list[str]:
    """Group paragraphs into parts of at most `limit` tokens (a lone long paragraph stays whole)."""
    parts: list[str] = []
    current: list[str] = []
    for para in [p for p in re.split(r"\n[ \t]*\n", text) if p.strip()]:
        candidate = "\n\n".join([*current, para])
        if current and _tokens(candidate) > limit:
            parts.append("\n\n".join(current))
            current = [para]
        else:
            current.append(para)
    if current:
        parts.append("\n\n".join(current))
    return parts or [text]


def _section_pieces(
    item: Item, limit: int, header: str | None, sep: str = "\n\n"
) -> list[_Piece]:
    full = sep.join(x for x in (header, item.text) if x)
    if _tokens(full) <= limit:
        return [_Piece(item, full, item.text)]
    parts = _split_paragraphs(item.text, limit)
    pieces = []
    for n, part in enumerate(parts):
        text = sep.join(x for x in (header, part) if x) if n == 0 else part
        pieces.append(_Piece(item, text, part))
    return pieces


def _merge(pieces: list[_Piece], limit: int, joiner: str) -> list[list[_Piece]]:
    groups: list[list[_Piece]] = []
    chars = 0
    for piece in pieces:
        if groups and groups[-1]:
            new_chars = chars + len(joiner) + len(piece.text)
            if estimate_tokens("x" * new_chars) <= limit:
                groups[-1].append(piece)
                chars = new_chars
                continue
        groups.append([piece])
        chars = len(piece.text)
    return groups


class _Builder:
    def __init__(self) -> None:
        self.used: set[str] = set()

    def build(self, pieces: list[_Piece], joiner: str, title: str | None = None) -> Chunk:
        items = list({id(p.item): p.item for p in pieces}.values())
        item_ids = [i.id for i in items]
        chunk_id = make_chunk_id(item_ids)
        part = 0
        while chunk_id in self.used:  # a section split in several parts
            part += 1
            chunk_id = make_chunk_id([*item_ids, f"part:{part}"])
        self.used.add(chunk_id)
        stamps = [i.timestamp for i in items if i.timestamp is not None]
        langs = Counter(i.lang for i in items if i.lang)
        first = items[0]
        return Chunk(
            id=chunk_id,
            source=first.source,
            kind=first.kind,
            container=first.container,
            item_ids=item_ids,
            start=min(stamps) if stamps else None,
            end=max(stamps) if stamps else None,
            participants=list(dict.fromkeys(i.author for i in items)),
            text=joiner.join(p.text for p in pieces),
            self_text="\n".join(p.own for p in pieces if p.item.is_self and p.own),
            title=title if title is not None else (first.title or first.container),
            lang=langs.most_common(1)[0][0] if langs else None,
        )


def _chat_chunks(items: list[Item], config: ChunkingConfig, builder: _Builder) -> list[Chunk]:
    gap = timedelta(hours=config.chat_gap_hours)
    chunks: list[Chunk] = []
    current: list[_Piece] = []
    chars = 0
    last = None
    for item in items:
        line = render_message(item)
        new_chars = chars + len(line) + (1 if current else 0)
        silent = item.timestamp is not None and last is not None and item.timestamp - last >= gap
        if current and (silent or estimate_tokens("x" * new_chars) > config.chat_max_tokens):
            chunks.append(builder.build(current, "\n", title=item.container))
            current, new_chars = [], len(line)
        current.append(_Piece(item, line, item.text))
        chars = new_chars
        last = item.timestamp or last
    if current:
        chunks.append(builder.build(current, "\n", title=current[0].item.container))
    return chunks


def chunk_items(items: Iterable[Item], config: ChunkingConfig) -> list[Chunk]:
    """Chunk items per container: chats by silence and size, notes by merged sections."""
    groups: dict[tuple[str, str], list[Item]] = {}
    for item in items:
        groups.setdefault((item.source, item.container), []).append(item)
    target = config.note_target_tokens
    chunks: list[Chunk] = []
    for group in groups.values():
        group.sort(key=lambda i: i.seq)
        builder = _Builder()
        # A container normally holds one kind; split runs defensively.
        runs: list[list[Item]] = []
        for item in group:
            if runs and runs[-1][0].kind == item.kind:
                runs[-1].append(item)
            else:
                runs.append([item])
        for run in runs:
            kind = run[0].kind
            if kind == Kind.MESSAGE:
                chunks.extend(_chat_chunks(run, config, builder))
            elif kind in SECTIONED:
                pieces = [p for i in run for p in _section_pieces(i, target, _heading(i))]
                chunks.extend(builder.build(g, "\n\n") for g in _merge(pieces, target, "\n\n"))
            elif kind in PER_ITEM:
                for i in run:
                    for p in _section_pieces(i, 2 * target, None):
                        chunks.append(builder.build([p], "\n\n"))
            else:  # PER_CONTAINER and anything new
                pieces = []
                for i in run:
                    if kind == Kind.ASSISTANT and not i.is_self:
                        pieces.append(_Piece(i, f"{i.author}: {context_text(i)}", ""))
                    elif kind in TURNS:
                        pieces.extend(_section_pieces(i, 2 * target, f"{i.author}:", " "))
                    elif kind == Kind.MEDICAL:
                        pieces.extend(_section_pieces(i, 2 * target, _heading(i)))
                    else:
                        pieces.extend(_section_pieces(i, 2 * target, None))
                chunks.extend(
                    builder.build(g, "\n\n") for g in _merge(pieces, 2 * target, "\n\n")
                )
    return chunks
