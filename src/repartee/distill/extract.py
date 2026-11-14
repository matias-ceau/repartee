"""Stage 1, extract: one chunk in, first-person observations with evidence out.

The model sees the chunk with every line prefixed by its item id and cites those ids.
Code then keeps only evidence that is really in the chunk (and, for interview and
mirror chunks, only the owner's turns), and drops observations left without any.
"""

from __future__ import annotations

import regex
from pydantic import BaseModel, Field

from repartee.config import OwnerConfig
from repartee.distill.prompts import EXTRACT_SYSTEM, fill
from repartee.llm.base import LLMRequest, Message, SystemBlock
from repartee.schema import Chunk, Item, Kind, Observation, SubjectKind, stable_hash

#: How much a kind of source is worth before the model's own confidence is applied.
SOURCE_PRIOR: dict[Kind, float] = {
    Kind.INTERVIEW: 0.9,
    Kind.JOURNAL: 0.8,
    Kind.MEDICAL: 0.8,
    Kind.ESSAY: 0.75,
    Kind.NOTE: 0.7,
    Kind.TASTE: 0.7,
    Kind.DRAFT: 0.6,
    Kind.MESSAGE: 0.6,
    Kind.MIRROR: 0.5,
}
#: Applied when none of an observation's evidence was written by the owner.
OTHERS_ONLY_FACTOR = 0.7
#: Kinds where only the owner's turns may be evidence (the other speaker is the mirror).
OWNER_ONLY_KINDS = {Kind.INTERVIEW, Kind.MIRROR}

EXTRACT_MAX_TOKENS = 8000

_DATE = regex.compile(r"^\d{4}(?:-(?:0[1-9]|1[0-2])(?:-(?:0[1-9]|[12]\d|3[01]))?)?$")
_PARAGRAPHS = regex.compile(r"\n[ \t]*\n")

_KIND_LABEL: dict[Kind, str] = {
    Kind.MESSAGE: "a chat conversation",
    Kind.NOTE: "a note",
    Kind.JOURNAL: "a journal entry",
    Kind.ESSAY: "an essay or piece of writing",
    Kind.INTERVIEW: "an interview transcript (the mirror asking, me answering)",
    Kind.MEDICAL: "a health note",
    Kind.TASTE: "a list of tastes (music, books or similar)",
    Kind.MIRROR: "a conversation with the mirror",
    Kind.DRAFT: "a text I wrote and sent",
}


class ExtractedObservation(BaseModel):
    """One observation as the model returns it (flat, for structured output)."""

    subject_kind: SubjectKind
    subject: str
    text: str
    when: str | None = None
    confidence: float = 0.5
    evidence: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Structured output of an ``extract`` request."""

    observations: list[ExtractedObservation] = Field(default_factory=list)


def _item_text(item: Item, chunk_text: str) -> str:
    """The part of an item that is in this chunk (a long section is split across chunks)."""
    if item.text in chunk_text:
        return item.text
    parts = [p for p in _PARAGRAPHS.split(item.text) if p.strip() and p in chunk_text]
    return "\n\n".join(parts) if parts else item.text


def render_chunk(chunk: Chunk, items: list[Item], owner: OwnerConfig) -> str:
    """The user turn of an extract request: context, then every line with its item id."""
    owner_only = chunk.kind in OWNER_ONLY_KINDS
    people = [p for p in chunk.participants if not owner.is_self(p)]
    head = [f"Material: {_KIND_LABEL.get(chunk.kind, str(chunk.kind))}, "
            f"from the source \"{chunk.source}\", \"{chunk.container}\"."]
    if chunk.title and chunk.title != chunk.container:
        head.append(f"Title: {chunk.title}")
    if chunk.start:
        span = f"{chunk.start:%Y-%m-%d}"
        if chunk.end and f"{chunk.end:%Y-%m-%d}" != span:
            span += f" to {chunk.end:%Y-%m-%d}"
        head.append(f"Dates: {span}")
    if owner_only:
        head.append("Speakers: \"me\" is me; \"mirror\" is the mirror, whose turns are "
                    "context only and never evidence.")
    elif people:
        head.append(f"Other people in it: {', '.join(people)}")
    head.append(f"Each line starts with its id in square brackets. Lines marked \"me\" were "
                f"written by me ({owner.name}).")

    lines = []
    for item in items:
        if item.is_self:
            who = "me"
        elif owner_only:
            who = "mirror (context only)"
        else:
            who = item.author
        stamp = ""
        if item.timestamp is not None:
            stamp = (f"{item.timestamp:%Y-%m-%d %H:%M} " if chunk.kind == Kind.MESSAGE
                     else f"{item.timestamp:%Y-%m-%d} ")
        text = _item_text(item, chunk.text).strip()
        if chunk.kind == Kind.MESSAGE or owner_only:
            lines.append(f"[{item.id}] {stamp}{who}: {text}")
        else:
            title = f" — {item.title}" if item.title else ""
            lines.append(f"[{item.id}] {stamp}{who}{title}:\n{text}")
    joiner = "\n" if chunk.kind == Kind.MESSAGE else "\n\n"
    return "\n".join(head) + "\n\n<material>\n" + joiner.join(lines) + "\n</material>"


def build_request(chunk: Chunk, items: list[Item], owner: OwnerConfig) -> LLMRequest:
    """The extract request for one chunk: cacheable instructions, the chunk in the user turn."""
    return LLMRequest(
        tier="bulk",
        purpose="extract",
        system=[SystemBlock(text=fill(EXTRACT_SYSTEM, owner.name), cache=True)],
        messages=[Message(role="user", text=render_chunk(chunk, items, owner))],
        output_schema=ExtractionResult,
        max_tokens=EXTRACT_MAX_TOKENS,
    )


def _clean_id(raw: str) -> str:
    return raw.strip().strip("[]()").strip()


def valid_date(value: str | None) -> str | None:
    """The value when it is YYYY, YYYY-MM or YYYY-MM-DD, else None."""
    if not value:
        return None
    value = value.strip()
    return value if _DATE.match(value) else None


def clean_observations(
    chunk: Chunk, items: list[Item], result: ExtractionResult
) -> tuple[list[Observation], int]:
    """Validated observations of one chunk, and how many were dropped."""
    owner_only = chunk.kind in OWNER_ONLY_KINDS
    allowed = {i.id for i in items if i.is_self or not owner_only}
    own = {i.id for i in items if i.is_self}
    prior = SOURCE_PRIOR.get(chunk.kind, 0.6)
    kept: dict[str, Observation] = {}
    dropped = 0
    for raw in result.observations:
        text = " ".join(raw.text.split())
        subject = " ".join(raw.subject.split())
        evidence = list(dict.fromkeys(e for e in map(_clean_id, raw.evidence) if e in allowed))
        if not text or not subject or not evidence:
            dropped += 1
            continue
        confidence = min(max(float(raw.confidence), 0.0), 1.0) * prior
        if not own.intersection(evidence):
            confidence *= OTHERS_ONLY_FACTOR
        obs_id = "ob-" + stable_hash(chunk.id, raw.subject_kind, subject, text)
        if obs_id in kept:
            dropped += 1
            continue
        kept[obs_id] = Observation(
            id=obs_id, chunk_id=chunk.id, subject_kind=raw.subject_kind, subject=subject,
            text=text, when=valid_date(raw.when), confidence=round(confidence, 4),
            evidence=evidence,
        )
    return list(kept.values()), dropped
