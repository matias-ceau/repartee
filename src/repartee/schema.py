"""Shared data types. Every module speaks these; none redefines them.

See docs/superpowers/specs/2026-11-14-mirror-design.md section 7.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- ids


def stable_hash(*parts: object, length: int = 12) -> str:
    """Short hex digest of the parts, joined so that boundaries cannot collide."""
    joined = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]


def make_item_id(prefix: str, *key_parts: object) -> str:
    """Stable item id from a natural key, e.g. make_item_id("wa", chat, ts, author, text, n)."""
    return f"{prefix}-{stable_hash(*key_parts)}"


def make_chunk_id(item_ids: list[str]) -> str:
    return f"ch-{stable_hash(*item_ids)}"


def is_holdout(item_id: str, percent: int = 5) -> bool:
    """Deterministic holdout split: true for about `percent`% of ids."""
    return int(hashlib.sha1(item_id.encode("utf-8")).hexdigest()[:8], 16) % 100 < percent


def slugify(label: str) -> str:
    """File-name-safe, lowercase, Unicode-preserving slug ("Ana Ruiz" -> "ana-ruiz")."""
    import regex

    text = regex.sub(r"[^\p{L}\p{N}]+", "-", label.strip().lower())
    return text.strip("-") or "untitled"


# ------------------------------------------------------------------------ corpus


class Kind(StrEnum):
    """What an item is, independent of where it came from."""

    MESSAGE = "message"  # chat message
    NOTE = "note"  # vault note section
    JOURNAL = "journal"
    ESSAY = "essay"  # long-form writing meant to be read
    INTERVIEW = "interview"  # one turn of an interview transcript
    MEDICAL = "medical"
    TASTE = "taste"  # music, books, ...
    MIRROR = "mirror"  # one turn of a session with the mirror
    DRAFT = "draft"  # final text of an accepted draft


#: Short prefix per source for item ids.
ID_PREFIX: dict[str, str] = {
    "whatsapp": "wa",
    "vault": "nt",
    "writing": "wr",
    "tastes": "ts",
    "interviews": "iv",
    "conversations": "mr",
    "drafts": "dr",
}


class Item(BaseModel):
    """Smallest unit from a source: one message, note section, journal entry, turn."""

    id: str
    source: str  # adapter name, a key of ID_PREFIX
    kind: Kind
    container: str  # chat name, note path relative to its root, document id
    author: str  # display name as found in the source
    is_self: bool  # written by the owner
    timestamp: datetime | None = None  # naive local time as written in the source
    text: str
    title: str | None = None
    lang: str | None = None  # ISO 639-1 when detected
    seq: int = 0  # order within the container
    #: Reserved for the blind voice test: never used for style statistics, style
    #: cards, samples or reply pairs. Memory search is not affected. See is_holdout().
    holdout: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Unit for search and extraction: a chat session, a note section, an entry."""

    id: str
    source: str
    kind: Kind
    container: str
    item_ids: list[str]
    start: datetime | None = None
    end: datetime | None = None
    participants: list[str] = Field(default_factory=list)  # author names, owner included
    text: str  # rendered for reading, e.g. "[2021-03-12 14:05] Ana: ..."
    self_text: str = ""  # only what the owner wrote, newline-joined
    title: str | None = None
    lang: str | None = None


class ReplyPair(BaseModel):
    """What someone said and what the owner answered, from a chat."""

    id: str
    container: str
    recipient: str  # the other party (or the chat name for groups)
    timestamp: datetime | None = None
    context: str  # the incoming message(s), rendered with author names
    reply: list[str]  # the owner's burst, one entry per message
    context_item_ids: list[str] = Field(default_factory=list)
    reply_item_ids: list[str] = Field(default_factory=list)
    lang: str | None = None


class SearchFilters(BaseModel):
    sources: list[str] | None = None
    kinds: list[Kind] | None = None
    containers: list[str] | None = None
    participant: str | None = None  # a person's name (matched case-insensitively)
    author: Literal["self", "other", "any"] = "any"
    after: datetime | None = None
    before: datetime | None = None


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    snippet: str


# -------------------------------------------------------------------- self-model


class PageKind(StrEnum):
    CORE = "core"
    PERSON = "person"
    TOPIC = "topic"
    TASTE = "taste"
    BEHAVIOUR = "behaviour"
    TIMELINE = "timeline"
    HEALTH = "health"
    VOICE = "voice"
    META = "meta"  # open-questions.md and the like


#: Directory under self/ for each kind that has one page per subject.
PAGE_DIRS: dict[PageKind, str] = {
    PageKind.PERSON: "people",
    PageKind.TOPIC: "topics",
    PageKind.TASTE: "tastes",
    PageKind.BEHAVIOUR: "behaviour",
    PageKind.TIMELINE: "timeline",
    PageKind.HEALTH: "health",
    PageKind.VOICE: "voice",
}

#: Sections the pipeline may write claims into, per page kind. "Earlier" holds
#: superseded claims. Anything else on a page belongs to the owner.
PIPELINE_SECTIONS: dict[PageKind, list[str]] = {
    PageKind.PERSON: ["Who", "Relationship", "How we talk", "Shared history", "Earlier"],
    PageKind.TOPIC: ["What I think", "What I know", "Earlier"],
    PageKind.TASTE: ["Likes", "Dislikes", "Patterns", "Earlier"],
    PageKind.BEHAVIOUR: ["Pattern", "Examples", "Exceptions", "Earlier"],
    PageKind.TIMELINE: ["Events"],
    PageKind.HEALTH: ["Observations", "Conditions", "Treatments", "Earlier"],
}

#: Page that collects observations whose subject has too little material for its own page.
CATCH_ALL_SLUG = "misc"

#: Subject kinds an observation may target.
SubjectKind = Literal["person", "topic", "taste", "behaviour", "timeline", "health"]


def page_path(kind: PageKind | str, slug: str) -> str:
    """Page key relative to self/, without extension, e.g. "people/ana-ruiz"."""
    return f"{PAGE_DIRS[PageKind(kind)]}/{slug}"


class Observation(BaseModel):
    """One thing the extractor noticed in one chunk."""

    id: str = ""  # filled by the pipeline: "ob-" + hash(chunk_id, subject, text)
    chunk_id: str = ""  # filled by the pipeline
    subject_kind: SubjectKind
    subject: str  # person's name, topic label, year for timeline, ...
    text: str  # first person, standalone, one fact
    when: str | None = None  # ISO date, YYYY-MM or YYYY
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)  # item ids from the chunk


class ClaimStatus(StrEnum):
    INFERRED = "inferred"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class ClaimOrigin(StrEnum):
    PIPELINE = "pipeline"
    USER = "user"
    INTERVIEW = "interview"


class Claim(BaseModel):
    """One line of the self-model."""

    id: str  # "c-" + 8 hex; also the Obsidian block id on the line
    page: str  # page key, e.g. "people/ana-ruiz"
    section: str
    text: str
    status: ClaimStatus = ClaimStatus.INFERRED
    origin: ClaimOrigin = ClaimOrigin.PIPELINE
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)  # item ids
    valid_from: str | None = None
    valid_to: str | None = None
    created: datetime | None = None
    updated: datetime | None = None


# Operations: the only way the pipeline changes the self-model.


class AddClaim(BaseModel):
    op: Literal["add"] = "add"
    page: str
    section: str
    text: str
    evidence: list[str]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    valid_from: str | None = None
    valid_to: str | None = None
    origin: ClaimOrigin = ClaimOrigin.PIPELINE


class SupportClaim(BaseModel):
    op: Literal["support"] = "support"
    claim_id: str
    evidence: list[str]


class ReviseClaim(BaseModel):
    op: Literal["revise"] = "revise"
    claim_id: str
    text: str
    evidence: list[str] = Field(default_factory=list)


class SupersedeClaim(BaseModel):
    op: Literal["supersede"] = "supersede"
    claim_id: str  # the claim that stopped being true
    valid_to: str | None = None  # when it stopped, if known
    text: str  # the new claim
    evidence: list[str]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    valid_from: str | None = None


class AskQuestion(BaseModel):
    op: Literal["question"] = "question"
    text: str
    page: str | None = None
    related_claims: list[str] = Field(default_factory=list)


class CreatePage(BaseModel):
    op: Literal["create_page"] = "create_page"
    kind: PageKind
    slug: str
    title: str
    aliases: list[str] = Field(default_factory=list)


ClaimOp = Annotated[
    AddClaim | SupportClaim | ReviseClaim | SupersedeClaim | AskQuestion | CreatePage,
    Field(discriminator="op"),
]


class OpResult(BaseModel):
    """Outcome of applying one ClaimOp."""

    op: str
    ok: bool
    claim_id: str | None = None
    reason: str | None = None  # why it was refused, when ok is False


# ------------------------------------------------------------------------- voice


class DraftRequest(BaseModel):
    intent: str  # what to say, in the owner's words
    recipient: str | None = None  # person slug or name
    register_name: str | None = None  # e.g. "whatsapp.fr"; inferred when absent
    context: str | None = None  # e.g. the thread being answered
    variants: int = 1
    #: Item ids never to use as samples or reply pairs. The evals set this to the
    #: real text a draft is being compared with, so the mirror cannot copy from it.
    exclude_ids: list[str] = Field(default_factory=list)


class DraftResult(BaseModel):
    register_name: str
    variants: list[list[str]]  # each variant is one or more messages or paragraphs
    used: list[str] = Field(default_factory=list)  # ids of samples and pages used
    deviations: list[str] = Field(default_factory=list)  # what the style check still flags
