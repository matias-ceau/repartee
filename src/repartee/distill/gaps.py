"""Stage 5, gaps: find thin, uncertain or contradictory areas and ask about them."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from repartee.config import OwnerConfig
from repartee.distill.prompts import GAPS_SYSTEM, fill
from repartee.llm.base import LLMRequest, Message, SystemBlock
from repartee.schema import PAGE_DIRS, ClaimStatus, PageKind, stable_hash
from repartee.selfmodel import SelfModel

MAX_QUESTIONS = 10
LEAST_CERTAIN = 25
CHANGED = 20
EARLIER = "Earlier"
COVERED_KINDS = [k for k in PAGE_DIRS if k is not PageKind.VOICE]


class GapQuestion(BaseModel):
    """One question for the owner (flat, for structured output)."""

    text: str
    page: str | None = None


class GapsResult(BaseModel):
    """Structured output of a ``gaps`` request."""

    questions: list[GapQuestion] = Field(default_factory=list)


def overview(model: SelfModel) -> str:
    """What the self-model holds, compactly: pages, coverage, weak and changed claims."""
    claims = model.claims()
    per_page = Counter(c.page for c in claims)
    confirmed = Counter(c.page for c in claims if c.status == ClaimStatus.CONFIRMED)
    lines = ["Pages (claims, of which confirmed):"]
    for kind in COVERED_KINDS:
        for page in model.pages(kind):
            lines.append(f"- {page.key} \"{page.title}\": {per_page[page.key]} "
                         f"({confirmed[page.key]})")
    empty = [str(k) for k in COVERED_KINDS if not model.pages(k)]
    if empty:
        lines.append(f"Kinds with no page at all: {', '.join(empty)}")
    weak = sorted((c for c in claims if c.status == ClaimStatus.INFERRED
                   and c.section != EARLIER), key=lambda c: c.confidence)[:LEAST_CERTAIN]
    if weak:
        lines += ["", "Least certain claims:"]
        lines += [f"- ({c.page}, {c.confidence:.2f}) {c.text}" for c in weak]
    changed = [c for c in claims if c.section == EARLIER][:CHANGED]
    if changed:
        lines += ["", "Claims that stopped being true (superseded):"]
        lines += [f"- ({c.page}, until {c.valid_to or '?'}) {c.text}" for c in changed]
    return "\n".join(lines)


def fingerprint(model: SelfModel) -> str:
    """Changes when the overview changes (open questions are not part of it)."""
    return stable_hash(overview(model), length=16)


def build_request(model: SelfModel, owner: OwnerConfig) -> LLMRequest:
    """The gaps request: overview plus the questions already waiting."""
    waiting = model.questions()
    text = overview(model) + "\n\nQuestions already waiting:\n"
    text += "\n".join(f"- {q.text}" for q in waiting) if waiting else "(none)"
    return LLMRequest(
        tier="smart",
        purpose="gaps",
        system=[SystemBlock(text=fill(GAPS_SYSTEM, owner.name), cache=True)],
        messages=[Message(role="user", text=text)],
        output_schema=GapsResult,
        max_tokens=4000,
    )


def add_questions(model: SelfModel, result: GapsResult) -> int:
    """Add up to MAX_QUESTIONS new questions; returns how many were new."""
    before = {q.id for q in model.questions(include_resolved=True)}
    added = 0
    for question in result.questions[:MAX_QUESTIONS]:
        page = (question.page or "").strip().removesuffix(".md") or None
        if page is not None and model.page(page) is None:
            page = None
        try:
            q = model.add_question(question.text, page)
        except ValueError:
            continue
        if q.id not in before:
            before.add(q.id)
            added += 1
    return added
