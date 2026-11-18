"""Stage 4, rectify: inferred claims checked against the key facts the owner confirmed.

When a page's confirmed facts change (the owner edited a property, or said something in an
interview), the inferred claims on that page and the ones elsewhere that name its subject
(title or aliases) go to the model in one request for that subject. Each comes back as keep,
revise (right event, wrong detail) or retract (false), applied through ClaimOps: a
retraction rests on the confirmed fact, a revision cites the fact's evidence.

Incremental: a page is sent again only when its confirmed facts changed since its last
check, and then with all its claims; otherwise only claims not checked yet are sent.

Open questions about the subject (filed on its page, or naming it) go in the same request:
one the confirmed facts already answer is ticked, so the owner is not asked twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from repartee.config import OwnerConfig
from repartee.distill.prompts import RECTIFY_SYSTEM, fill
from repartee.distill.state import RectifyState
from repartee.llm.base import LLMRequest, Message, SystemBlock
from repartee.schema import Claim, ClaimStatus, Fact, RetractClaim, ReviseClaim, stable_hash
from repartee.selfmodel import SelfModel
from repartee.selfmodel.model import Question
from repartee.selfmodel.pages import has_facts, mentions, value_key


class Verdict(BaseModel):
    """What to do with one inferred claim, given the confirmed facts."""

    claim_id: str
    action: Literal["keep", "revise", "retract"]
    text: str | None = Field(default=None, description="revise: the corrected claim.")
    slot: str | None = Field(default=None, description="The confirmed fact that decides it.")
    reason: str = Field(default="", description="Why, in a few plain words.")


class Answered(BaseModel):
    """An open question the confirmed facts already answer."""

    question_id: str
    reason: str = Field(default="", description="Which fact answers it, in a few words.")


class RectifyResult(BaseModel):
    """Verdicts for the claims to revise or retract (claims left out are kept), and the
    open questions the facts answer (questions left out stay open)."""

    verdicts: list[Verdict] = Field(default_factory=list)
    answered: list[Answered] = Field(default_factory=list)


@dataclass
class Job:
    """One subject to check: its confirmed facts and the claims to send."""

    page: str
    mark: str  # fingerprint of the confirmed facts
    facts: list[Fact]
    claims: list[Claim]  # to check now (may be empty: nothing new, only record the mark)
    candidates: list[str] = field(default_factory=list)  # every claim concerning the subject
    questions: list[Question] = field(default_factory=list)  # open, to check now

    @property
    def checked(self) -> list[str]:
        """What to record as checked once this job is done: claims and questions."""
        return [*self.candidates, *(q.id for q in self.questions)]

    @property
    def needed(self) -> bool:
        """True when there is something to send."""
        return bool(self.claims or self.questions)


def fact_mark(facts: Sequence[Fact]) -> str:
    """Changes when the confirmed facts change in meaning."""
    return stable_hash(*sorted(f"{f.slot}={'|'.join(value_key(f.value))}" for f in facts),
                       length=16)


def candidates(model: SelfModel, key: str, inferred: Sequence[Claim] | None = None
               ) -> list[Claim]:
    """Inferred claims on the page, and inferred claims elsewhere that name its subject."""
    names = model.subject_names(key)
    if inferred is None:
        inferred = model.claims(status=ClaimStatus.INFERRED)
    return [c for c in inferred if c.page == key or mentions(c.text, names)]


def open_questions(model: SelfModel, key: str, waiting: Sequence[Question] | None = None
                   ) -> list[Question]:
    """Open questions filed on the page, or naming its subject."""
    names = model.subject_names(key)
    if waiting is None:
        waiting = model.questions()
    return [q for q in waiting if q.page == key or mentions(q.text, names)]


def plan_jobs(model: SelfModel, state: RectifyState) -> list[Job]:
    """The subjects whose confirmed facts changed or that have unchecked claims or questions."""
    jobs = []
    inferred = model.claims(status=ClaimStatus.INFERRED)
    waiting = model.questions()
    for page in model.pages():
        if not has_facts(page.kind):
            continue
        facts = model.facts(page=page.key, status=ClaimStatus.CONFIRMED)
        if not facts:
            continue
        mark = fact_mark(facts)
        found = candidates(model, page.key, inferred)
        asked = open_questions(model, page.key, waiting)
        todo, ask = found, asked
        if state.facts(page.key) == mark:
            done = state.checked(page.key)
            todo = [c for c in found if c.id not in done]
            ask = [q for q in asked if q.id not in done]
            if not todo and not ask:
                continue
        jobs.append(Job(page=page.key, mark=mark, facts=facts, claims=todo,
                        candidates=[c.id for c in found], questions=ask))
    return jobs


def build_request(model: SelfModel, job: Job, owner: OwnerConfig) -> LLMRequest:
    """The rectify request for one subject."""
    page = model.page(job.page)
    title = page.title if page else job.page
    aliases = f", also known as {', '.join(page.aliases)}" if page and page.aliases else ""
    lines = [f"Subject: {job.page} \"{title}\"{aliases}", "", "Confirmed key facts:"]
    lines += [f"- {f.slot}: {f.shown()}" for f in job.facts]
    lines += ["", "Inferred claims to check:"]
    for claim in job.claims:
        lines.append(f"- {claim.id} (on {claim.page}, {claim.section}) {claim.text}")
    if not job.claims:
        lines.append("(none)")
    lines += ["", "Open questions to check:"]
    lines += [f"- {q.id} {q.text}" for q in job.questions] or ["(none)"]
    return LLMRequest(
        tier="smart",
        purpose="rectify",
        system=[SystemBlock(text=fill(RECTIFY_SYSTEM, owner.name), cache=True)],
        messages=[Message(role="user", text="\n".join(lines))],
        output_schema=RectifyResult,
    )


def to_ops(job: Job, result: RectifyResult) -> list[object]:
    """Verdicts as ClaimOps, for claims of this job only."""
    asked = {c.id for c in job.claims}
    by_slot = {f.slot: f for f in job.facts}
    ops: list[object] = []
    for verdict in result.verdicts:
        claim_id = verdict.claim_id.strip()
        if claim_id not in asked or verdict.action == "keep":
            continue
        slot = (verdict.slot or "").strip()
        fact = by_slot.get(slot) or job.facts[0]
        reason = " ".join(verdict.reason.split()) or (
            f"it contradicts the confirmed {fact.slot}: {fact.shown()}")
        if verdict.action == "retract":
            ops.append(RetractClaim(claim_id=claim_id, reason=reason, evidence=fact.evidence,
                                    fact_page=job.page, fact_slot=fact.slot))
        elif verdict.action == "revise" and (verdict.text or "").strip():
            ops.append(ReviseClaim(claim_id=claim_id, text=verdict.text.strip(),
                                   evidence=fact.evidence))
    return ops


def answered(job: Job, result: RectifyResult) -> list[str]:
    """Ids of this job's questions that the facts answer."""
    asked = {q.id for q in job.questions}
    return [a.question_id.strip() for a in result.answered if a.question_id.strip() in asked]
