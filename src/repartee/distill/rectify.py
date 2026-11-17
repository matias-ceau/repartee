"""Stage 4, rectify: inferred claims checked against the key facts the owner confirmed.

When a page's confirmed facts change (the owner edited a property, or said something in an
interview), the inferred claims on that page and the ones elsewhere that name its subject
(title or aliases) go to the model in one request for that subject. Each comes back as keep,
revise (right event, wrong detail) or retract (false), applied through ClaimOps: a
retraction rests on the confirmed fact, a revision cites the fact's evidence.

Incremental: a page is sent again only when its confirmed facts changed since its last
check, and then with all its claims; otherwise only claims not checked yet are sent.
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
from repartee.selfmodel.pages import has_facts, mentions, value_key


class Verdict(BaseModel):
    """What to do with one inferred claim, given the confirmed facts."""

    claim_id: str
    action: Literal["keep", "revise", "retract"]
    text: str | None = Field(default=None, description="revise: the corrected claim.")
    slot: str | None = Field(default=None, description="The confirmed fact that decides it.")
    reason: str = Field(default="", description="Why, in a few plain words.")


class RectifyResult(BaseModel):
    """Verdicts for the claims to revise or retract; claims left out are kept."""

    verdicts: list[Verdict] = Field(default_factory=list)


@dataclass
class Job:
    """One subject to check: its confirmed facts and the claims to send."""

    page: str
    mark: str  # fingerprint of the confirmed facts
    facts: list[Fact]
    claims: list[Claim]  # to check now (may be empty: nothing new, only record the mark)
    candidates: list[str] = field(default_factory=list)  # every claim concerning the subject


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


def plan_jobs(model: SelfModel, state: RectifyState) -> list[Job]:
    """The subjects whose confirmed facts changed or that have unchecked claims."""
    jobs = []
    inferred = model.claims(status=ClaimStatus.INFERRED)
    for page in model.pages():
        if not has_facts(page.kind):
            continue
        facts = model.facts(page=page.key, status=ClaimStatus.CONFIRMED)
        if not facts:
            continue
        mark = fact_mark(facts)
        found = candidates(model, page.key, inferred)
        todo = found
        if state.facts(page.key) == mark:
            done = state.checked(page.key)
            todo = [c for c in found if c.id not in done]
            if not todo:
                continue
        jobs.append(Job(page=page.key, mark=mark, facts=facts, claims=todo,
                        candidates=[c.id for c in found]))
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
