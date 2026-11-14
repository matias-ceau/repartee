"""Stage 3, consolidate: fold a page's new observations into claim operations.

The model sees the page's live claims and a batch of observations, each with a short
ref (``o1``, ``o2``...). It returns flat operations that cite refs, never item ids;
evidence and confidence are computed here from the cited observations, so a claim can
only rest on material an observation was extracted from.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from repartee.config import OwnerConfig
from repartee.distill.extract import valid_date
from repartee.distill.prompts import CONSOLIDATE_SYSTEM, fill
from repartee.llm.base import LLMRequest, Message, SystemBlock
from repartee.schema import (
    PIPELINE_SECTIONS,
    AddClaim,
    AskQuestion,
    ClaimOrigin,
    ClaimStatus,
    Kind,
    Observation,
    OpResult,
    ReviseClaim,
    SupersedeClaim,
    SupportClaim,
)
from repartee.selfmodel import SelfModel
from repartee.selfmodel.ops import DUPLICATE

BATCH_SIZE = 40
MAX_CONFIDENCE = 0.95
EARLIER = "Earlier"

_SOURCE_LABEL: dict[str, str] = {
    "message": "chat", "note": "note", "journal": "journal", "essay": "essay",
    "interview": "interview", "medical": "health note", "taste": "tastes list",
    "mirror": "mirror session", "draft": "sent text",
}


class PlannedOp(BaseModel):
    """One change to the page, citing the observations (o1, o2...) it rests on."""

    op: Literal["add", "support", "revise", "supersede", "question"]
    claim_id: str | None = None
    section: str | None = None
    text: str | None = None
    observations: list[str] = Field(default_factory=list, description="Refs such as o1.")
    valid_from: str | None = None
    valid_to: str | None = None


class ConsolidationResult(BaseModel):
    """The operations that bring the page up to date; empty when nothing should change."""

    ops: list[PlannedOp] = Field(default_factory=list)


@dataclass
class BatchOutcome:
    """What applying one batch did."""

    applied: int = 0
    refused: int = 0
    questions: int = 0
    claim_ids: list[str] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)  # "op: reason", for the log


def batches(observations: Sequence[Observation], size: int = BATCH_SIZE) -> list[list[Observation]]:
    """Observations in date order, cut into batches."""
    ordered = sorted(observations, key=lambda o: (o.when or "9999", o.chunk_id, o.id))
    return [ordered[i: i + size] for i in range(0, len(ordered), size)]


def refs(batch: Sequence[Observation]) -> dict[str, Observation]:
    """Short refs (o1, o2, ...) for a batch, in batch order."""
    return {f"o{n}": obs for n, obs in enumerate(batch, 1)}


def build_request(model: SelfModel, key: str, batch: Sequence[Observation], owner: OwnerConfig,
                  source_of: dict[str, str]) -> LLMRequest:
    """The consolidate request for one batch of one page."""
    page = model.page(key)
    if page is None:
        raise KeyError(key)
    sections = [s for s in PIPELINE_SECTIONS.get(page.kind, []) if s != EARLIER]
    aliases = f", also known as {', '.join(page.aliases)}" if page.aliases else ""
    lines = [f"Page: {key} (a {page.kind} page) \"{page.title}\"{aliases}",
             f"Sections you may add to: {', '.join(sections)}", "", "Current claims:"]
    claims = model.claims(page=key)
    for claim in claims:
        status = "confirmed by me" if claim.status == ClaimStatus.CONFIRMED else "inferred"
        span = ""
        if claim.valid_from or claim.valid_to:
            span = f", valid {claim.valid_from or '?'} to {claim.valid_to or 'now'}"
        lines.append(f"- {claim.id} [{claim.section}] ({status}, confidence "
                     f"{claim.confidence:.2f}{span}) {claim.text}")
    if not claims:
        lines.append("(none yet)")
    lines += ["", "New observations:"]
    for ref, obs in refs(batch).items():
        source = _SOURCE_LABEL.get(source_of.get(obs.chunk_id, ""), "material")
        when = f", {obs.when}" if obs.when else ""
        lines.append(f"- {ref} (from a {source}{when}, confidence {obs.confidence:.2f}; "
                     f"{obs.subject_kind}: {obs.subject}) {obs.text}")
    return LLMRequest(
        tier="smart",
        purpose="consolidate",
        system=[SystemBlock(text=fill(CONSOLIDATE_SYSTEM, owner.name), cache=True)],
        messages=[Message(role="user", text="\n".join(lines))],
        output_schema=ConsolidationResult,
    )


def combined_confidence(values: Iterable[float]) -> float:
    """The strongest observation, raised a quarter of the way to 1 by each other one."""
    ordered = sorted(values, reverse=True)
    if not ordered:
        return 0.5
    c = ordered[0]
    for _ in ordered[1:]:
        c += (1 - c) * 0.25
    return round(min(c, MAX_CONFIDENCE), 4)


def to_claim_ops(planned: Sequence[PlannedOp], batch_refs: dict[str, Observation], key: str,
                 item_kinds: dict[str, Kind]) -> tuple[list[object], int]:
    """Convert the model's flat ops to ClaimOps; returns (ops, number malformed)."""
    ops: list[object] = []
    bad = 0
    for p in planned:
        cited = [batch_refs[r.strip()] for r in dict.fromkeys(p.observations)
                 if r.strip() in batch_refs]
        evidence = list(dict.fromkeys(e for obs in cited for e in obs.evidence))
        text = " ".join((p.text or "").split())
        claim_id = (p.claim_id or "").strip()
        valid_from = valid_date(p.valid_from)
        if valid_from is None:
            dates = {obs.when for obs in cited if obs.when}
            valid_from = dates.pop() if len(dates) == 1 else None
        confidence = combined_confidence(obs.confidence for obs in cited)
        if p.op == "add" and text and p.section and evidence:
            interview = all(item_kinds.get(e) == Kind.INTERVIEW for e in evidence)
            ops.append(AddClaim(
                page=key, section=p.section, text=text, evidence=evidence,
                confidence=confidence, valid_from=valid_from, valid_to=valid_date(p.valid_to),
                origin=ClaimOrigin.INTERVIEW if interview else ClaimOrigin.PIPELINE))
        elif p.op == "support" and claim_id and evidence:
            ops.append(SupportClaim(claim_id=claim_id, evidence=evidence))
        elif p.op == "revise" and claim_id and text:
            ops.append(ReviseClaim(claim_id=claim_id, text=text, evidence=evidence))
        elif p.op == "supersede" and claim_id and text and evidence:
            ops.append(SupersedeClaim(
                claim_id=claim_id, text=text, evidence=evidence, confidence=confidence,
                valid_from=valid_from, valid_to=valid_date(p.valid_to)))
        elif p.op == "question" and text:
            ops.append(AskQuestion(text=text, page=key))
        else:
            bad += 1
    return ops, bad


def apply_batch(model: SelfModel, ops: Sequence[object],
                existing: Callable[[Iterable[str]], set[str]] | None) -> BatchOutcome:
    """Apply ops; a duplicate add becomes support for the claim it duplicates."""
    outcome = BatchOutcome()

    def count(result: OpResult) -> None:
        if result.ok:
            outcome.applied += 1
            if result.op == "question":
                outcome.questions += 1
            elif result.claim_id and result.claim_id.startswith("c-"):
                outcome.claim_ids.append(result.claim_id)
        else:
            outcome.refused += 1
            outcome.refusals.append(f"{result.op}: {result.reason}")

    for op, result in zip(ops, model.apply(ops, existing_evidence=existing), strict=True):
        if (not result.ok and result.reason == DUPLICATE and result.claim_id
                and result.claim_id.startswith("c-") and getattr(op, "evidence", None)):
            (result,) = model.apply(
                [SupportClaim(claim_id=result.claim_id, evidence=list(op.evidence))],
                existing_evidence=existing)
        count(result)
    return outcome
