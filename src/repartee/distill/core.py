"""Stage 5, core: rewrite the generated block of ``core.md`` from the strongest claims.

The request has two blocks. First what the owner stated: confirmed key facts, confirmed
claims and claims from interviews, which the summary must reflect. Then the inferred claims.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection, Iterable

from pydantic import BaseModel

from repartee.config import OwnerConfig
from repartee.distill.prompts import CORE_SYSTEM, fill
from repartee.llm.base import LLMRequest, Message, SystemBlock
from repartee.schema import Claim, ClaimOrigin, ClaimStatus, Fact, PageKind, stable_hash
from repartee.selfmodel import SelfModel

MAX_CLAIMS = 150
EARLIER = "Earlier"
#: Order in which kinds take turns when the cap forces a choice.
KIND_ORDER = (PageKind.PERSON, PageKind.TOPIC, PageKind.BEHAVIOUR, PageKind.TIMELINE,
              PageKind.TASTE, PageKind.HEALTH)


class CoreResult(BaseModel):
    """The generated part of core.md, as markdown."""

    markdown: str


def stated(claim: Claim, said: Collection[str] = ()) -> bool:
    """True for what the owner said: confirmed claims, claims from their interviews, and
    claims in ``said`` (ones that cite the owner's own interview or session turn)."""
    return (claim.status == ClaimStatus.CONFIRMED or claim.origin == ClaimOrigin.INTERVIEW
            or claim.id in said)


def said_claims(model: SelfModel, statements: Callable[[Iterable[str]], set[str]]) -> set[str]:
    """Ids of claims whose evidence includes something the owner stated."""
    claims = model.claims()
    mine = statements({e for c in claims for e in c.evidence})
    return {c.id for c in claims if mine.intersection(c.evidence)}


def _strength(claim: Claim, said: Collection[str] = ()) -> tuple:
    confirmed = claim.status == ClaimStatus.CONFIRMED
    return (not confirmed, not stated(claim, said), -claim.confidence,
            -(claim.updated.timestamp() if claim.updated else 0))


def select_facts(model: SelfModel) -> list[Fact]:
    """Key facts of subject pages: confirmed ones first, then inferred."""
    facts = [f for f in model.facts() if (p := model.page(f.page)) is not None
             and p.kind in KIND_ORDER]
    return sorted(facts, key=lambda f: f.status != ClaimStatus.CONFIRMED)


def current_claims(model: SelfModel) -> list[Claim]:
    """Live claims that still hold: not in Earlier, no end date."""
    return [c for c in model.claims() if c.section != EARLIER and not c.valid_to
            and c.status != ClaimStatus.REJECTED]


def select_claims(model: SelfModel, cap: int = MAX_CLAIMS,
                  said: Collection[str] = ()) -> list[Claim]:
    """The strongest current claims, taking turns across page kinds up to the cap."""
    by_kind: dict[PageKind, list[Claim]] = defaultdict(list)
    for claim in current_claims(model):
        page = model.page(claim.page)
        if page is not None and page.kind in KIND_ORDER:
            by_kind[page.kind].append(claim)
    for claims in by_kind.values():
        claims.sort(key=lambda c: _strength(c, said))
    chosen: list[Claim] = []
    depth = 0
    while len(chosen) < cap and any(depth < len(v) for v in by_kind.values()):
        for kind in KIND_ORDER:
            if depth < len(by_kind[kind]) and len(chosen) < cap:
                chosen.append(by_kind[kind][depth])
        depth += 1
    return chosen


def fingerprint(claims: list[Claim], facts: list[Fact] | None = None,
                said: Collection[str] = ()) -> str:
    """Changes when the selected claims or facts change in a way that matters to the core."""
    parts = [f"{c.id}|{c.text}|{c.status}|{c.confidence:.2f}"
             + ("|stated" if c.id in said else "") for c in claims]
    parts += [f"{f.page}:{f.slot}|{f.shown()}|{f.status}" for f in facts or []]
    return stable_hash(*sorted(parts), length=16)


def _block(model: SelfModel, claims: list[Claim], facts: list[Fact], *,
           with_status: bool, said: Collection[str] = ()) -> list[str]:
    by_page: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        by_page[claim.page].append(claim)
    facts_by_page: dict[str, list[Fact]] = defaultdict(list)
    for fact in facts:
        facts_by_page[fact.page].append(fact)
    lines = []
    for key in sorted(set(by_page) | set(facts_by_page)):
        page = model.page(key)
        title = page.title if page else key
        lines.append(f"## {title} ({key})")
        shown = "; ".join(f"{f.slot}: {f.shown()}" for f in facts_by_page[key])
        if shown:
            lines.append(f"- key facts: {shown}")
        for claim in by_page[key]:
            if claim.status == ClaimStatus.CONFIRMED:
                mark = "confirmed"
            elif claim.origin == ClaimOrigin.INTERVIEW or claim.id in said:
                mark = "said in an interview"
            else:
                mark = f"inferred {claim.confidence:.2f}"
            since = f", since {claim.valid_from}" if claim.valid_from else ""
            lines.append(f"- [{claim.section}; {mark}{since}] {claim.text}" if with_status
                         else f"- [{claim.section}{since}] {claim.text}")
        lines.append("")
    return lines


def build_request(model: SelfModel, claims: list[Claim], owner: OwnerConfig,
                  facts: list[Fact] | None = None, said: Collection[str] = ()) -> LLMRequest:
    """The core request: what I stated myself first, then inferred claims, grouped by page."""
    facts = facts or []
    mine = [f for f in facts if f.status == ClaimStatus.CONFIRMED]
    guessed = [f for f in facts if f.status != ClaimStatus.CONFIRMED]
    first = _block(model, [c for c in claims if stated(c, said)], mine, with_status=True,
                   said=said)
    second = _block(model, [c for c in claims if not stated(c, said)], guessed,
                    with_status=True)
    parts = []
    if first:
        parts.append("# What I have stated myself (reflect all of it; it overrides anything "
                     "inferred below)\n\n" + "\n".join(first).strip())
    if second:
        parts.append("# What was inferred from my material (less certain)\n\n"
                     + "\n".join(second).strip())
    return LLMRequest(
        tier="smart",
        purpose="core",
        system=[SystemBlock(text=fill(CORE_SYSTEM, owner.name), cache=True)],
        messages=[Message(role="user", text="The strongest claims of my self-model:\n\n"
                          + "\n\n".join(parts))],
        output_schema=CoreResult,
        max_tokens=8000,
    )


def clean_markdown(markdown: str) -> str:
    """The model's text, without anything that would break the generated block."""
    lines = [line for line in markdown.strip().splitlines()
             if "repartee:generated" not in line]
    return "\n".join(lines).strip()
