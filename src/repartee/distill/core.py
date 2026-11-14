"""Stage 4, core: rewrite the generated block of ``core.md`` from the strongest claims."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel

from repartee.config import OwnerConfig
from repartee.distill.prompts import CORE_SYSTEM, fill
from repartee.llm.base import LLMRequest, Message, SystemBlock
from repartee.schema import Claim, ClaimStatus, PageKind, stable_hash
from repartee.selfmodel import SelfModel

MAX_CLAIMS = 150
EARLIER = "Earlier"
#: Order in which kinds take turns when the cap forces a choice.
KIND_ORDER = (PageKind.PERSON, PageKind.TOPIC, PageKind.BEHAVIOUR, PageKind.TIMELINE,
              PageKind.TASTE, PageKind.HEALTH)


class CoreResult(BaseModel):
    """The generated part of core.md, as markdown."""

    markdown: str


def _strength(claim: Claim) -> tuple:
    confirmed = claim.status == ClaimStatus.CONFIRMED
    return (not confirmed, -claim.confidence, -(claim.updated.timestamp() if claim.updated else 0))


def current_claims(model: SelfModel) -> list[Claim]:
    """Live claims that still hold: not in Earlier, no end date."""
    return [c for c in model.claims() if c.section != EARLIER and not c.valid_to
            and c.status != ClaimStatus.REJECTED]


def select_claims(model: SelfModel, cap: int = MAX_CLAIMS) -> list[Claim]:
    """The strongest current claims, taking turns across page kinds up to the cap."""
    by_kind: dict[PageKind, list[Claim]] = defaultdict(list)
    for claim in current_claims(model):
        page = model.page(claim.page)
        if page is not None and page.kind in KIND_ORDER:
            by_kind[page.kind].append(claim)
    for claims in by_kind.values():
        claims.sort(key=_strength)
    chosen: list[Claim] = []
    depth = 0
    while len(chosen) < cap and any(depth < len(v) for v in by_kind.values()):
        for kind in KIND_ORDER:
            if depth < len(by_kind[kind]) and len(chosen) < cap:
                chosen.append(by_kind[kind][depth])
        depth += 1
    return chosen


def fingerprint(claims: list[Claim]) -> str:
    """Changes when the selected claims change in a way that matters to the core."""
    return stable_hash(*sorted(f"{c.id}|{c.text}|{c.status}|{c.confidence:.2f}" for c in claims),
                       length=16)


def build_request(model: SelfModel, claims: list[Claim], owner: OwnerConfig) -> LLMRequest:
    """The core request: selected claims grouped by page."""
    by_page: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        by_page[claim.page].append(claim)
    lines = []
    for key in sorted(by_page):
        page = model.page(key)
        title = page.title if page else key
        lines.append(f"## {title} ({key})")
        for claim in by_page[key]:
            mark = "confirmed" if claim.status == ClaimStatus.CONFIRMED else (
                f"inferred {claim.confidence:.2f}")
            since = f", since {claim.valid_from}" if claim.valid_from else ""
            lines.append(f"- [{claim.section}; {mark}{since}] {claim.text}")
        lines.append("")
    return LLMRequest(
        tier="smart",
        purpose="core",
        system=[SystemBlock(text=fill(CORE_SYSTEM, owner.name), cache=True)],
        messages=[Message(role="user", text="The strongest claims of my self-model:\n\n"
                          + "\n".join(lines).strip())],
        output_schema=CoreResult,
        max_tokens=8000,
    )


def clean_markdown(markdown: str) -> str:
    """The model's text, without anything that would break the generated block."""
    lines = [line for line in markdown.strip().splitlines()
             if "repartee:generated" not in line]
    return "\n".join(lines).strip()
