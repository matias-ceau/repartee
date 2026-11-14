"""Review: the owner walks through unconfirmed claims with the material behind them.

Confirm, edit or reject one claim at a time, lowest confidence first. Every action is
saved straight away, and the claims index is resynced so lookups see it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel

from repartee.mirror.context import claim_status, clip, fmt_when, indent
from repartee.schema import Claim, ClaimStatus
from repartee.selfmodel import SelfModel
from repartee.store import Corpus

EXCERPT_CHARS = 300


class Excerpt(BaseModel):
    """A piece of evidence, trimmed for display."""

    item_id: str
    when: datetime | None
    container: str
    author: str
    text: str  # trimmed to about 300 characters


class ReviewCard(BaseModel):
    """One claim with its page title and evidence."""

    claim: Claim
    page_title: str
    evidence: list[Excerpt]


def explain_claim(model: SelfModel, corpus: Corpus, claim_id: str) -> ReviewCard | None:
    """A claim with excerpts of the material behind it; None for an unknown id."""
    claim = model.claim(claim_id)
    if claim is None:
        return None
    page = model.page(claim.page)
    evidence = [
        Excerpt(item_id=item.id, when=item.timestamp, container=item.container,
                author=item.author, text=clip(item.text, EXCERPT_CHARS, flat=False))
        for item in corpus.get_items(claim.evidence)
    ]
    return ReviewCard(claim=claim, page_title=page.title if page else claim.page,
                      evidence=evidence)


def render_card(card: ReviewCard) -> str:
    """Plain text for a terminal: the claim, its status and its evidence."""
    claim = card.claim
    lines = [f"{card.page_title} › {claim.section}  [{claim.id} · {claim_status(claim)}]",
             f"  {claim.text}"]
    if card.evidence:
        lines.append("Evidence:")
        for ex in card.evidence:
            lines.append(f"  {fmt_when(ex.when)} · {ex.container} · {ex.author}")
            lines.append(indent(ex.text, "    "))
    else:
        missing = len(claim.evidence)
        lines.append(f"Evidence: {missing} item(s) no longer in the index." if missing
                     else "Evidence: none (written by the owner).")
    return "\n".join(lines)


class ReviewQueue:
    """Unconfirmed claims, lowest confidence first; optionally only `claim_ids`."""

    def __init__(self, model: SelfModel, corpus: Corpus, claim_ids: Sequence[str] | None = None):
        self.model = model
        self.corpus = corpus
        claims = model.claims(status=ClaimStatus.INFERRED)
        if claim_ids is not None:
            wanted = set(claim_ids)
            claims = [c for c in claims if c.id in wanted]
        self._ids = [c.id for c in sorted(claims, key=lambda c: c.confidence)]

    def _pending(self, claim_id: str) -> bool:
        claim = self.model.claim(claim_id)
        return claim is not None and claim.status == ClaimStatus.INFERRED

    def _head(self) -> str | None:
        while self._ids and not self._pending(self._ids[0]):
            self._ids.pop(0)  # confirmed, edited or removed elsewhere since
        return self._ids[0] if self._ids else None

    def _require(self) -> str:
        head = self._head()
        if head is None:
            raise LookupError("no claim left to review")
        return head

    def _done(self, claim_id: str) -> None:
        self._ids.remove(claim_id)
        self.model.save()
        self.corpus.sync_claims(self.model.claims())

    def remaining(self) -> int:
        """Claims left, the current one included."""
        return sum(1 for i in self._ids if self._pending(i))

    def next(self) -> ReviewCard | None:
        """The current claim, or None when the queue is done."""
        head = self._head()
        return explain_claim(self.model, self.corpus, head) if head else None

    def confirm(self) -> None:
        """The current claim is right as written."""
        claim_id = self._require()
        self.model.confirm(claim_id)
        self._done(claim_id)

    def edit(self, text: str) -> None:
        """The current claim, rewritten by the owner (and so confirmed)."""
        claim_id = self._require()
        self.model.edit(claim_id, text)
        self._done(claim_id)

    def reject(self) -> None:
        """The current claim is wrong; it is removed and will not be inferred again."""
        claim_id = self._require()
        self.model.reject(claim_id)
        self._done(claim_id)

    def skip(self) -> None:
        """Leave the current claim for another time."""
        self._ids.remove(self._require())
