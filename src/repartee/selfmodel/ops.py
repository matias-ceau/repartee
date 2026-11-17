"""Pipeline operations on the self-model, validated before anything is touched.

Each op either changes the model completely or not at all, and comes back as an
``OpResult``. A refused op carries one of the reason codes below.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, TypeAdapter

from repartee.schema import (
    FACT_SLOTS,
    PAGE_DIRS,
    PIPELINE_SECTIONS,
    AddClaim,
    AskQuestion,
    Claim,
    ClaimOp,
    ClaimOrigin,
    ClaimStatus,
    CreatePage,
    Fact,
    OpResult,
    PageKind,
    RetractClaim,
    ReviseClaim,
    SetFact,
    SupersedeClaim,
    SupportClaim,
    page_path,
    slugify,
)
from repartee.selfmodel.pages import (
    EARLIER,
    canonical_section,
    clean_text,
    has_facts,
    link_text,
    mentions,
    new_claim_id,
    new_page,
    norm_text,
    value_key,
)

if TYPE_CHECKING:
    from repartee.selfmodel.model import SelfModel

log = logging.getLogger(__name__)

UNKNOWN_CLAIM = "unknown_claim"  # no live or rejected claim has this id
CONFIRMED = "confirmed"  # the owner vouched for it; only support is allowed
REJECTED = "rejected"  # the owner deleted it
SECTION = "section"  # not a pipeline section of that page kind, or the line is the owner's
NO_PAGE = "no_page"  # add to a page that does not exist (create_page first)
NO_EVIDENCE = "no_evidence"
UNKNOWN_EVIDENCE = "unknown_evidence"  # none of the evidence ids exist
MATCHES_REJECTED = "matches_rejected"  # the owner already rejected this text on that page
DUPLICATE = "duplicate"  # same normalized text as a live claim (claim_id names it) or question
INVALID = "invalid"  # malformed op, empty or unsafe text, forbidden origin or page kind
UNBACKED = "unbacked"  # a retraction with neither a confirmed fact nor the owner's own words
ERROR = "error"  # a bug: logged, op not applied
#: A refusal for CONFIRMED on set_fact or retract carries the id of the question it asked
#: instead (claim_id "q-...").

MAX_FACT_CHARS = 120

_ADAPTER: TypeAdapter = TypeAdapter(ClaimOp)

Evidence = Callable[[Iterable[str]], set[str]] | None


@dataclass
class _Ctx:
    existing: Evidence  # which evidence ids exist (None: trust them all)
    statements: Evidence  # which evidence ids are the owner's own interview/mirror turns


def apply_ops(model: SelfModel, ops: Sequence[object], existing: Evidence,
              statements: Evidence = None) -> list[OpResult]:
    """Apply each op in order; never raises for a bad op.

    ``statements`` tells which evidence ids are the owner's own turns in an interview or a
    mirror session. Without it nothing counts as the owner's word: no fact is confirmed by
    evidence and no retraction can rest on evidence.
    """
    ctx = _Ctx(existing, statements)
    results = []
    for raw in ops:
        name = str(getattr(raw, "op", None) or (raw.get("op") if isinstance(raw, dict) else "?"))
        try:
            data = raw.model_dump(warnings=False) if isinstance(raw, BaseModel) else raw
            op = _ADAPTER.validate_python(data)
        except Exception:  # noqa: BLE001 - any malformed input is a refusal
            results.append(OpResult(op=name, ok=False, reason=INVALID))
            continue
        try:
            results.append(_HANDLERS[op.op](model, op, ctx))
        except Exception:  # noqa: BLE001 - a bug must not stop the run
            log.exception("self-model: applying %r failed", op)
            results.append(OpResult(op=op.op, ok=False, reason=ERROR))
    for r in results:
        if not r.ok:
            log.info("self-model: %s refused (%s) %s", r.op, r.reason, r.claim_id or "")
    return results


def _refuse(op: str, reason: str, claim_id: str | None = None) -> OpResult:
    return OpResult(op=op, ok=False, reason=reason, claim_id=claim_id)


def _evidence(ids: list[str], existing: Evidence, required: bool) -> tuple[list[str], str | None]:
    clean = []
    for i in ids:
        i = i.strip()
        if i and i not in clean:
            clean.append(i)
    if not clean:
        return [], NO_EVIDENCE if required else None
    if existing is not None:
        known = set(existing(clean))
        clean = [i for i in clean if i in known]
        if not clean:
            return [], UNKNOWN_EVIDENCE if required else None
    return clean, None


def _lookup(model: SelfModel, op: str, claim_id: str) -> tuple[Claim | None, OpResult | None]:
    claim = model._claims.get(claim_id)
    if claim is not None:
        return claim, None
    if claim_id in model._rejected_ids():
        return None, _refuse(op, REJECTED, claim_id)
    return None, _refuse(op, UNKNOWN_CLAIM, claim_id)


def _pipeline_line(model: SelfModel, claim: Claim) -> bool:
    """True when the claim sits as a top-level line in a pipeline section of its page."""
    page = model._pages.get(claim.page)
    line = page.claim_line(claim.id) if page else None
    return bool(line and line.top_level and canonical_section(page.kind, line.section))


def _text_conflict(model: SelfModel, key: str, text: str, skip: str | None) -> OpResult | None:
    norm = norm_text(text)
    for rec in model._rejected.get(key, []):
        if norm_text(rec.get("text", "")) == norm:
            return _refuse("", MATCHES_REJECTED)
    for claim in model._claims.values():
        if claim.page == key and claim.id != skip and claim.section != EARLIER \
                and norm_text(claim.text) == norm:
            return _refuse("", DUPLICATE, claim.id)
    return None


def _with_op(result: OpResult, op: str) -> OpResult:
    return result.model_copy(update={"op": op})


def _add(model: SelfModel, op: AddClaim, ctx: _Ctx) -> OpResult:
    existing = ctx.existing
    key = op.page.strip().removesuffix(".md")
    page = model._pages.get(key)
    if page is None:
        return _refuse(op.op, NO_PAGE)
    section = canonical_section(page.kind, op.section)
    if section is None:
        return _refuse(op.op, SECTION)
    text = clean_text(op.text)
    if text is None or op.origin == ClaimOrigin.USER:
        return _refuse(op.op, INVALID)
    evidence, reason = _evidence(op.evidence, existing, required=True)
    if reason:
        return _refuse(op.op, reason)
    conflict = _text_conflict(model, key, text, skip=None)
    if conflict:
        return _with_op(conflict, op.op)
    cid = _fresh_id(model)
    page.add_claim(section, text, cid)
    now = _now()
    model._claims[cid] = Claim(
        id=cid, page=key, section=section, text=text, origin=op.origin,
        confidence=op.confidence, evidence=evidence, valid_from=op.valid_from,
        valid_to=op.valid_to, created=now, updated=now,
    )
    return OpResult(op=op.op, ok=True, claim_id=cid)


def _support(model: SelfModel, op: SupportClaim, ctx: _Ctx) -> OpResult:
    claim, refusal = _lookup(model, op.op, op.claim_id)
    if refusal:
        return refusal
    evidence, reason = _evidence(op.evidence, ctx.existing, required=True)
    if reason:
        return _refuse(op.op, reason, op.claim_id)
    new = [e for e in evidence if e not in claim.evidence]
    if new:
        c = claim.confidence
        for _ in new:
            c = c + (1 - c) * 0.25
        claim.confidence = max(claim.confidence, min(c, 0.95))
        claim.evidence = claim.evidence + new
        claim.updated = _now()
    return OpResult(op=op.op, ok=True, claim_id=claim.id)


def _changeable(model: SelfModel, op: str, claim_id: str) -> tuple[Claim | None, OpResult | None]:
    claim, refusal = _lookup(model, op, claim_id)
    if refusal:
        return None, refusal
    if claim.status == ClaimStatus.CONFIRMED:
        return None, _refuse(op, CONFIRMED, claim_id)
    if claim.status == ClaimStatus.REJECTED:
        return None, _refuse(op, REJECTED, claim_id)
    if not _pipeline_line(model, claim):
        return None, _refuse(op, SECTION, claim_id)
    return claim, None


def _revise(model: SelfModel, op: ReviseClaim, ctx: _Ctx) -> OpResult:
    claim, refusal = _changeable(model, op.op, op.claim_id)
    if refusal:
        return refusal
    text = clean_text(op.text)
    if text is None:
        return _refuse(op.op, INVALID, op.claim_id)
    conflict = _text_conflict(model, claim.page, text, skip=claim.id)
    if conflict:
        return _with_op(conflict, op.op)
    evidence, _ = _evidence(op.evidence, ctx.existing, required=False)
    model._pages[claim.page].set_claim_text(claim.id, text)
    claim.text = text
    claim.evidence = claim.evidence + [e for e in evidence if e not in claim.evidence]
    claim.updated = _now()
    return OpResult(op=op.op, ok=True, claim_id=claim.id)


def _supersede(model: SelfModel, op: SupersedeClaim, ctx: _Ctx) -> OpResult:
    claim, refusal = _changeable(model, op.op, op.claim_id)
    if refusal:
        return refusal
    page = model._pages[claim.page]
    if claim.section == EARLIER or EARLIER not in PIPELINE_SECTIONS.get(page.kind, []):
        return _refuse(op.op, SECTION, op.claim_id)
    text = clean_text(op.text)
    if text is None:
        return _refuse(op.op, INVALID, op.claim_id)
    evidence, reason = _evidence(op.evidence, ctx.existing, required=True)
    if reason:
        return _refuse(op.op, reason, op.claim_id)
    conflict = _text_conflict(model, claim.page, text, skip=None)
    if conflict:
        return _with_op(conflict, op.op)
    section = claim.section
    now = _now()
    page.move_claim(claim.id, EARLIER)
    claim.section, claim.updated = EARLIER, now
    claim.valid_to = op.valid_to or op.valid_from or claim.valid_to
    cid = _fresh_id(model)
    page.add_claim(section, text, cid)
    model._claims[cid] = Claim(
        id=cid, page=claim.page, section=section, text=text, origin=claim.origin,
        confidence=op.confidence, evidence=evidence, valid_from=op.valid_from,
        created=now, updated=now,
    )
    return OpResult(op=op.op, ok=True, claim_id=cid)


def _question(model: SelfModel, op: AskQuestion, ctx: _Ctx) -> OpResult:
    text = clean_text(op.text)
    if text is None:
        return _refuse(op.op, INVALID)
    for q in model.questions(include_resolved=True):
        if norm_text(q.text) == norm_text(text):
            return _refuse(op.op, DUPLICATE, q.id)
    q = model.add_question(text, op.page)
    return OpResult(op=op.op, ok=True, claim_id=q.id)


def _create_page(model: SelfModel, op: CreatePage, ctx: _Ctx) -> OpResult:
    kind = PageKind(op.kind)
    if kind not in PAGE_DIRS or kind is PageKind.VOICE:
        return _refuse(op.op, INVALID)
    slug = slugify(op.slug)
    key = page_path(kind, slug)
    if key in model._pages:
        return OpResult(op=op.op, ok=True)
    title = " ".join(op.title.split()) or slug
    model._pages[key] = new_page(kind, slug, title, op.aliases)
    return OpResult(op=op.op, ok=True)


def _stated(ctx: _Ctx, evidence: list[str]) -> bool:
    """True when some of the evidence is the owner's own interview or mirror-session turn."""
    return bool(evidence and ctx.statements is not None and set(ctx.statements(evidence)))


def _fact_value(model: SelfModel, slot: str, raw: object) -> str | None:
    """A value safe for one frontmatter line; names in link slots become [[links]] to pages."""
    if not isinstance(raw, str):
        return None
    value = " ".join(raw.split())
    if not value or len(value) > MAX_FACT_CHARS or "\n" in value:
        return None
    if FACT_SLOTS[slot].link:
        name = link_text(value)
        key = model.resolve(PageKind.PERSON, name)
        page = model._pages.get(key) if key else None
        if page is not None:
            base = key.rsplit("/", 1)[-1]
            return f"[[{base}]]" if base == page.title else f"[[{base}|{page.title}]]"
        return name
    return value


def _set_fact(model: SelfModel, op: SetFact, ctx: _Ctx) -> OpResult:
    key = op.page.strip().removesuffix(".md")
    page = model._pages.get(key)
    if page is None:
        return _refuse(op.op, NO_PAGE)
    slot = op.slot.strip().lower()
    if slot not in FACT_SLOTS or not has_facts(page.kind) or not page.frontmatter_ok():
        return _refuse(op.op, INVALID)
    value = _fact_value(model, slot, op.value)
    if value is None:
        return _refuse(op.op, INVALID)
    evidence, reason = _evidence(op.evidence, ctx.existing, required=True)
    if reason:
        return _refuse(op.op, reason)
    if model.fact_rejected(key, slot, value):
        return _refuse(op.op, MATCHES_REJECTED)
    stated = _stated(ctx, evidence)
    spec = FACT_SLOTS[slot]
    now = _now()
    current = model._facts.get(key, {}).get(slot)
    if current is not None and value_key(value)[0] in value_key(current.value):
        # the same value again: more evidence, and the owner's word confirms an inferred one
        new = [e for e in evidence if e not in current.evidence]
        if new:
            current.evidence = current.evidence + new
            current.confidence = max(current.confidence, min(
                current.confidence + (1 - current.confidence) * 0.25 * len(new), 0.95))
            current.updated = now
        if stated and current.status == ClaimStatus.INFERRED and not spec.multi:
            current.status, current.origin, current.updated = (
                ClaimStatus.CONFIRMED, ClaimOrigin.INTERVIEW, now)
        return OpResult(op=op.op, ok=True)
    if current is not None and current.status == ClaimStatus.CONFIRMED:
        q = model.add_question(
            f"You gave {page.title}'s {slot} as {current.shown()}, but something in your "
            f"material suggests {link_text(value)}. Which is right?", key)
        return _refuse(op.op, CONFIRMED, q.id)
    if current is not None and spec.multi:
        current.value = current.values() + [value]
        current.evidence = current.evidence + [e for e in evidence if e not in current.evidence]
        current.updated = now  # stays inferred: the values already there are not stated
        fact = current
    else:
        fact = Fact(
            page=key, slot=slot, value=[value] if spec.multi else value,
            status=ClaimStatus.CONFIRMED if stated else ClaimStatus.INFERRED,
            origin=ClaimOrigin.INTERVIEW if stated else ClaimOrigin.PIPELINE,
            confidence=1.0 if stated else op.confidence, evidence=evidence,
            created=current.created if current else now, updated=now)
        model._facts.setdefault(key, {})[slot] = fact
    page.set_property(slot, fact.value)
    return OpResult(op=op.op, ok=True)


def _concerns(model: SelfModel, claim: Claim, key: str) -> bool:
    """True when the claim is on the page or names the page's subject."""
    return claim.page == key or mentions(claim.text, model.subject_names(key))


def _retract(model: SelfModel, op: RetractClaim, ctx: _Ctx) -> OpResult:
    claim, refusal = _lookup(model, op.op, op.claim_id)
    if refusal:
        return refusal
    reason = clean_text(op.reason)
    if reason is None:
        return _refuse(op.op, INVALID, op.claim_id)
    if claim.status == ClaimStatus.CONFIRMED:
        q = model.add_question(
            f"You confirmed \"{claim.text}\", but {reason.rstrip('.')}. Is it still right?",
            claim.page)
        return _refuse(op.op, CONFIRMED, q.id)
    if not _pipeline_line(model, claim):
        return _refuse(op.op, SECTION, op.claim_id)
    backing: dict[str, object] = {}
    if op.fact_slot:
        key = (op.fact_page or claim.page).strip().removesuffix(".md")
        fact = model._facts.get(key, {}).get(op.fact_slot.strip())
        if fact is not None and fact.status == ClaimStatus.CONFIRMED \
                and _concerns(model, claim, key):
            backing = {"fact": f"{key}:{fact.slot}"}
    if not backing:
        evidence, _ = _evidence(op.evidence, ctx.existing, required=False)
        if _stated(ctx, evidence):
            backing = {"retract_evidence": evidence}
    if not backing:
        return _refuse(op.op, UNBACKED, op.claim_id)
    model._pages[claim.page].remove_claim(claim.id)
    model._record_rejection(claim.page, claim, _now(), retracted=True, reason=reason, **backing)
    del model._claims[claim.id]
    return OpResult(op=op.op, ok=True, claim_id=claim.id)


def _fresh_id(model: SelfModel) -> str:
    taken = model._rejected_ids()
    cid = new_claim_id()
    while cid in model._claims or cid in taken:
        cid = new_claim_id()
    return cid


def _now():
    from repartee.selfmodel.model import _now as now

    return now()


_HANDLERS = {
    "add": _add,
    "support": _support,
    "revise": _revise,
    "supersede": _supersede,
    "question": _question,
    "create_page": _create_page,
    "set_fact": _set_fact,
    "retract": _retract,
}
