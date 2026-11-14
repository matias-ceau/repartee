"""The self-model: pages on disk, claim metadata in sidecars, owner edits reconciled on load.

Layout under ``self/``: one markdown page per subject (``people/ana-ruiz.md``), and
for each page a sidecar ``.claims/<page key>.json``::

    {"version": 1,
     "claims": {"c-3fa91c20": {"section", "text", "status", "origin", "confidence",
                               "evidence", "valid_from", "valid_to", "created", "updated"}},
     "rejected": [{"id", "text", "norm_hash", "section", "evidence", "rejected_at",
                   "status", "origin", "confidence", "valid_from", "valid_to", "created"}]}

The markdown is the truth for a claim's text, page and section; the sidecar holds
what the markdown does not show, plus the text as last saved so owner edits can
be told apart.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from repartee.schema import (
    Claim,
    ClaimOp,
    ClaimOrigin,
    ClaimStatus,
    OpResult,
    PageKind,
    slugify,
    stable_hash,
)
from repartee.selfmodel.pages import (
    CORE_KEY,
    QUESTIONS_KEY,
    Page,
    clean_text,
    new_claim_id,
    new_page,
    new_question_id,
    norm_text,
    same_text,
)

log = logging.getLogger(__name__)

SIDECAR_DIR = ".claims"
SIDECAR_VERSION = 1
_ENC = ("utf-8", "surrogateescape")


class SelfModelError(Exception):
    """The self-model on disk cannot be used as it is."""


class SelfModelConflict(SelfModelError):
    """Files changed on disk since load; nothing was written. Reload and redo the work."""


class ReconcileReport(BaseModel):
    """What load() found the owner had changed in the markdown."""

    edited: list[str] = []  # claim ids whose text the owner changed -> confirmed
    added: list[str] = []  # new ids given to owner-written list items (and copies)
    rejected: list[str] = []  # claim ids whose line the owner deleted
    moved: list[str] = []  # claim ids found on another page or section than before
    restored: list[str] = []  # previously rejected ids whose line came back


class PruneReport(BaseModel):
    """What prune_dead_evidence() dropped."""

    evidence_dropped: int = 0
    claims_removed: list[str] = []


class Question(BaseModel):
    """One line of open-questions.md."""

    id: str  # "q-" + 8 hex
    text: str
    page: str | None = None
    resolved: bool = False


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _dedupe(ids: Iterable[str]) -> list[str]:
    out: list[str] = []
    for i in ids:
        if i not in out:
            out.append(i)
    return out


class SelfModel:
    """The owner's self-model in ``self_dir``: load, query, change, save."""

    def __init__(self, self_dir: Path):
        self.dir = Path(self_dir)
        self._reset()

    def _reset(self) -> None:
        self._pages: dict[str, Page] = {}
        self._disk_pages: dict[str, bytes] = {}
        self._claims: dict[str, Claim] = {}
        self._rejected: dict[str, list[dict]] = {}
        self._disk_sidecars: dict[str, bytes] = {}
        self._sidecar_baseline: dict[str, bytes | None] = {}  # canonical form of what was read

    # ------------------------------------------------------------------ paths
    def _page_file(self, key: str) -> Path:
        return self.dir / f"{key}.md"

    def _sidecar_file(self, key: str) -> Path:
        return self.dir / SIDECAR_DIR / f"{key}.json"

    # ------------------------------------------------------------------- load
    def load(self) -> ReconcileReport:
        """Read pages and sidecars, then reconcile the owner's edits (spec 10.3)."""
        self._reset()
        if not self.dir.is_dir():
            return ReconcileReport()
        for path in sorted(self.dir.rglob("*.md")):
            rel = path.relative_to(self.dir)
            if any(part.startswith(".") for part in rel.parts) or not path.is_file():
                continue
            key = rel.with_suffix("").as_posix()
            raw = path.read_bytes()
            self._disk_pages[key] = raw
            self._pages[key] = Page.parse(key, raw.decode(*_ENC))
        known = self._read_sidecars()
        for key in self._disk_sidecars:
            claims = [c for k, c in known.values() if k == key]
            self._sidecar_baseline[key] = _dump_sidecar(claims, self._rejected.get(key, []))
        return self._reconcile(known)

    def _read_sidecars(self) -> dict[str, tuple[str, Claim]]:
        known: dict[str, tuple[str, Claim]] = {}
        root = self.dir / SIDECAR_DIR
        if not root.is_dir():
            return known
        for path in sorted(root.rglob("*.json")):
            key = path.relative_to(root).with_suffix("").as_posix()
            raw = path.read_bytes()
            where = f"{SIDECAR_DIR}/{key}.json"
            try:
                data = json.loads(raw.decode(*_ENC))
                if not isinstance(data, dict):
                    raise ValueError("not a JSON object")
                claims = data.get("claims") or {}
                rejected = data.get("rejected") or []
                if not isinstance(claims, dict) or not isinstance(rejected, list):
                    raise ValueError("'claims' must be an object and 'rejected' a list")
                if not all(isinstance(r, dict) and isinstance(r.get("text"), str)
                           for r in rejected):
                    raise ValueError("every rejected entry needs a text")
                parsed = {}
                for cid, meta in claims.items():
                    if not isinstance(meta, dict):
                        raise ValueError(f"claim {cid} is not an object")
                    fields = {"section": "", "text": "", **meta, "id": cid, "page": key}
                    parsed[cid] = Claim.model_validate(fields)
            except (ValueError, ValidationError, UnicodeError) as exc:
                raise SelfModelError(
                    f"self/{where} cannot be read ({exc}). Fix it (a git merge conflict?) "
                    f"or delete it: its claims then count as written by you."
                ) from exc
            self._disk_sidecars[key] = raw
            self._rejected[key] = rejected
            for cid, claim in parsed.items():
                if cid in known:
                    page = self._pages.get(key)
                    if not (page and page.find_claim(cid) is not None):
                        continue
                known[cid] = (key, claim)
        return known

    def _reconcile(self, known: dict[str, tuple[str, Claim]]) -> ReconcileReport:
        report = ReconcileReport()
        now = _now()
        occ: dict[str, list] = {}
        candidates = []
        for key in sorted(self._pages):
            for cl in self._pages[key].claim_lines():
                if cl.claim_id:
                    occ.setdefault(cl.claim_id, []).append((key, cl))
                else:
                    candidates.append((key, cl))
        renames: dict[str, set[str]] = {}
        new_ids: list[tuple[str, int, Claim]] = []  # (page key, line, claim) to give fresh ids

        def copy_of(base: Claim, key: str, cl) -> None:
            claim = base.model_copy(update={
                "page": key, "section": cl.section, "text": cl.text,
                "status": ClaimStatus.CONFIRMED, "origin": ClaimOrigin.USER,
                "created": now, "updated": now, "evidence": list(base.evidence),
            })
            new_ids.append((key, cl.line_no, claim))

        for cid, (skey, stored) in known.items():
            found = occ.get(cid, [])
            if not found:
                self._record_rejection(skey, stored, now)
                report.rejected.append(cid)
                continue
            same = [o for o in found if o[0] == skey]
            pick = next((o for o in same if same_text(o[1].text, stored.text)),
                        same[0] if same else found[0])
            key, cl = pick
            claim = stored.model_copy(update={"page": key, "section": cl.section, "text": cl.text})
            if stored.text and not same_text(cl.text, stored.text):
                claim.status, claim.origin, claim.updated = (
                    ClaimStatus.CONFIRMED, ClaimOrigin.USER, now)
                report.edited.append(cid)
            if key != skey or cl.section != stored.section:
                report.moved.append(cid)
            if key != skey:
                renames.setdefault(skey, set()).add(key)
            self._claims[cid] = claim
            for other in found:
                if other is not pick:
                    copy_of(claim, *other)

        for cid, found in occ.items():
            if cid in known:
                continue
            rec_key, rec = self._find_rejection(cid, [k for k, _ in found])
            pick = next((o for o in found if o[0] == rec_key), found[0])
            key, cl = pick
            if rec is not None:
                self._rejected[rec_key].remove(rec)
                claim = Claim(
                    id=cid, page=key, section=cl.section, text=cl.text,
                    status=rec.get("status", ClaimStatus.INFERRED),
                    origin=rec.get("origin", ClaimOrigin.PIPELINE),
                    confidence=rec.get("confidence", 0.5), evidence=rec.get("evidence", []),
                    valid_from=rec.get("valid_from"), valid_to=rec.get("valid_to"),
                    created=rec.get("created"), updated=now,
                )
                if not same_text(cl.text, rec["text"]):
                    claim.status, claim.origin = ClaimStatus.CONFIRMED, ClaimOrigin.USER
                    report.edited.append(cid)
                report.restored.append(cid)
            else:
                claim = Claim(id=cid, page=key, section=cl.section, text=cl.text,
                              status=ClaimStatus.CONFIRMED, origin=ClaimOrigin.USER,
                              created=now, updated=now)
                report.added.append(cid)
            self._claims[cid] = claim
            for other in found:
                if other is not pick:
                    copy_of(claim, *other)

        for key, cl in candidates:
            claim = Claim(id="", page=key, section=cl.section, text=cl.text,
                          status=ClaimStatus.CONFIRMED, origin=ClaimOrigin.USER,
                          created=now, updated=now)
            new_ids.append((key, cl.line_no, claim))

        taken = set(occ) | set(known) | self._rejected_ids()
        for key, line_no, claim in new_ids:
            cid = new_claim_id()
            while cid in taken:
                cid = new_claim_id()
            taken.add(cid)
            claim.id = cid
            self._pages[key].set_line_id(line_no, cid)
            self._claims[cid] = claim
            report.added.append(cid)

        # a page renamed by the owner keeps its rejections
        for old, targets in renames.items():
            if old not in self._pages and len(targets) == 1 and self._rejected.get(old):
                (target,) = targets
                self._rejected.setdefault(target, []).extend(self._rejected.pop(old))

        self._reconcile_questions()
        for name in ("edited", "added", "rejected", "moved", "restored"):
            if getattr(report, name):
                log.info("self-model: owner %s %s", name, getattr(report, name))
        return report

    def _reconcile_questions(self) -> None:
        page = self._pages.get(QUESTIONS_KEY)
        if not page:
            return
        seen = {q[1] for q in page.question_lines() if q[1]}
        for line_no, qid, *_ in page.question_lines():
            if qid is None:
                new = new_question_id()
                while new in seen:
                    new = new_question_id()
                seen.add(new)
                page.set_line_id(line_no, new)

    def _record_rejection(self, key: str, claim: Claim, when: datetime) -> None:
        meta = claim.model_dump(mode="json", exclude={"page", "updated"})
        meta["norm_hash"] = stable_hash(norm_text(claim.text))
        meta["rejected_at"] = when.isoformat()
        records = self._rejected.setdefault(key, [])
        if not any(r.get("id") == claim.id for r in records):
            records.append(meta)

    def _find_rejection(self, cid: str, prefer: list[str]) -> tuple[str, dict | None]:
        order = prefer + [k for k in sorted(self._rejected) if k not in prefer]
        for key in order:
            for rec in self._rejected.get(key, []):
                if rec.get("id") == cid:
                    return key, rec
        return "", None

    def _rejected_ids(self) -> set[str]:
        return {r.get("id") for recs in self._rejected.values() for r in recs if r.get("id")}

    # ------------------------------------------------------------------- save
    def _sidecar_bytes(self, key: str) -> bytes | None:
        claims = [c for c in self._claims.values() if c.page == key]
        return _dump_sidecar(claims, self._rejected.get(key, []))

    def save(self) -> None:
        """Write the pages and sidecars that changed; refuse if any changed on disk meanwhile."""
        writes: list[tuple[Path, bytes | None, bytes | None, str, str]] = []
        for key, page in self._pages.items():
            new = page.render().encode(*_ENC)
            old = self._disk_pages.get(key)
            if new != old:
                writes.append((self._page_file(key), new, old, "page", key))
        sidecar_keys = set(self._disk_sidecars) | {c.page for c in self._claims.values()}
        sidecar_keys |= {k for k, v in self._rejected.items() if v}
        for key in sorted(sidecar_keys):
            new = self._sidecar_bytes(key)
            old = self._disk_sidecars.get(key)
            if new != old and (key not in self._sidecar_baseline
                               or new != self._sidecar_baseline[key]):
                writes.append((self._sidecar_file(key), new, old, "sidecar", key))
        conflicts = []
        for path, _, old, _, _ in writes:
            current = path.read_bytes() if path.is_file() else None
            if current != old:
                conflicts.append(str(path.relative_to(self.dir)))
        if conflicts:
            raise SelfModelConflict(
                "changed on disk since the self-model was loaded, nothing written: "
                + ", ".join(conflicts)
            )
        # pages first: a crash between the two leaves claims adopted, never rejected
        for path, new, _, what, key in sorted(writes, key=lambda w: w[3] != "page"):
            if new is None:
                path.unlink(missing_ok=True)
                self._disk_sidecars.pop(key, None)
                self._sidecar_baseline.pop(key, None)
                continue
            _atomic_write(path, new)
            (self._disk_pages if what == "page" else self._disk_sidecars)[key] = new
            if what == "sidecar":
                self._sidecar_baseline[key] = new

    # ------------------------------------------------------------------ query
    def pages(self, kind: PageKind | None = None) -> list[Page]:
        """Pages, sorted by key, optionally of one kind."""
        return [self._pages[k] for k in sorted(self._pages)
                if kind is None or self._pages[k].kind == kind]

    def page(self, key: str) -> Page | None:
        """The page with this key, if any."""
        return self._pages.get(key)

    def read_page(self, key: str) -> str | None:
        """The page's current text, if it exists."""
        page = self._pages.get(key)
        return page.render() if page else None

    def resolve(self, kind: PageKind, label: str) -> str | None:
        """Page key for a subject: slug, title or alias, then a close match; None if unsure."""
        want = " ".join(str(label).split()).casefold()
        if not want:
            return None
        pages = self.pages(PageKind(kind))
        slug = slugify(label)
        tiers = [
            [p.key for p in pages if p.key.rsplit("/", 1)[-1] == slug
             or p.title.casefold() == want],
            [p.key for p in pages if any(a.casefold() == want for a in p.aliases)],
        ]
        for tier in tiers:
            if tier:
                return tier[0] if len(set(tier)) == 1 else None
        best: dict[str, float] = {}
        for p in pages:
            for name in [p.title, *p.aliases]:
                ratio = difflib.SequenceMatcher(None, want, name.casefold()).ratio()
                if ratio >= 0.88:
                    best[p.key] = max(best.get(p.key, 0.0), ratio)
        if not best:
            return None
        top = max(best.values())
        winners = [k for k, v in best.items() if v == top]
        return winners[0] if len(winners) == 1 else None

    def _ordered_claims(self) -> list[Claim]:
        out = []
        for key in sorted(self._pages):
            for cl in self._pages[key].claim_lines():
                claim = self._claims.get(cl.claim_id or "")
                if claim is not None and claim.page == key:
                    out.append(claim)
        return out

    def claims(self, page: str | None = None, status: ClaimStatus | None = None) -> list[Claim]:
        """Copies of the live claims in page and line order, optionally filtered."""
        return [c.model_copy(deep=True) for c in self._ordered_claims()
                if (page is None or c.page == page) and (status is None or c.status == status)]

    def claim(self, claim_id: str) -> Claim | None:
        """A copy of one claim, or None."""
        claim = self._claims.get(claim_id)
        return claim.model_copy(deep=True) if claim else None

    # ----------------------------------------------------------------- change
    def apply(
        self,
        ops: Sequence[ClaimOp],
        existing_evidence: Callable[[Iterable[str]], set[str]] | None = None,
    ) -> list[OpResult]:
        """Apply pipeline operations; each refusal is an OpResult with a reason, never raised."""
        from repartee.selfmodel.ops import apply_ops

        return apply_ops(self, ops, existing_evidence)

    def _owner_claim(self, claim_id: str) -> Claim:
        claim = self._claims.get(claim_id)
        if claim is None:
            raise KeyError(claim_id)
        return claim

    def confirm(self, claim_id: str) -> None:
        """The owner vouches for a claim as written."""
        claim = self._owner_claim(claim_id)
        claim.status, claim.updated = ClaimStatus.CONFIRMED, _now()

    def reject(self, claim_id: str) -> None:
        """The owner says a claim is wrong: its line goes, its text is remembered."""
        claim = self._owner_claim(claim_id)
        page = self._pages.get(claim.page)
        if page and page.find_claim(claim_id) is not None:
            page.remove_claim(claim_id)
        self._record_rejection(claim.page, claim, _now())
        del self._claims[claim_id]

    def edit(self, claim_id: str, text: str) -> None:
        """The owner rewrites a claim: new text, confirmed, origin user."""
        claim = self._owner_claim(claim_id)
        clean = clean_text(text)
        if clean is None:
            raise ValueError("a claim needs some text and no block id")
        page = self._pages.get(claim.page)
        if page and page.find_claim(claim_id) is not None:
            page.set_claim_text(claim_id, clean)
        claim.text, claim.status, claim.origin, claim.updated = (
            clean, ClaimStatus.CONFIRMED, ClaimOrigin.USER, _now())

    def remap_evidence(self, mapping: dict[str, str]) -> int:
        """Rewrite evidence ids (renamed source items); returns references rewritten."""
        count = 0
        holders = [c.evidence for c in self._claims.values()]
        holders += [r.setdefault("evidence", []) for recs in self._rejected.values() for r in recs]
        for ev in holders:
            if any(e in mapping for e in ev):
                count += sum(1 for e in ev if e in mapping)
                ev[:] = _dedupe(mapping.get(e, e) for e in ev)
        return count

    def prune_dead_evidence(self, existing: Callable[[Iterable[str]], set[str]]) -> PruneReport:
        """Drop evidence ids that no longer exist; forget inferred claims left with none."""
        report = PruneReport()
        all_ids = {e for c in self._claims.values() for e in c.evidence}
        if not all_ids:
            return report
        alive = set(existing(sorted(all_ids)))
        for claim in list(self._claims.values()):
            kept = [e for e in claim.evidence if e in alive]
            if len(kept) == len(claim.evidence):
                continue
            report.evidence_dropped += len(claim.evidence) - len(kept)
            had = bool(claim.evidence)
            claim.evidence = kept
            if had and not kept and claim.status == ClaimStatus.INFERRED:
                page = self._pages.get(claim.page)
                if page and page.find_claim(claim.id) is not None:
                    page.remove_claim(claim.id)
                del self._claims[claim.id]
                report.claims_removed.append(claim.id)
                log.info("self-model: forgot %s (%s): its evidence is gone", claim.id, claim.text)
        return report

    # -------------------------------------------------------------- questions
    def questions(self, include_resolved: bool = False) -> list[Question]:
        """Questions in open-questions.md, open ones only unless asked."""
        page = self._pages.get(QUESTIONS_KEY)
        if not page:
            return []
        return [Question(id=qid, text=text, page=pg, resolved=done)
                for _, qid, done, text, pg in page.question_lines()
                if qid and (include_resolved or not done)]

    def add_question(self, text: str, page: str | None = None) -> Question:
        """Append a question unless the same one is already there (returns that one)."""
        clean = clean_text(text)
        if clean is None:
            raise ValueError("a question needs some text and no block id")
        for q in self.questions(include_resolved=True):
            if norm_text(q.text) == norm_text(clean):
                return q
        qpage = self._pages.get(QUESTIONS_KEY)
        if qpage is None:
            qpage = new_page(PageKind.META, QUESTIONS_KEY, "Open questions")
            self._pages[QUESTIONS_KEY] = qpage
        taken = {q.id for q in self.questions(include_resolved=True)}
        qid = new_question_id()
        while qid in taken:
            qid = new_question_id()
        page_ref = clean_text(page) if page else None
        suffix = f" (page: {page_ref})" if page_ref and "(" not in page_ref else ""
        qpage.append_question(f"- [ ] {clean}{suffix} ^{qid}")
        return Question(id=qid, text=clean, page=page_ref if suffix else None)

    def resolve_question(self, question_id: str) -> None:
        """Tick a question as resolved."""
        page = self._pages.get(QUESTIONS_KEY)
        for line_no, qid, done, *_ in page.question_lines() if page else []:
            if qid == question_id:
                if not done:
                    page.tick_question(line_no)
                return
        raise KeyError(question_id)

    # ------------------------------------------------------------- core, voice
    def core_text(self) -> str:
        """Pinned plus generated core, "" when there is no core page."""
        page = self._pages.get(CORE_KEY)
        if not page:
            return ""
        parts = [page.section_text("Pinned").strip(), page.generated().strip()]
        return "\n\n".join(p for p in parts if p)

    def set_core_generated(self, markdown: str) -> None:
        """Rewrite the generated block of core.md, creating the page if needed."""
        page = self._pages.get(CORE_KEY)
        if page is None:
            page = self._pages[CORE_KEY] = new_page(PageKind.CORE, CORE_KEY, "Core")
        page.set_generated(markdown)

    def voice_registers(self) -> list[str]:
        """Register names that have a style card."""
        return [p.key.split("/", 1)[1] for p in self.pages(PageKind.VOICE)]

    def voice_card(self, register_name: str) -> str | None:
        """A style card's text (generated block and the owner's notes), no frontmatter."""
        page = self._pages.get(f"voice/{register_name}")
        return page.body().strip("\n") if page else None

    def set_voice_card(self, register_name: str, generated_markdown: str) -> None:
        """Rewrite a style card's generated block, creating the card if needed."""
        key = f"voice/{register_name}"
        page = self._pages.get(key)
        if page is None:
            page = new_page(PageKind.VOICE, register_name, register_name)  # validates the name
            self._pages[key] = page
        page.set_generated(generated_markdown)


def _dump_sidecar(claims: Iterable[Claim], rejected: list[dict]) -> bytes | None:
    body = {c.id: c.model_dump(mode="json", exclude={"id", "page"}) for c in claims}
    if not body and not rejected:
        return None
    data = {"version": SIDECAR_VERSION, "claims": body, "rejected": rejected}
    return (json.dumps(data, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode(*_ENC)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
