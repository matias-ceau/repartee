"""What `repartee status` shows: is the home healthy, and what is waiting.

Also keeps the small summary of the last ingest (``derived/last-ingest.json``) that status
reads back, since parse warnings are otherwise only seen once, when ingest runs.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from repartee.distill import Distiller
from repartee.home import Home
from repartee.runtime import Runtime
from repartee.sync import IngestReport

log = logging.getLogger(__name__)

LAST_INGEST_FILE = "last-ingest.json"
MAX_SAVED_WARNINGS = 200
SHOWN_WARNINGS = 10
SHOWN_QUESTIONS = 3


class SourceSummary(BaseModel):
    """One source's parse, as remembered for status."""

    source: str
    path: str
    files: int = 0
    items: int = 0
    unparsed_lines: int = 0
    warnings: int = 0


class LastIngest(BaseModel):
    """What the last ingest read, kept so status can show it later."""

    at: datetime
    items_added: int = 0
    items_updated: int = 0
    items_removed: int = 0
    chunks: int = 0
    unparsed_lines: int = 0
    sources: list[SourceSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)  # capped at MAX_SAVED_WARNINGS
    warnings_total: int = 0
    embedder: str | None = None  # the embedder it ran with; None means keyword only
    embedder_error: str | None = None  # why there was none, when known


def display_path(path: str | Path, home: Home) -> str:
    """A path relative to the home when it is inside it, else as given."""
    try:
        return Path(path).resolve().relative_to(home.root.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


def last_ingest_path(home: Home) -> Path:
    """Where the last ingest summary lives."""
    return home.derived_dir / LAST_INGEST_FILE


def save_last_ingest(rt: Runtime, report: IngestReport) -> LastIngest:
    """Remember the parts of an ingest report that status shows."""
    warnings = report.warnings
    reasons = [r for r in [rt.corpus.embedder_error, *rt.embedder_notes] if r]
    summary = LastIngest(
        at=datetime.now().replace(microsecond=0),
        items_added=report.items.added,
        items_updated=report.items.updated,
        items_removed=report.items.removed,
        chunks=report.chunks.added + report.chunks.updated + report.chunks.unchanged,
        unparsed_lines=report.unparsed_lines,
        sources=[SourceSummary(source=p.source, path=display_path(p.path, rt.home),
                               files=p.files, items=p.items,
                               unparsed_lines=p.unparsed_lines, warnings=len(p.warnings))
                 for p in report.parse],
        warnings=warnings[:MAX_SAVED_WARNINGS],
        warnings_total=len(warnings),
        embedder=rt.corpus.embedder_name,
        embedder_error="; ".join(dict.fromkeys(reasons)) or None,
    )
    path = last_ingest_path(rt.home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return summary


def load_last_ingest(home: Home) -> LastIngest | None:
    """The last ingest summary, None when there was none (or it cannot be read)."""
    path = last_ingest_path(home)
    try:
        return LastIngest.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, ValidationError) as error:
        log.warning("cannot read %s: %s", path, error)
        return None


class Status(BaseModel):
    """Everything status reports, gathered without loading a model or calling one."""

    home: str
    owner: str
    llm: str | None = None  # the chat model when a client is available
    llm_error: str | None = None
    items: int = 0
    chunks: int = 0
    by_source: dict[str, int] = Field(default_factory=dict)
    by_kind: dict[str, int] = Field(default_factory=dict)
    first: datetime | None = None
    last: datetime | None = None
    self_items: int = 0
    holdout_items: int = 0
    reply_pairs: int = 0
    embeddings_enabled: bool = True
    embedding_model: str = ""
    embedded_chunks: int = 0
    embedder_error: str | None = None
    pages_by_kind: dict[str, int] = Field(default_factory=dict)
    claims_by_status: dict[str, int] = Field(default_factory=dict)
    open_questions: list[str] = Field(default_factory=list)
    voice_registers: list[str] = Field(default_factory=list)
    chunks_pending: int = 0
    observations_pending: int = 0
    pending_cost_usd: float = 0.0
    hand_edits: dict[str, int] = Field(default_factory=dict)
    last_ingest: LastIngest | None = None

    @property
    def keyword_only(self) -> bool:
        """True when no chunk can be found by meaning."""
        return not self.embeddings_enabled or (self.chunks > 0 and self.embedded_chunks == 0)


def collect_status(rt: Runtime) -> Status:
    """Gather the status of the home behind rt; needs no model client and no embedder."""
    stats = rt.corpus.stats()
    last = load_last_ingest(rt.home)
    model = rt.model
    pages = Counter(str(page.kind) for page in model.pages())
    claims = Counter(str(claim.status) for claim in model.claims())
    # Planning makes no request, so it works without a client.
    plan = Distiller(rt.home, rt.config, rt.corpus, model, rt.llm).plan()  # type: ignore[arg-type]
    embedder_error = stats["embedder_error"] or "; ".join(rt.embedder_notes) or None
    if embedder_error is None and last is not None and last.embedder is None:
        embedder_error = last.embedder_error
    return Status(
        home=str(rt.home.root),
        owner=rt.config.owner.name,
        llm=rt.config.models.chat if rt.llm is not None else None,
        llm_error=rt.llm_error if rt.llm is None else None,
        items=stats["items"],
        chunks=stats["chunks"],
        by_source=stats["by_source"],
        by_kind=stats["by_kind"],
        first=stats["first"],
        last=stats["last"],
        self_items=stats["self_items"],
        holdout_items=stats["holdout_items"],
        reply_pairs=stats["reply_pairs"],
        embeddings_enabled=rt.config.embeddings.enabled,
        embedding_model=rt.config.embeddings.model,
        embedded_chunks=stats["embedded_chunks"],
        embedder_error=embedder_error,
        pages_by_kind=dict(sorted(pages.items())),
        claims_by_status=dict(sorted(claims.items())),
        open_questions=[q.text for q in model.questions()],
        voice_registers=model.voice_registers(),
        chunks_pending=plan.chunks_pending,
        observations_pending=plan.observations_pending,
        pending_cost_usd=plan.est_cost_usd,
        hand_edits={name: len(getattr(rt.reconciled, name))
                    for name in type(rt.reconciled).model_fields
                    if getattr(rt.reconciled, name)},
        last_ingest=last,
    )


# ------------------------------------------------------------------- rendering


def counted(count: int, word: str, plural: str | None = None) -> str:
    """'1 item', '2 items', '1,204 items'."""
    return f"{count:,} {word if count == 1 else (plural or word + 's')}"


def _counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{name} {count:,}" for name, count in counts.items())


def _day(when: datetime | None) -> str:
    return when.strftime("%Y-%m-%d") if when else "?"


def search_line(status: Status) -> str:
    """How the mirror finds things: by meaning and keyword, or by keyword only, and why."""
    model = status.embedding_model.rsplit("/", 1)[-1]
    if not status.embeddings_enabled:
        return "keyword search only (embeddings are turned off in repartee.yaml)"
    if status.chunks == 0:
        return f"embeddings on ({model}); nothing indexed yet"
    reason = f": {status.embedder_error}" if status.embedder_error else ""
    if status.embedded_chunks == 0:
        hint = reason or " (no chunk has an embedding yet; run `repartee ingest`)"
        return f"keyword search only{hint}"
    total = f"{status.embedded_chunks:,} of {status.chunks:,} chunks embedded ({model})"
    if status.embedded_chunks < status.chunks:
        return f"{total}; the rest are found by keyword only{reason}"
    return f"keyword and meaning, {total}"


def _row(label: str, text: str) -> str:
    return f"{label:<13}{text}"


def _ingest_lines(last: LastIngest | None) -> list[str]:
    if last is None:
        return [_row("Last ingest", "never; run `repartee ingest`")]
    changed = (f"{last.items_added:,} new, {last.items_updated:,} changed, "
               f"{last.items_removed:,} removed")
    problems = (f"{counted(last.unparsed_lines, 'unparsed line')}, "
                f"{counted(last.warnings_total, 'warning')}")
    lines = [_row("Last ingest", f"{last.at:%Y-%m-%d %H:%M}: {changed}; {problems}")]
    for source in last.sources:
        if source.unparsed_lines or source.warnings:
            lines.append(_row("", f"{source.source} {source.path}: "
                                  f"{counted(source.unparsed_lines, 'unparsed line')}, "
                                  f"{counted(source.warnings, 'warning')}"))
    lines += [_row("", f"- {warning}") for warning in last.warnings[:SHOWN_WARNINGS]]
    if last.warnings_total > SHOWN_WARNINGS:
        lines.append(_row("", f"... and {last.warnings_total - SHOWN_WARNINGS:,} more "
                              f"in derived/{LAST_INGEST_FILE}"))
    return lines


def render_status(status: Status) -> str:
    """Status as short plain lines for a terminal."""
    lines = [_row("Home", f"{status.home} ({status.owner})"),
             _row("Model", status.llm or f"none: {status.llm_error or 'no model client'}")]

    span = f", {_day(status.first)} to {_day(status.last)}" if status.items else ""
    lines.append(_row("Material", f"{counted(status.items, 'item')} in "
                                  f"{counted(status.chunks, 'chunk')}{span}"))
    if status.items:
        lines += [_row("", f"by source: {_counts(status.by_source)}"),
                  _row("", f"by kind: {_counts(status.by_kind)}"),
                  _row("", f"{status.self_items:,} written by you, "
                           f"{counted(status.reply_pairs, 'reply pair')}, "
                           f"{status.holdout_items:,} held out for evals")]
    lines.append(_row("Search", search_line(status)))

    pages = sum(status.pages_by_kind.values())
    claims = sum(status.claims_by_status.values())
    detail = f" ({_counts(status.pages_by_kind)})" if pages else ""
    lines.append(_row("Self-model", f"{counted(pages, 'page')}{detail}"))
    if claims:
        lines.append(_row("", f"{counted(claims, 'claim')}: {_counts(status.claims_by_status)}"))
    unconfirmed = status.claims_by_status.get("inferred", 0)
    if unconfirmed:
        lines.append(_row("", f"{counted(unconfirmed, 'claim')} to review: "
                              "run `repartee review`"))
    if status.hand_edits:
        lines.append(_row("", f"your edits picked up: {_counts(status.hand_edits)}"))
    questions = status.open_questions
    lines.append(_row("Questions", counted(len(questions), "open question")))
    lines += [_row("", f"- {text}") for text in questions[:SHOWN_QUESTIONS]]
    if len(questions) > SHOWN_QUESTIONS:
        lines.append(_row("", "... see self/open-questions.md"))
    registers = ", ".join(status.voice_registers) or "no style cards yet; run `repartee voice`"
    lines.append(_row("Voice", registers))

    if status.chunks_pending or status.observations_pending:
        waiting = counted(status.chunks_pending, "chunk") + " not yet distilled"
        if status.observations_pending:
            waiting += (f", {counted(status.observations_pending, 'observation')} "
                        "not yet consolidated")
        lines.append(_row("Distill", f"{waiting} (about ${status.pending_cost_usd:,.2f}); "
                                     "run `repartee distill`"))
    elif status.chunks:
        lines.append(_row("Distill", "up to date"))
    lines += _ingest_lines(status.last_ingest)
    return "\n".join(lines)
