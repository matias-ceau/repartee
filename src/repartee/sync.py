"""The sequences that keep a home up to date: ingest, distill, voice.

Each step is incremental. Nothing here talks to a person; the CLI and the TUI do that.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from repartee.distill.pipeline import DistillReport
from repartee.gitops import autocommit
from repartee.ingest import ingest_home
from repartee.ingest.base import ParseReport
from repartee.ingest.chunking import chunk_items
from repartee.runtime import Runtime
from repartee.selfmodel.model import PruneReport
from repartee.store.corpus import SyncStats
from repartee.voice.pairs import mine_reply_pairs

Progress = Callable[[str, int, int], None]


class IngestReport(BaseModel):
    """What one ingest run read and what it changed in the index."""

    parse: list[ParseReport] = Field(default_factory=list)
    items: SyncStats = Field(default_factory=SyncStats)
    chunks: SyncStats = Field(default_factory=SyncStats)
    reply_pairs: int = 0
    evidence_remapped: int = 0
    pruned: PruneReport = Field(default_factory=PruneReport)

    @property
    def unparsed_lines(self) -> int:
        return sum(report.unparsed_lines for report in self.parse)

    @property
    def warnings(self) -> list[str]:
        return [warning for report in self.parse for warning in report.warnings]


class SyncReport(BaseModel):
    ingest: IngestReport
    distill: DistillReport | None = None
    registers: list[str] = Field(default_factory=list)  # style cards written


def run_ingest(rt: Runtime, *, commit: bool = True) -> IngestReport:
    """Parse every source into the index and repair the self-model's evidence links."""
    items, parse_reports = ingest_home(rt.home, rt.config)
    chunks = chunk_items(items, rt.config.chunking)
    item_stats = rt.corpus.sync_items(items)
    chunk_stats = rt.corpus.sync_chunks(chunks)
    pairs = mine_reply_pairs(items)
    rt.corpus.sync_reply_pairs(pairs)

    remapped = rt.model.remap_evidence(item_stats.remapped) if item_stats.remapped else 0
    pruned = rt.model.prune_dead_evidence(rt.corpus.existing_item_ids)
    rt.model.save()
    rt.corpus.sync_claims(rt.model.claims())

    report = IngestReport(
        parse=parse_reports, items=item_stats, chunks=chunk_stats, reply_pairs=len(pairs),
        evidence_remapped=remapped, pruned=pruned,
    )
    if commit:
        _commit(rt, "ingest")
    return report


async def run_distill(rt: Runtime, *, commit: bool = True, **options: object) -> DistillReport:
    """Run the distill pipeline (options go to Distiller.run) and re-index the claims."""
    report = await rt.distiller().run(**options)  # type: ignore[arg-type]
    rt.model.save()
    rt.corpus.sync_claims(rt.model.claims())
    if commit:
        _commit(rt, "distill")
    return report


async def run_voice(rt: Runtime, *, commit: bool = True) -> list[str]:
    """Rebuild style statistics and cards; returns the registers written."""
    registers = await rt.voice_builder().build()
    if commit:
        _commit(rt, "voice")
    return registers


async def run_sync(rt: Runtime, *, batch: bool = False,
                   progress: Progress | None = None) -> SyncReport:
    """Ingest, then distill and rebuild the voice. Without a model client: ingest and statistics."""
    ingest = run_ingest(rt, commit=False)
    distill = None
    if rt.llm is not None:
        distill = await run_distill(rt, commit=False, batch=batch, progress=progress)
    registers = await run_voice(rt, commit=False)
    _commit(rt, "sync")
    return SyncReport(ingest=ingest, distill=distill, registers=registers)


async def distil_file(rt: Runtime, path: Path) -> list[str]:
    """Ingest and distil one new transcript; returns the ids of the claims it produced.

    This is the hook an interview calls when it ends. Ingest runs in a thread so the
    interface stays responsive while it parses.
    """
    await asyncio.to_thread(run_ingest, rt, commit=False)
    container = _container_of(rt, path)
    report = await run_distill(rt, commit=False, containers=[container])
    _commit(rt, f"interview {path.stem}")
    return report.claim_ids


def _container_of(rt: Runtime, path: Path) -> str:
    """The container name ingest gives a transcript: its path inside its folder."""
    resolved = Path(path).resolve()
    for folder in (rt.home.interviews_dir, rt.home.conversations_dir, rt.home.drafts_dir):
        try:
            return resolved.relative_to(folder.resolve()).as_posix()
        except ValueError:
            continue
    return resolved.name


def _commit(rt: Runtime, what: str) -> None:
    autocommit(rt.home, f"repartee: {what}", enabled=rt.config.git.autocommit)
