"""The distill pipeline: extract, route, consolidate, rectify, core, gaps (spec section 11).

Every stage is incremental and restartable:

- extract skips chunks whose text was already extracted with the current prompt version
  (``derived/observations/``), and writes each window of results as soon as it arrives;
- consolidate skips observations already folded in (``derived/consolidated.txt``), saving
  the self-model and the consumed ids after each batch;
- rectify checks a subject only when its confirmed key facts changed or it has claims
  not checked yet (``derived/rectify-state.json``);
- core and gaps run only when their input changed (``derived/distill-state.json``).

A failed request is recorded in ``DistillReport.errors`` and retried by the next run; it
never stops the rest of the run. Only a self-model changed on disk during the run stops it.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime

from pydantic import BaseModel, Field

from repartee.config import HomeConfig, estimate_tokens
from repartee.distill import consolidate as cons
from repartee.distill import core as core_stage
from repartee.distill import gaps as gaps_stage
from repartee.distill import rectify as rectify_stage
from repartee.distill.cost import Spend
from repartee.distill.extract import (
    ExtractionResult,
    build_request,
    clean_observations,
)
from repartee.distill.prompts import (
    CONSOLIDATE_SYSTEM,
    CORE_SYSTEM,
    EXTRACT_SYSTEM,
    EXTRACT_VERSION,
    GAPS_SYSTEM,
    RECTIFY_SYSTEM,
)
from repartee.distill.route import norm_label, route
from repartee.distill.state import (
    ConsumedSet,
    ExtractionRecord,
    Fingerprints,
    ObservationLog,
    RectifyState,
    now_iso,
)
from repartee.home import Home
from repartee.llm.base import LLMClient, LLMRequest, LLMResponse
from repartee.llm.pricing import estimate_cost
from repartee.schema import Chunk, Kind, Observation, SearchFilters
from repartee.selfmodel import SelfModel, SelfModelConflict
from repartee.store.corpus import Corpus

log = logging.getLogger(__name__)

STAGES = ("extract", "route", "consolidate", "rectify", "core", "gaps")

#: Tokens added per item id and per request around the chunk text (headers, ids).
ID_OVERHEAD_TOKENS = 8
REQUEST_OVERHEAD_TOKENS = 80
#: Plan heuristics: output relative to input, and consolidation relative to extraction.
OUTPUT_SHARE = 0.25
CONSOLIDATE_SHARE = 0.30
OBSERVATION_TOKENS = 30  # an observation line in a consolidate request, on average
CORE_OUTPUT_TOKENS = 1800
GAPS_OUTPUT_TOKENS = 600
RECTIFY_OUTPUT_TOKENS = 300
BATCH_WINDOW = 1000  # extract requests per batch submission


class StageEstimate(BaseModel):
    """Estimated requests, tokens and cost of one stage."""

    stage: str
    requests: int = 0  # answered requests (failed ones are in errors)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class DistillPlan(BaseModel):
    """What a run would do and cost, before spending anything."""

    chunks_total: int
    chunks_pending: int
    observations_pending: int = 0  # extracted earlier, not yet consolidated
    est_input_tokens: int
    est_output_tokens: int
    est_cost_usd: float
    stages: list[StageEstimate] = Field(default_factory=list)


class DistillReport(BaseModel):
    """What a run did."""

    chunks_extracted: int = 0
    observations: int = 0  # new observations kept from this run's extraction
    observations_dropped: int = 0  # unsupported, or with no page they could go to
    pages_created: int = 0
    ops_applied: int = 0
    ops_refused: int = 0
    questions_added: int = 0
    facts_set: int = 0  # key facts set or supported
    claim_ids: list[str] = []  # claims added or changed in this run
    retracted: list[str] = []  # inferred claims withdrawn as false (owner-backed)
    requests: int = 0  # answered requests (failed ones are in errors)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0  # estimated from the usage the responses reported
    errors: list[str] = []


class _Abort(Exception):
    """The self-model changed on disk; stop the run."""


Progress = Callable[[str, int, int], None]


class Distiller:
    """Runs the distill stages for one home."""

    def __init__(self, home: Home, config: HomeConfig, corpus: Corpus, model: SelfModel,
                 llm: LLMClient):
        self.home = home
        self.config = config
        self.corpus = corpus
        self.model = model
        self.llm = llm
        derived = home.derived_dir
        self.observations = ObservationLog(derived / "observations")
        self.consumed = ConsumedSet(derived / "consolidated.txt")
        self.fingerprints = Fingerprints(derived / "distill-state.json")
        self.rectify_state = RectifyState(derived / "rectify-state.json")
        self.statements = cons.owner_statements(corpus.get_items)

    # ------------------------------------------------------------------ selection

    def _filters(self, since: datetime | None, source: str | None,
                 containers: Sequence[str] | None) -> SearchFilters | None:
        if since is None and source is None and containers is None:
            return None
        return SearchFilters(sources=[source] if source else None, after=since,
                             containers=list(containers) if containers is not None else None)

    def _chunks(self, *, limit: int | None, since: datetime | None, source: str | None,
                containers: Sequence[str] | None = None) -> tuple[int, list[Chunk]]:
        total = 0
        pending: list[Chunk] = []
        for chunk in self.corpus.iter_chunks(self._filters(since, source, containers)):
            total += 1
            if limit is not None and len(pending) >= limit:
                continue
            if not self.observations.is_current(chunk.id, self.corpus.chunk_hash(chunk.id),
                                                EXTRACT_VERSION):
                pending.append(chunk)
        return total, pending

    def _current_observations(self) -> list[Observation]:
        """Observations of chunks that still exist with the text they were extracted from."""
        out = []
        for record in self.observations.records.values():
            if record.prompt_version != EXTRACT_VERSION:
                continue
            if self.corpus.chunk_hash(record.chunk_id) != record.chunk_hash:
                continue
            out.extend(record.observations)
        return out

    # ----------------------------------------------------------------------- plan

    def plan(self, *, limit: int | None = None, since: datetime | None = None,
             source: str | None = None, batch: bool = False) -> DistillPlan:
        """Chunk counts, estimated tokens and cost per stage. Makes no request.

        Estimates ignore prompt caching, so a real run usually costs less.
        """
        models = self.config.models
        total, pending = self._chunks(limit=limit, since=since, source=source)
        system = estimate_tokens(EXTRACT_SYSTEM)
        ex_in = sum(estimate_tokens(c.text) + ID_OVERHEAD_TOKENS * len(c.item_ids)
                    + REQUEST_OVERHEAD_TOKENS + system for c in pending)
        ex_out = round(ex_in * OUTPUT_SHARE)
        waiting = [o for o in self._current_observations() if o.id not in self.consumed]
        cons_in = round(ex_in * CONSOLIDATE_SHARE)
        cons_out = round(ex_out * CONSOLIDATE_SHARE)
        cons_requests = math.ceil(len(pending) / 4)
        if waiting:
            pages = len({(o.subject_kind, norm_label(o.subject)) for o in waiting})
            cons_requests += pages
            cons_in += (len(waiting) * OBSERVATION_TOKENS
                        + pages * estimate_tokens(CONSOLIDATE_SYSTEM))
            cons_out += round(len(waiting) * OBSERVATION_TOKENS * 0.5)
        stages = [
            StageEstimate(stage="extract", requests=len(pending), input_tokens=ex_in,
                          output_tokens=ex_out,
                          cost_usd=estimate_cost(models.bulk, ex_in, ex_out, batch=batch)),
            StageEstimate(stage="consolidate", requests=cons_requests if cons_in else 0,
                          input_tokens=cons_in, output_tokens=cons_out,
                          cost_usd=estimate_cost(models.smart, cons_in, cons_out)),
        ]
        changes = bool(pending or waiting)
        jobs = [j for j in rectify_stage.plan_jobs(self.model, self.rectify_state) if j.claims]
        if jobs:
            r_in = sum(estimate_tokens(RECTIFY_SYSTEM) + REQUEST_OVERHEAD_TOKENS
                       + sum(estimate_tokens(c.text) + 12 for c in j.claims) for j in jobs)
            r_out = RECTIFY_OUTPUT_TOKENS * len(jobs)
            stages.append(StageEstimate(
                stage="rectify", requests=len(jobs), input_tokens=r_in, output_tokens=r_out,
                cost_usd=estimate_cost(models.smart, r_in, r_out)))
        # new claims from this run feed core and gaps: count them in, up to the core cap
        expected = min(cons_out, core_stage.MAX_CLAIMS * 25) if changes else 0
        claims = core_stage.select_claims(self.model)
        facts = core_stage.select_facts(self.model)
        if (claims or facts or changes) and (changes or core_stage.fingerprint(claims, facts)
                                             != self.fingerprints.get("core")):
            c_in = estimate_tokens(CORE_SYSTEM) + min(
                sum(estimate_tokens(c.text) + 12 for c in claims) + expected,
                core_stage.MAX_CLAIMS * 40)
            stages.append(StageEstimate(
                stage="core", requests=1, input_tokens=c_in, output_tokens=CORE_OUTPUT_TOKENS,
                cost_usd=estimate_cost(models.smart, c_in, CORE_OUTPUT_TOKENS)))
        if (claims or changes) and (changes or gaps_stage.fingerprint(self.model)
                                    != self.fingerprints.get("gaps")):
            g_in = (estimate_tokens(GAPS_SYSTEM) + estimate_tokens(gaps_stage.overview(self.model))
                    + expected // 4)
            stages.append(StageEstimate(
                stage="gaps", requests=1, input_tokens=g_in, output_tokens=GAPS_OUTPUT_TOKENS,
                cost_usd=estimate_cost(models.smart, g_in, GAPS_OUTPUT_TOKENS)))
        return DistillPlan(
            chunks_total=total,
            chunks_pending=len(pending),
            observations_pending=len(waiting),
            est_input_tokens=sum(s.input_tokens for s in stages),
            est_output_tokens=sum(s.output_tokens for s in stages),
            est_cost_usd=round(sum(s.cost_usd for s in stages), 6),
            stages=stages,
        )

    # ------------------------------------------------------------------------ run

    async def run(self, *, stages: Sequence[str] = STAGES, limit: int | None = None,
                  since: datetime | None = None, source: str | None = None,
                  containers: Sequence[str] | None = None, batch: bool = False,
                  progress: Progress | None = None) -> DistillReport:
        """Run the chosen stages in order; failures are recorded, not raised."""
        unknown = [s for s in stages if s not in STAGES]
        if unknown:
            raise ValueError(f"unknown distill stage(s): {', '.join(unknown)}")
        report = DistillReport()
        spend = Spend(self.config.models)
        tell = progress or (lambda stage, done, total: None)
        try:
            if "extract" in stages:
                await self._guarded(report, "extract", self._extract(
                    report, spend, tell, limit=limit, since=since, source=source,
                    containers=containers, batch=batch))
            if "route" in stages or "consolidate" in stages:
                await self._guarded(report, "consolidate", self._route_and_consolidate(
                    report, spend, tell, consolidate="consolidate" in stages))
            if "rectify" in stages:
                await self._guarded(report, "rectify", self._rectify(report, spend, tell))
            if "core" in stages:
                await self._guarded(report, "core", self._core(report, spend, tell))
            if "gaps" in stages:
                await self._guarded(report, "gaps", self._gaps(report, spend, tell))
        except _Abort as stop:
            report.errors.append(str(stop))
        report.claim_ids = [c for c in dict.fromkeys(report.claim_ids)
                            if c not in set(report.retracted)]
        report.requests = spend.requests
        report.input_tokens = spend.input_tokens
        report.output_tokens = spend.output_tokens
        report.cost_usd = round(spend.cost_usd, 6)
        return report

    async def _guarded(self, report: DistillReport, stage: str, work) -> None:
        """Run one stage; a bug in it is recorded and the next stage still runs."""
        try:
            await work
        except _Abort:
            raise
        except Exception as error:  # noqa: BLE001 - one stage must not lose the run
            log.exception("distill: %s stage failed", stage)
            report.errors.append(f"{stage}: {type(error).__name__}: {error}")
        self._save()

    def _save(self) -> None:
        try:
            self.model.save()
        except SelfModelConflict as error:
            raise _Abort(f"stopped: the self-model was edited during the run ({error}). "
                         "Run distill again; finished work is kept.") from error

    async def _complete_many(self, requests: list[LLMRequest], *, batch: bool = False
                             ) -> list[LLMResponse | Exception]:
        """complete_many that turns even an unexpected exception into per-request errors."""
        try:
            return list(await self.llm.complete_many(
                requests, concurrency=self.config.distill.concurrency, batch=batch))
        except Exception as error:  # noqa: BLE001 - recorded per request by the caller
            log.exception("distill: a group of %d requests failed", len(requests))
            return [error] * len(requests)

    # -------------------------------------------------------------------- extract

    async def _extract(self, report: DistillReport, spend: Spend, tell: Progress, *,
                       limit: int | None, since: datetime | None, source: str | None,
                       containers: Sequence[str] | None, batch: bool) -> None:
        _, pending = self._chunks(limit=limit, since=since, source=source,
                                  containers=containers)
        owner = self.config.owner
        window = BATCH_WINDOW if batch else max(1, self.config.distill.concurrency) * 4
        done = 0
        tell("extract", 0, len(pending))
        for start in range(0, len(pending), window):
            jobs = []
            for chunk in pending[start: start + window]:
                items = self.corpus.get_items(chunk.item_ids)
                if not items:
                    report.errors.append(f"extract {chunk.id}: its items are not in the index")
                    continue
                jobs.append((chunk, items, build_request(chunk, items, owner)))
            responses = await self._complete_many([j[2] for j in jobs], batch=batch)
            records = []
            for (chunk, items, _), response in zip(jobs, responses, strict=True):
                if isinstance(response, Exception):
                    report.errors.append(f"extract {chunk.id}: {response}")
                    continue
                spend.add("bulk", response, batch=batch)
                result = response.parsed
                if not isinstance(result, ExtractionResult):
                    report.errors.append(f"extract {chunk.id}: reply was not an observation "
                                         f"list (stop reason {response.stop_reason})")
                    continue
                kept, dropped = clean_observations(chunk, items, result)
                report.observations += len(kept)
                report.observations_dropped += dropped
                records.append(ExtractionRecord(
                    chunk_id=chunk.id, chunk_hash=self.corpus.chunk_hash(chunk.id) or "",
                    prompt_version=EXTRACT_VERSION, at=now_iso(), observations=kept))
            self.observations.append(records)
            report.chunks_extracted += len(records)
            done += len(pending[start: start + window])
            tell("extract", done, len(pending))

    # -------------------------------------------------------- route, consolidate

    def _chat_people(self) -> set[str]:
        people: set[str] = set()
        for chunk in self.corpus.iter_chunks(SearchFilters(kinds=[Kind.MESSAGE])):
            people.update(chunk.participants)
        return people

    async def _route_and_consolidate(self, report: DistillReport, spend: Spend,
                                     tell: Progress, *, consolidate: bool) -> None:
        current = self._current_observations()
        pending = [o for o in current if o.id not in self.consumed]
        if not pending:
            return
        counts = Counter((o.subject_kind, norm_label(o.subject)) for o in current)
        tell("route", 0, len(pending))
        routing = await route(
            self.model, pending, label_counts=dict(counts), chat_people=self._chat_people(),
            owner=self.config.owner,
            min_observations=self.config.distill.min_observations_for_page,
            llm=self.llm, spend=spend, embedder=self.corpus.embedder)
        report.errors.extend(routing.errors)
        report.pages_created += len(routing.created)
        report.observations_dropped += len(routing.dropped)
        self._save()
        self.consumed.add(routing.dropped)
        tell("route", len(pending), len(pending))
        if consolidate:
            await self._consolidate(routing.pages, report, spend, tell)

    async def _consolidate(self, pages: dict[str, list[Observation]], report: DistillReport,
                           spend: Spend, tell: Progress) -> None:
        queues = {key: cons.batches(obs) for key, obs in sorted(pages.items())}
        total = sum(len(q) for q in queues.values())
        chunk_ids = {o.chunk_id for obs in pages.values() for o in obs}
        source_of = {}
        for chunk_id in chunk_ids:
            chunk = self.corpus.get_chunk(chunk_id)
            if chunk is not None:
                source_of[chunk_id] = str(chunk.kind)
        evidence = {e for obs in pages.values() for o in obs for e in o.evidence}
        item_kinds = {i.id: i.kind for i in self.corpus.get_items(evidence)}
        owner = self.config.owner
        done = 0
        tell("consolidate", 0, total)
        # Round r sends the r-th batch of every page at once; a page's batches stay in order.
        for round_no in range(max((len(q) for q in queues.values()), default=0)):
            jobs = [(key, q[round_no]) for key, q in queues.items() if round_no < len(q)]
            requests = [cons.build_request(self.model, key, b, owner, source_of)
                        for key, b in jobs]
            responses = await self._complete_many(requests)
            for (key, batch), response in zip(jobs, responses, strict=True):
                done += 1
                if isinstance(response, Exception):
                    report.errors.append(f"consolidate {key}: {response}")
                    continue
                spend.add("smart", response)
                result = response.parsed
                if not isinstance(result, cons.ConsolidationResult):
                    report.errors.append(f"consolidate {key}: reply was not a list of "
                                         f"operations (stop reason {response.stop_reason})")
                    continue
                ops, malformed = cons.to_claim_ops(result.ops, cons.refs(batch), key,
                                                   item_kinds)
                outcome = cons.apply_batch(self.model, ops, self.corpus.existing_item_ids,
                                           self.statements)
                self._count(report, outcome)
                report.ops_refused += malformed
                if outcome.refusals:
                    log.info("distill: %s refused %s", key, "; ".join(outcome.refusals))
                self._save()
                self.consumed.add(o.id for o in batch)
                tell("consolidate", done, total)

    @staticmethod
    def _count(report: DistillReport, outcome: cons.BatchOutcome) -> None:
        report.ops_applied += outcome.applied
        report.ops_refused += outcome.refused
        report.questions_added += outcome.questions
        report.facts_set += outcome.facts
        report.claim_ids.extend(outcome.claim_ids)
        report.retracted.extend(outcome.retracted)

    # -------------------------------------------------------------------- rectify

    async def _rectify(self, report: DistillReport, spend: Spend, tell: Progress) -> None:
        jobs = rectify_stage.plan_jobs(self.model, self.rectify_state)
        if not jobs:
            return
        asks = [j for j in jobs if j.claims]
        for job in jobs:
            if not job.claims:
                self.rectify_state.set(job.page, job.mark, job.candidates)
        tell("rectify", 0, len(asks))
        responses = await self._complete_many(
            [rectify_stage.build_request(self.model, j, self.config.owner) for j in asks])
        for done, (job, response) in enumerate(zip(asks, responses, strict=True), 1):
            tell("rectify", done, len(asks))
            if isinstance(response, Exception):
                report.errors.append(f"rectify {job.page}: {response}")
                continue
            spend.add("smart", response)
            result = response.parsed
            if not isinstance(result, rectify_stage.RectifyResult):
                report.errors.append(f"rectify {job.page}: reply was not a list of verdicts "
                                     f"(stop reason {response.stop_reason})")
                continue
            outcome = cons.apply_batch(self.model, rectify_stage.to_ops(job, result),
                                       self.corpus.existing_item_ids, self.statements)
            self._count(report, outcome)
            if outcome.refusals:
                log.info("distill: rectify %s refused %s", job.page, "; ".join(outcome.refusals))
            self._save()
            self.rectify_state.set(job.page, job.mark, job.candidates)
        self.rectify_state.save()

    # ------------------------------------------------------------------ core, gaps

    async def _core(self, report: DistillReport, spend: Spend, tell: Progress) -> None:
        claims = core_stage.select_claims(self.model)
        facts = core_stage.select_facts(self.model)
        if not claims and not facts:
            return
        mark = core_stage.fingerprint(claims, facts)
        if mark == self.fingerprints.get("core"):
            return
        tell("core", 0, 1)
        responses = await self._complete_many(
            [core_stage.build_request(self.model, claims, self.config.owner, facts)])
        response = responses[0]
        tell("core", 1, 1)
        if isinstance(response, Exception):
            report.errors.append(f"core: {response}")
            return
        spend.add("smart", response)
        result = response.parsed
        text = core_stage.clean_markdown(result.markdown) if isinstance(
            result, core_stage.CoreResult) else ""
        if not text:
            report.errors.append("core: the reply had no text; core.md left as it was")
            return
        self.model.set_core_generated(text)
        self._save()
        self.fingerprints.set("core", mark)

    async def _gaps(self, report: DistillReport, spend: Spend, tell: Progress) -> None:
        if not self.model.claims():
            return
        mark = gaps_stage.fingerprint(self.model)
        if mark == self.fingerprints.get("gaps"):
            return
        tell("gaps", 0, 1)
        responses = await self._complete_many(
            [gaps_stage.build_request(self.model, self.config.owner)])
        response = responses[0]
        tell("gaps", 1, 1)
        if isinstance(response, Exception):
            report.errors.append(f"gaps: {response}")
            return
        spend.add("smart", response)
        result = response.parsed
        if not isinstance(result, gaps_stage.GapsResult):
            report.errors.append("gaps: the reply had no questions")
            return
        report.questions_added += gaps_stage.add_questions(self.model, result)
        self._save()
        self.fingerprints.set("gaps", mark)
