"""The `repartee` command line (spec section 15).

Output rules: results on stdout, progress and notes on stderr, so `ask` and `draft` can be
piped. A mistake a person can make (no home, a home in the wrong place, no API key, a
refusal) is one line on stderr and exit code 1, never a traceback.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import click

from repartee.config import resolve_home
from repartee.distill.pipeline import STAGES, DistillPlan, DistillReport, StageEstimate
from repartee.gitops import autocommit
from repartee.guard import UnsafeHome
from repartee.home import Home, HomeNotFound
from repartee.llm.base import LLMError
from repartee.mirror.agent import DraftingUnavailable, render_draft
from repartee.runtime import Runtime, open_runtime
from repartee.scaffold import HomeExists, init_home
from repartee.selfmodel.model import SelfModelConflict
from repartee.store.embed import default_model_cache

#: Errors a person can cause or fix; shown as one line, exit code 1.
PERSON_ERRORS = (HomeNotFound, UnsafeHome, HomeExists, LLMError, DraftingUnavailable,
                 SelfModelConflict)
EmbedderChoice = Literal["auto"] | None
#: Folders every home has; not worth a line in the ingest report while empty.
BUILT_IN_SOURCES = {"interviews", "conversations", "drafts"}
SHOWN_WARNINGS = 10


# --------------------------------------------------------------------- plumbing


def _one_line(text: str) -> str:
    return " ".join(part.strip() for part in str(text).splitlines() if part.strip())


def _note(text: str) -> None:
    """A note or progress line on stderr."""
    click.echo(text, err=True)


class _EchoHandler(logging.Handler):
    """Log records as plain lines on stderr (the current one, so test runners see them)."""

    def __init__(self, debug: bool):
        super().__init__(logging.DEBUG if debug else logging.WARNING)
        self.debug = debug

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage()
            if self.debug and record.exc_info:
                text += "\n" + logging.Formatter().formatException(record.exc_info)
            click.echo(f"repartee: {text}" if self.debug else f"repartee: {_one_line(text)}",
                       err=True)
        except Exception:  # noqa: BLE001 - logging must never break a command
            self.handleError(record)


def _setup_logging(debug: bool) -> None:
    logger = logging.getLogger("repartee")
    for handler in [h for h in logger.handlers if isinstance(h, _EchoHandler)]:
        logger.removeHandler(handler)
    logger.addHandler(_EchoHandler(debug))
    logger.setLevel(logging.DEBUG if debug else logging.WARNING)


class _Group(click.Group):
    """Turns the errors a person can fix into one line on stderr and exit code 1."""

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except PERSON_ERRORS as error:
            if ctx.obj and ctx.obj.get("debug"):
                raise
            message = _one_line(str(error)) or type(error).__name__
            click.echo(f"repartee: {message}", err=True)
            ctx.exit(1)


def _home_option(ctx: click.Context) -> str | None:
    return ctx.obj.get("home") if ctx.obj else None


def _attach_embedder(rt: Runtime) -> None:
    """Load the configured embedding model into the open corpus, as open_runtime's "auto"
    would; says so first when the model has to be downloaded."""
    from repartee import runtime

    config = rt.config.embeddings
    if not config.enabled:
        return
    cache = default_model_cache()
    if not (cache.is_dir() and any(cache.iterdir())):
        _note(f"Loading the embedding model {config.model}: about 220 MB, downloaded once, "
              "then kept in the cache.")
    rt.corpus.embedder = runtime.get_embedder(config, reasons=rt.embedder_notes)


def _open(ctx: click.Context, embedder: EmbedderChoice, *, need_llm: bool = False) -> Runtime:
    """Open the runtime (closed when the command ends).

    The embedding model is loaded last, so a missing home or key is reported before a
    download starts.
    """
    rt = open_runtime(_home_option(ctx), embedder=None)
    ctx.call_on_close(rt.close)
    if need_llm:
        rt.require_llm()
    if embedder == "auto":
        _attach_embedder(rt)
    return rt


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _progress() -> Callable[[str, int, int], None]:
    """Progress on stderr: rewritten in place on a terminal, one line per stage otherwise."""
    tty = sys.stderr.isatty()

    def tell(stage: str, done: int, total: int) -> None:
        line = f"{stage}: {done:,}/{total:,}"
        if tty:
            click.echo(f"\r\033[K{line}", err=True, nl=done >= total)
        elif done == 0 or done >= total:
            _note(line)

    return tell


def _usd(value: float) -> str:
    return f"${value:,.2f}" if value >= 0.01 or value == 0 else f"${value:.4f}"


# ------------------------------------------------------------------------ group


@click.group(name="repartee", cls=_Group, invoke_without_command=True)
@click.option("--home", "home", type=click.Path(file_okay=False, path_type=Path),
              envvar=None, help="The home to use (default: $REPARTEE_HOME, the user config, "
                                "or $XDG_DATA_HOME/repartee).")
@click.option("--debug", is_flag=True, help="Show log details and tracebacks.")
@click.version_option(package_name="repartee", prog_name="repartee")
@click.pass_context
def main(ctx: click.Context, home: Path | None, debug: bool) -> None:
    """Repartee: a mirror of you, learnt from your own material.

    With no command, opens the chat.
    """
    ctx.ensure_object(dict)
    ctx.obj.update(home=str(home) if home else None, debug=debug)
    _setup_logging(debug)
    if ctx.invoked_subcommand is None:
        _tui(ctx, "chat")


def _tui(ctx: click.Context, mode: str) -> None:
    rt = _open(ctx, "auto")
    from repartee.tui.app import run

    run(rt, mode)


# ------------------------------------------------------------------------- init


@main.command()
@click.argument("path", required=False, type=click.Path(file_okay=False, path_type=Path))
@click.option("--owner", default="Me", show_default=True, help="Your name as it should appear.")
@click.option("--force", is_flag=True, help="Add a home to a directory that already has files.")
@click.option("--no-git", is_flag=True, help="Do not make the home a git repository.")
@click.pass_context
def init(ctx: click.Context, path: Path | None, owner: str, force: bool, no_git: bool) -> None:
    """Create a home: folders, repartee.yaml, .gitignore, git init."""
    root = Path(path) if path else resolve_home(_home_option(ctx))
    existed = Home(root.expanduser().resolve()).exists()
    home = init_home(root, owner, force=force, git=not no_git)
    if existed:
        click.echo(f"{home.root} is already a Repartee home; added anything missing.")
        return
    click.echo(f"Created a Repartee home at {home.root}.")
    click.echo(f"Next: list your sources in {home.config_path}, then run `repartee sync`.")
    if Path(home.root) != resolve_home(None):
        click.echo(f"Point Repartee at it with --home {home.root} or "
                   f"REPARTEE_HOME={home.root}.")


# ----------------------------------------------------------------------- ingest


def _ingest(rt: Runtime) -> None:
    """Ingest without committing, remember the summary for status, and print it."""
    from repartee.status import Status, counted, save_last_ingest, search_line
    from repartee.sync import run_ingest

    _note("Reading sources...")
    report = run_ingest(rt, commit=False)
    summary = save_last_ingest(rt, report)
    stats = rt.corpus.stats()
    click.echo(f"{counted(stats['items'], 'item')} in {counted(stats['chunks'], 'chunk')} "
               f"({report.items.added:,} new, {report.items.updated:,} changed, "
               f"{report.items.removed:,} removed), {counted(report.reply_pairs, 'reply pair')}.")
    shown = [s for s in summary.sources if s.files or s.source not in BUILT_IN_SOURCES]
    width = max((len(s.source) for s in shown), default=0)
    for source in shown:
        found = (f"{counted(source.files, 'file')}, {counted(source.items, 'item')}"
                 if source.files else "no files found")
        click.echo(f"  {source.source:<{width}}  {source.path}: {found}")
    click.echo(f"{counted(report.unparsed_lines, 'unparsed line')}, "
               f"{counted(len(report.warnings), 'warning')}.")
    for warning in report.warnings[:SHOWN_WARNINGS]:
        click.echo(f"  - {warning}")
    if len(report.warnings) > SHOWN_WARNINGS:
        click.echo(f"  ... and {len(report.warnings) - SHOWN_WARNINGS:,} more; "
                   "`repartee status` keeps them.")
    if report.evidence_remapped:
        click.echo(f"{counted(report.evidence_remapped, 'evidence link')} followed moved items.")
    if report.pruned.claims_removed:
        click.echo(f"{counted(len(report.pruned.claims_removed), 'claim')} forgotten: "
                   "the material behind them is gone.")
    if summary.embedder is None:
        status = Status(home="", owner="", chunks=stats["chunks"],
                        embedded_chunks=stats["embedded_chunks"],
                        embeddings_enabled=rt.config.embeddings.enabled,
                        embedding_model=rt.config.embeddings.model,
                        embedder_error=summary.embedder_error)
        click.echo(f"Search: {search_line(status)}")


def _commit(rt: Runtime, what: str) -> None:
    autocommit(rt.home, f"repartee: {what}", enabled=rt.config.git.autocommit)


@main.command()
@click.pass_context
def ingest(ctx: click.Context) -> None:
    """Parse sources into the index."""
    rt = _open(ctx, "auto")
    _ingest(rt)
    _commit(rt, "ingest")


# ---------------------------------------------------------------------- distill


def _selected(plan: DistillPlan, stages: Sequence[str]) -> list[StageEstimate]:
    """The plan's estimates for the chosen stages (route is estimated with consolidate)."""
    return [s for s in plan.stages
            if s.stage in stages or (s.stage == "consolidate" and "route" in stages)]


def _plan_lines(plan: DistillPlan, stages: Sequence[str], batch: bool) -> list[str]:
    rows = _selected(plan, stages)
    lines = [f"Distill plan: {plan.chunks_total:,} chunks, {plan.chunks_pending:,} not yet "
             f"distilled; {plan.observations_pending:,} observations waiting to be "
             "consolidated.",
             f"  {'stage':<12}{'requests':>9}{'tokens in':>12}{'tokens out':>12}"
             f"{'est. cost':>11}"]
    for s in rows:
        lines.append(f"  {s.stage:<12}{s.requests:>9,}{s.input_tokens:>12,}"
                     f"{s.output_tokens:>12,}{_usd(s.cost_usd):>11}")
    lines.append(f"  {'total':<12}{sum(s.requests for s in rows):>9,}"
                 f"{sum(s.input_tokens for s in rows):>12,}"
                 f"{sum(s.output_tokens for s in rows):>12,}"
                 f"{_usd(sum(s.cost_usd for s in rows)):>11}")
    note = "The estimate ignores prompt caching and usually errs high."
    if batch:
        note += " Extraction goes through the Batches API at half price."
    lines.append(note)
    return lines


def _confirm_spend(rt: Runtime, cost: float, yes: bool) -> bool:
    limit = rt.config.distill.confirm_above_usd
    if yes or cost <= limit:
        return True
    return click.confirm(f"This may cost about {_usd(cost)}, above your limit of {_usd(limit)} "
                         "(distill.confirm_above_usd). Continue?", default=False, err=True)


def _report_lines(report: DistillReport) -> list[str]:
    from repartee.status import counted

    lines = [
        f"Read {counted(report.chunks_extracted, 'chunk')}: "
        f"{counted(report.observations, 'observation')} kept, "
        f"{report.observations_dropped:,} dropped.",
        f"{counted(report.pages_created, 'page')} created, "
        f"{counted(len(report.claim_ids), 'claim')} added or changed "
        f"({counted(report.ops_applied, 'change')} applied, {report.ops_refused:,} refused), "
        f"{counted(report.questions_added, 'question')} added.",
        *([f"{counted(report.facts_set, 'key fact')} set or supported."]
          if report.facts_set else []),
        *([f"{counted(len(report.retracted), 'inferred claim')} retracted as false: "
           f"{', '.join(report.retracted[:5])}{' ...' if len(report.retracted) > 5 else ''}."]
          if report.retracted else []),
        f"Spent about {_usd(report.cost_usd)} on {counted(report.requests, 'request')} "
        f"({report.input_tokens:,} tokens in, {report.output_tokens:,} out).",
    ]
    if report.claim_ids:
        lines.append("Review the new claims with `repartee review`; "
                     "`repartee why CLAIM_ID` shows what one rests on.")
    return lines


def _report_errors(report: DistillReport) -> None:
    if not report.errors:
        return
    _note(f"{len(report.errors):,} problems; the next run retries them:")
    for error in report.errors[:10]:
        _note(f"  {_one_line(error)}")
    if len(report.errors) > 10:
        _note(f"  ... and {len(report.errors) - 10:,} more.")


async def _distill(rt: Runtime, *, yes: bool, batch: bool, stages: Sequence[str],
                   **selection: Any) -> DistillReport | None:
    """Plan, ask if needed, run; prints as it goes. None when nothing ran."""
    from repartee.sync import run_distill

    plan = rt.distiller().plan(batch=batch, **selection)
    if not any(s.requests for s in _selected(plan, stages)):
        click.echo("Nothing new to distill.")
        return None
    click.echo("\n".join(_plan_lines(plan, stages, batch)))
    if not _confirm_spend(rt, sum(s.cost_usd for s in _selected(plan, stages)), yes):
        click.echo("Stopped. Nothing was spent.")
        return None
    report = await run_distill(rt, commit=False, stages=list(stages), batch=batch,
                               progress=_progress(), **selection)
    click.echo("\n".join(_report_lines(report)))
    _report_errors(report)
    return report


@main.command()
@click.option("--dry-run", is_flag=True, help="Show the plan and estimated cost; spend nothing.")
@click.option("--yes", "-y", is_flag=True, help="Do not ask, whatever the estimate.")
@click.option("--limit", type=click.IntRange(min=1), help="Extract at most this many chunks.")
@click.option("--since", type=click.DateTime(["%Y-%m-%d", "%Y-%m-%dT%H:%M"]),
              help="Only material from this date on (YYYY-MM-DD).")
@click.option("--source", help="Only this source (whatsapp, vault, tastes, interviews, ...).")
@click.option("--batch", is_flag=True,
              help="Extract through the Batches API: half price, slower.")
@click.option("--stage", "stages", multiple=True, type=click.Choice(STAGES),
              help="Run only these stages (repeatable).")
@click.pass_context
def distill(ctx: click.Context, dry_run: bool, yes: bool, limit: int | None,
            since: datetime | None, source: str | None, batch: bool,
            stages: tuple[str, ...]) -> None:
    """Extract, route, consolidate, core, gaps."""
    chosen = list(stages) or list(STAGES)
    selection = {"limit": limit, "since": since, "source": source}
    if dry_run:
        from repartee.distill import Distiller

        rt = _open(ctx, None)
        plan = Distiller(rt.home, rt.config, rt.corpus, rt.model,
                         rt.llm).plan(batch=batch, **selection)  # type: ignore[arg-type]
        click.echo("\n".join(_plan_lines(plan, chosen, batch)))
        return
    rt = _open(ctx, "auto", need_llm=True)
    report = _run(_distill(rt, yes=yes, batch=batch, stages=chosen, **selection))
    if report is not None:
        _commit(rt, "distill")


# ------------------------------------------------------------------- voice, sync


async def _voice(rt: Runtime) -> None:
    from repartee.sync import run_voice

    _note("Rebuilding voice statistics...")
    registers = await run_voice(rt, commit=False)
    if registers:
        click.echo(f"Style cards rebuilt: {', '.join(registers)}.")
    else:
        click.echo("No register has enough of your writing for a style card yet.")
    if rt.llm is None:
        click.echo(f"Statistics only, no description: {rt.llm_error}")


@main.command()
@click.pass_context
def voice(ctx: click.Context) -> None:
    """Rebuild style statistics and cards."""
    rt = _open(ctx, None)
    _run(_voice(rt))
    _commit(rt, "voice")


@main.command()
@click.option("--yes", "-y", is_flag=True, help="Do not ask before distilling, whatever the cost.")
@click.option("--batch", is_flag=True, help="Extract through the Batches API: half price, slower.")
@click.pass_context
def sync(ctx: click.Context, yes: bool, batch: bool) -> None:
    """Ingest, distill, voice, incrementally."""
    rt = _open(ctx, "auto")

    async def steps() -> None:
        _ingest(rt)
        click.echo("")
        if rt.llm is None:
            click.echo(f"Distill skipped: {rt.llm_error}")
        else:
            await _distill(rt, yes=yes, batch=batch, stages=list(STAGES))
        click.echo("")
        await _voice(rt)

    _run(steps())
    _commit(rt, "sync")


# ----------------------------------------------------------------------- status


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Counts, coverage, unconfirmed claims, open questions."""
    from repartee.status import collect_status, render_status

    rt = _open(ctx, None)
    click.echo(render_status(collect_status(rt)))


@main.command()
@click.argument("claim_id")
@click.pass_context
def why(ctx: click.Context, claim_id: str) -> None:
    """A claim (or a key fact, PAGE:SLOT) with the material behind it."""
    from repartee.mirror.review import (
        explain_claim,
        explain_fact,
        parse_fact_ref,
        render_card,
        render_fact_card,
    )

    rt = _open(ctx, None)
    wanted = claim_id.strip().lstrip("^")
    ref = parse_fact_ref(wanted) if ":" in wanted else None
    if ref is not None:
        fact_card = explain_fact(rt.model, rt.corpus, *ref)
        if fact_card is None:
            click.echo(f"repartee: no key fact {wanted} in the self-model.", err=True)
            ctx.exit(1)
        click.echo(render_fact_card(fact_card))
        return
    card = explain_claim(rt.model, rt.corpus, wanted)
    if card is None:
        click.echo(f"repartee: no claim {wanted} in the self-model.", err=True)
        ctx.exit(1)
    click.echo(render_card(card))


@main.command()
@click.option("--all", "show_all", is_flag=True, help="List confirmed facts too.")
@click.option("--confirm", "confirm", multiple=True, metavar="PAGE:SLOT",
              help="Confirm an inferred fact as it is (repeatable).")
@click.option("--reject", "reject", multiple=True, metavar="PAGE:SLOT",
              help="Reject a fact: its property goes and is not inferred again (repeatable).")
@click.pass_context
def facts(ctx: click.Context, show_all: bool, confirm: tuple[str, ...],
          reject: tuple[str, ...]) -> None:
    """Key facts (page properties): inferred ones to confirm or reject."""
    from repartee.mirror.review import fact_status, parse_fact_ref
    from repartee.schema import ClaimStatus

    rt = _open(ctx, None)
    changes = [(r, rt.model.confirm_fact) for r in confirm] + [
        (r, rt.model.reject_fact) for r in reject]
    for ref, act in changes:
        parsed = parse_fact_ref(ref)
        try:
            if parsed is None:
                raise KeyError(ref)
            act(*parsed)
        except KeyError:
            click.echo(f"repartee: no key fact {ref} in the self-model.", err=True)
            ctx.exit(1)
    if changes:
        rt.model.save()
        _commit(rt, "facts reviewed")
        click.echo(f"Saved {len(changes)} change(s).")
        return
    shown = rt.model.facts(status=None if show_all else ClaimStatus.INFERRED)
    if not shown:
        click.echo("No key facts." if show_all else "No inferred key facts to review.")
        return
    for fact in shown:
        click.echo(f"{fact.page}:{fact.slot} = {fact.shown()}  [{fact_status(fact)}]")
    if not show_all:
        click.echo("Confirm with --confirm PAGE:SLOT, reject with --reject PAGE:SLOT; "
                   "`repartee why PAGE:SLOT` shows the evidence. Editing the property in "
                   "the page confirms it, deleting it rejects it.")


# ------------------------------------------------------------------- ask, draft


def _text_argument(words: Sequence[str], what: str) -> str:
    text = " ".join(words).strip()
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if not text:
        raise click.UsageError(f"Give the {what} as an argument or on stdin.")
    return text


@main.command()
@click.argument("question", nargs=-1)
@click.option("--verbose", "-v", is_flag=True, help="Show lookups on stderr.")
@click.pass_context
def ask(ctx: click.Context, question: tuple[str, ...], verbose: bool) -> None:
    """One answer on stdout (the question may also come on stdin)."""
    from repartee.mirror.agent import DoneEvent, ToolEvent

    text = _text_argument(question, "question")
    rt = _open(ctx, "auto", need_llm=True)
    mirror = rt.mirror("chat")

    async def answer() -> str:
        final = ""
        async for event in mirror.send(text):
            if isinstance(event, ToolEvent) and verbose:
                _note(f"looked up: {event.name}: {_one_line(event.summary)}")
            elif isinstance(event, DoneEvent):
                final = event.text
        return final

    try:
        final = _run(answer())
    finally:
        mirror.end_session()
    click.echo(final)


@main.command()
@click.argument("intent", nargs=-1)
@click.option("--to", "recipient", help="Who it is for (a name or person page).")
@click.option("--register", "register_name",
              help="Register to write in, e.g. whatsapp.fr (inferred otherwise).")
@click.option("--context", help="What you are answering, e.g. their last message.")
@click.option("--variants", type=click.IntRange(1, 5), default=1, show_default=True,
              help="How many versions to write.")
@click.option("--verbose", "-v", is_flag=True, help="Show register and sources on stderr.")
@click.pass_context
def draft(ctx: click.Context, intent: tuple[str, ...], recipient: str | None,
          register_name: str | None, context: str | None, variants: int,
          verbose: bool) -> None:
    """One draft on stdout, in your voice (what to say may also come on stdin)."""
    from repartee.schema import DraftRequest

    text = _text_argument(intent, "message to draft")
    rt = _open(ctx, "auto", need_llm=True)
    drafter = rt.drafter()
    request = DraftRequest(intent=text, recipient=recipient, register_name=register_name,
                           context=context, variants=variants)
    result = _run(drafter.draft(request))
    if verbose:
        _note(f"register: {result.register_name}; {len(result.used)} sources used")
        for deviation in result.deviations:
            _note(f"style check: {deviation}")
    click.echo(render_draft(result))


# --------------------------------------------------------------- TUI modes, evals


@main.command()
@click.pass_context
def interview(ctx: click.Context) -> None:
    """Let the mirror interview you, in the TUI."""
    _tui(ctx, "interview")


@main.command()
@click.pass_context
def review(ctx: click.Context) -> None:
    """Confirm, edit or reject unconfirmed claims, in the TUI."""
    _tui(ctx, "review")


@main.group(name="eval", cls=_Group)
def eval_group() -> None:
    """Measure the mirror: blind voice test, recall, style distance."""


@eval_group.command(name="voice")
@click.pass_context
def eval_voice(ctx: click.Context) -> None:
    """The blind voice test, in the TUI."""
    _tui(ctx, "eval")


def _recall_lines(result: Any) -> list[str]:
    """A recall result as a few lines: the score, then what went wrong, question by question."""
    from repartee.status import counted

    graded = len(result.questions) - result.failed
    lines = [f"Recall {result.recall:.0%}: {result.present:,} of "
             f"{counted(result.facts, 'expected fact')} present, {result.absent:,} absent, "
             f"{result.contradicted:,} contradicted, over {counted(graded, 'question')}."]
    if result.failed:
        lines.append(f"{counted(result.failed, 'question')} could not be graded.")
    for question in result.questions:
        if question.error:
            lines.append(f"  ! {question.question}: {_one_line(question.error)}")
            continue
        for grade in question.grades:
            if grade.verdict != "present":
                lines.append(f"  {grade.verdict}: {grade.fact} ({question.question})")
    if result.path:
        lines.append(f"Saved in {result.path}.")
    return lines


def _style_lines(report: dict[str, float]) -> list[str]:
    """Style distances as a table: one register per row, the mirror next to your own texts."""
    if not report:
        return ["No style profiles yet; run `repartee voice` first."]
    rows: dict[str, dict[str, float]] = {}
    for key, value in report.items():
        register, _, who = key.rpartition(":")
        rows.setdefault(register or key, {})[who] = value
    width = max(len("register"), *(len(r) for r in rows))
    lines = [f"{'register':<{width}}  {'mirror':>7}  {'you':>7}"]
    for register in sorted(rows, key=lambda r: (r == "overall", r)):
        cells = [f"{rows[register][who]:>7.2f}" if who in rows[register] else f"{'-':>7}"
                 for who in ("mirror", "owner")]
        lines.append(f"{register:<{width}}  {'  '.join(cells)}")
    lines.append("Distance from your measured style: 0 is identical, 1 a major deviation. "
                 "'you' is your own held-out texts, the floor to compare with.")
    return lines


@eval_group.command(name="recall")
@click.argument("questions_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_context
def eval_recall(ctx: click.Context, questions_file: Path) -> None:
    """Ask the questions in a YAML file and grade the answers."""
    from repartee.evals.recall import run_recall

    rt = _open(ctx, "auto", need_llm=True)
    _note(f"Asking the questions in {questions_file}...")
    result = _run(run_recall(rt, questions_file))
    click.echo("\n".join(_recall_lines(result)))


@eval_group.command(name="style")
@click.option("-n", "n", type=click.IntRange(min=1), default=20, show_default=True,
              help="How many texts to draft and compare.")
@click.pass_context
def eval_style(ctx: click.Context, n: int) -> None:
    """Style distance between drafts and your real texts."""
    from repartee.evals.style_distance import style_distance_report

    rt = _open(ctx, "auto", need_llm=True)
    _note(f"Drafting {n} texts to compare with yours...")
    report = _run(style_distance_report(rt, n=n))
    click.echo("\n".join(_style_lines(report)))
