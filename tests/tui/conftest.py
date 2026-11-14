"""Fixtures for TUI tests: a temp copy of the fictional sample persona, a fake model, no network."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from repartee.distill.consolidate import ConsolidationResult, PlannedOp
from repartee.distill.core import CoreResult
from repartee.distill.extract import ExtractedObservation, ExtractionResult
from repartee.distill.gaps import GapsResult
from repartee.distill.route import RouteResult
from repartee.llm import FakeLLM
from repartee.llm.base import LLMRequest
from repartee.runtime import Runtime, open_runtime
from repartee.schema import AddClaim, CreatePage, PageKind
from repartee.store.embed import HashEmbedder
from repartee.sync import run_ingest
from repartee.voice.prompts import DraftOut

PERSONA = Path(__file__).resolve().parents[2] / "examples" / "persona"
ITEM_LINE = re.compile(r"^\[([a-z]{2}-[0-9a-f]{12})\] (.*)$", re.MULTILINE)
OBSERVATION_LINE = re.compile(r"^- (o\d+) \(.*?\) (.*)$", re.MULTILINE)
SECTIONS_LINE = re.compile(r"^Sections you may add to: (.*)$", re.MULTILINE)
BURST = ["ouais pas de souci", "je te dis ça demain"]


def _extract(request: LLMRequest) -> ExtractionResult:
    """One observation per chunk, citing the first item the owner wrote."""
    lines = ITEM_LINE.findall(request.messages[-1].text)
    mine = [item_id for item_id, rest in lines if " me:" in f" {rest}"]
    cited = (mine or [item_id for item_id, _ in lines])[0]
    return ExtractionResult(observations=[ExtractedObservation(
        subject_kind="topic", subject="childhood",
        text=f"I said something about my childhood in {cited}.", evidence=[cited],
        confidence=0.9,
    )])


def _consolidate(request: LLMRequest) -> ConsolidationResult:
    text = request.messages[-1].text
    section = SECTIONS_LINE.search(text).group(1).split(", ")[0]
    return ConsolidationResult(ops=[
        PlannedOp(op="add", section=section, text=obs_text, observations=[ref])
        for ref, obs_text in OBSERVATION_LINE.findall(text)
    ])


def scripted() -> FakeLLM:
    """A fake model that can draft and distil; chat replies are added per test."""
    fake = FakeLLM()
    fake.on("extract", _extract)
    fake.on("route", RouteResult())
    fake.on("consolidate", _consolidate)
    fake.on("core", CoreResult(markdown="I am Camille."))
    fake.on("gaps", GapsResult(questions=[]))
    fake.on("draft", DraftOut(variants=[BURST]))
    fake.on("revise", DraftOut(variants=[BURST]))
    return fake


def _open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm) -> Runtime:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    home = tmp_path / "persona"
    shutil.copytree(PERSONA, home)
    runtime = open_runtime(home, llm=llm, embedder=HashEmbedder())
    run_ingest(runtime)
    return runtime


@pytest.fixture
def fake() -> FakeLLM:
    return scripted()


@pytest.fixture
def rt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeLLM) -> Runtime:
    runtime = _open(tmp_path, monkeypatch, fake)
    yield runtime
    runtime.close()


@pytest.fixture
def rt_no_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Runtime:
    runtime = _open(tmp_path, monkeypatch, None)
    yield runtime
    runtime.close()


def add_claims(rt: Runtime) -> list[str]:
    """Three inferred claims backed by real persona items, lowest confidence first."""
    items = [item for item in rt.corpus.iter_items() if item.is_self][:3]
    results = rt.model.apply([
        CreatePage(kind=PageKind.TOPIC, slug="getting-around", title="Getting around"),
        *[AddClaim(page="topics/getting-around", section="What I think",
                   text=f"Claim number {n} about getting around.", evidence=[item.id],
                   confidence=0.2 + n / 10)
          for n, item in enumerate(items)],
    ])
    assert all(r.ok for r in results), results
    rt.model.save()
    rt.corpus.sync_claims(rt.model.claims())
    return [r.claim_id for r in results[1:]]
