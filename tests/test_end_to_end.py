"""The whole path on the sample persona, with a rule-based fake model and no network.

Ingest, distill, voice, a chat turn with a lookup, a draft, an interview, and `why`.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from repartee.distill.consolidate import ConsolidationResult, PlannedOp
from repartee.distill.core import CoreResult
from repartee.distill.extract import ExtractedObservation, ExtractionResult
from repartee.distill.gaps import GapQuestion, GapsResult
from repartee.distill.route import RouteResult
from repartee.ingest import parse_draft, parse_transcript
from repartee.llm import FakeLLM
from repartee.llm.base import LLMRequest
from repartee.llm.fake import tool_call
from repartee.mirror.interview import Interview
from repartee.mirror.review import explain_claim
from repartee.runtime import Runtime, open_runtime
from repartee.store.embed import HashEmbedder
from repartee.sync import distil_file, run_ingest, run_sync
from repartee.voice.prompts import DraftOut, StyleCardOut

PERSONA = Path(__file__).resolve().parents[1] / "examples" / "persona"
ITEM_LINE = re.compile(r"^\[([a-z]{2}-[0-9a-f]{12})\] (.*)$", re.MULTILINE)
OBSERVATION_LINE = re.compile(r"^- (o\d+) \(.*?\) (.*)$", re.MULTILINE)
SECTIONS_LINE = re.compile(r"^Sections you may add to: (.*)$", re.MULTILINE)


def _extract(request: LLMRequest) -> ExtractionResult:
    """One observation per chunk, citing an item the owner wrote when there is one."""
    lines = ITEM_LINE.findall(request.messages[-1].text)
    mine = [item_id for item_id, rest in lines if " me:" in f" {rest}"]
    cited = (mine or [item_id for item_id, _ in lines])[0]
    return ExtractionResult(observations=[ExtractedObservation(
        subject_kind="topic", subject="everyday life",
        text=f"I wrote something worth keeping in {cited}.", evidence=[cited], confidence=0.9,
    )])


def _consolidate(request: LLMRequest) -> ConsolidationResult:
    """Add each new observation as a claim in the page's first section."""
    text = request.messages[-1].text
    section = SECTIONS_LINE.search(text).group(1).split(", ")[0]
    return ConsolidationResult(ops=[
        PlannedOp(op="add", section=section, text=obs_text, observations=[ref])
        for ref, obs_text in OBSERVATION_LINE.findall(text)
    ])


def scripted_model() -> FakeLLM:
    fake = FakeLLM()
    fake.on("extract", _extract)
    fake.on("route", RouteResult())
    fake.on("consolidate", _consolidate)
    fake.on("core", CoreResult(markdown="I am Camille. I live in Copenhagen and work on cities."))
    fake.on("gaps", GapsResult(questions=[GapQuestion(text="Where did you grow up?")]))
    fake.on("style_card", StyleCardOut(description="Short, lowercase, dry."))
    fake.on("draft", DraftOut(variants=[["ouais ça me va", "je te dis ça demain"]]))
    fake.on("revise", DraftOut(variants=[["ouais ça me va", "je te dis ça demain"]]))
    return fake


@pytest.fixture
def rt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Runtime:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    home = tmp_path / "persona"
    shutil.copytree(PERSONA, home)
    runtime = open_runtime(home, llm=scripted_model(), embedder=HashEmbedder())
    yield runtime
    runtime.close()


async def test_sample_persona_end_to_end(rt: Runtime) -> None:
    fake: FakeLLM = rt.llm  # type: ignore[assignment]

    # Ingest: every source parses cleanly.
    ingest = run_ingest(rt)
    assert ingest.unparsed_lines == 0
    assert ingest.items.added > 1100
    assert ingest.reply_pairs > 100
    stats = rt.corpus.stats()
    assert set(stats["by_source"]) >= {"whatsapp", "vault", "tastes", "interviews"}

    # Sync: distill builds pages with evidence, a core summary and questions; voice builds cards.
    report = await run_sync(rt)
    assert report.distill is not None and not report.distill.errors
    assert report.distill.chunks_extracted == stats["chunks"]
    claims = rt.model.claims()
    assert claims and all(claim.evidence for claim in claims)
    assert "Copenhagen" in rt.model.core_text()
    assert any(q.text == "Where did you grow up?" for q in rt.model.questions())
    assert {"whatsapp.fr", "whatsapp.en"} <= set(report.registers)
    assert "Short, lowercase, dry." in (rt.model.voice_card("whatsapp.fr") or "")
    assert (rt.home.derived_dir / "observations").is_dir()

    # A second sync finds nothing new to extract.
    before = len(fake.calls("extract"))
    await run_sync(rt)
    assert len(fake.calls("extract")) == before

    # Chat: one lookup in the raw record, then an answer; the session is saved and re-readable.
    mirror = rt.mirror("chat")
    fake.add(
        [tool_call("search_memory", query="vélo", person=None, source=None, after=None,
                   before=None, author=None)],
        "I cycle everywhere, as far as I can tell from my notes.",
    )
    events = [event async for event in mirror.send("How do I get around town?")]
    assert [e.type for e in events if e.type != "text"] == ["tool", "done"]
    assert events[-1].text.startswith("I cycle everywhere")
    session = mirror.end_session()
    assert session is not None
    turns = parse_transcript(session, rt.config.owner)
    assert [t.is_self for t in turns] == [True, False]

    # Draft: in the register the owner uses with that person, recorded when accepted.
    drafting = rt.mirror("draft")
    events = [e async for e in drafting.send("@Léa dis-lui que je serai en retard ce soir")]
    draft = next(e for e in events if e.type == "draft")
    assert draft.result.register_name == "whatsapp.fr"
    assert draft.result.variants[0] == ["ouais ça me va", "je te dis ça demain"]
    recorded = drafting.accept_draft()
    assert recorded is not None
    assert [i.text for i in parse_draft(recorded, rt.config.owner)] == draft.result.variants[0]
    drafting.end_session()

    # Interview: a question, an answer, and the transcript distilled into claims on finish.
    interviewer = rt.mirror("interview")
    fake.on("interview", "Where did you grow up?")

    async def on_finish(path: Path) -> list[str]:
        return await distil_file(rt, path)

    interview = Interview(interviewer, on_finish=on_finish)
    opening = [e async for e in interview.start()]
    assert opening[-1].type == "done"
    [_ async for _ in interview.answer("In Nantes, until I was eleven. Then we moved to Lyon.")]
    outcome = await interview.finish()
    assert outcome.transcript is not None and outcome.transcript.is_file()
    assert outcome.claim_ids

    # Why: a claim comes with excerpts of the material behind it.
    card = explain_claim(rt.model, rt.corpus, outcome.claim_ids[0])
    assert card is not None and card.evidence
    assert any("Nantes" in excerpt.text for excerpt in card.evidence)


def test_runtime_without_model_still_opens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from repartee.llm.base import LLMUnavailable

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    home = tmp_path / "persona"
    shutil.copytree(PERSONA, home)
    with open_runtime(home, llm=None, embedder=None) as runtime:
        assert run_ingest(runtime).unparsed_lines == 0
        assert runtime.corpus.stats()["search"] == "keyword"
        with pytest.raises(LLMUnavailable):
            runtime.mirror()
