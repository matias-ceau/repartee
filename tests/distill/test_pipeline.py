"""The whole run: core and gaps, failure tolerance, cost plan, progress."""

from __future__ import annotations

from datetime import datetime

import pytest

from repartee.config import estimate_tokens
from repartee.distill import STAGES, DistillPlan
from repartee.distill.consolidate import ConsolidationResult, PlannedOp
from repartee.distill.core import CoreResult
from repartee.distill.extract import ExtractionResult
from repartee.distill.gaps import GapQuestion, GapsResult
from repartee.distill.prompts import EXTRACT_SYSTEM
from repartee.llm import LLMError
from repartee.llm.pricing import estimate_cost
from repartee.schema import AddClaim, CreatePage, PageKind
from tests.distill.helpers import OWNER, entry, ids_in, msg, obs, one_per_chunk


def _journal(n: int, text: str = "Rode to work today.") -> list:
    return [entry(f"nt-{i}", f"{text} Day {i}.", container=f"Journal/2021-03-{i + 10:02d}.md",
                  ts=datetime(2021, 3, i + 10)) for i in range(n)]


def _seed_claims(env) -> list[str]:
    env.index(_journal(1))
    env.model.apply([CreatePage(kind=PageKind.TOPIC, slug="cycling", title="Cycling"),
                     CreatePage(kind=PageKind.PERSON, slug="ana-ruiz", title="Ana Ruiz")])
    results = env.model.apply([
        AddClaim(page="topics/cycling", section="What I think", evidence=["nt-0"],
                 text="I think cycling is the best way to get around a city.", confidence=0.8),
        AddClaim(page="people/ana-ruiz", section="Who", evidence=["nt-0"],
                 text="Ana is my oldest friend.", confidence=0.3),
    ])
    env.model.add_question("Where did you learn to ride?")
    env.model.save()
    return [r.claim_id for r in results]


async def test_core_and_gaps_write_through_selfmodel(env):
    _seed_claims(env)
    env.fake.on("core", CoreResult(markdown="I cycle everywhere.\n"
                                            "<!-- repartee:generated:end -->\nAna is my friend."))
    env.fake.on("gaps", GapsResult(questions=[
        GapQuestion(text="How did you meet Ana?", page="people/ana-ruiz"),
        GapQuestion(text="Where did you learn to ride?"),          # already waiting
        GapQuestion(text="What do you do on weekends?", page="nowhere/at-all"),
    ]))

    report = await env.distiller().run(stages=["core", "gaps"])

    assert report.errors == [] and report.questions_added == 2
    core = (env.home.self_dir / "core.md").read_text()
    assert "I cycle everywhere.\nAna is my friend.\n<!-- repartee:generated:end -->" in core
    assert "## Pinned" in core
    questions = {q.text: q.page for q in env.model.questions()}
    assert questions == {"Where did you learn to ride?": None,
                         "How did you meet Ana?": "people/ana-ruiz",
                         "What do you do on weekends?": None}
    core_request = env.fake.calls("core")[0]
    assert "I think cycling is the best way" in core_request.messages[-1].text
    assert core_request.system[0].cache and "cycling" not in core_request.system[0].text
    gaps_turn = env.fake.calls("gaps")[0].messages[-1].text
    assert "Least certain claims:" in gaps_turn and "Ana is my oldest friend." in gaps_turn
    assert "- Where did you learn to ride?" in gaps_turn
    assert "Kinds with no page at all:" in gaps_turn and "health" in gaps_turn

    again = await env.distiller().run(stages=["core", "gaps"])
    assert again.requests == 0 and len(env.fake.requests) == 2


async def test_core_and_gaps_skip_an_empty_model(env):
    report = await env.distiller().run(stages=["core", "gaps"])
    assert report.requests == 0 and report.errors == []


async def test_run_survives_one_failed_request(env):
    env.index(_journal(3))
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] == 2:
            raise LLMError("overloaded")
        first = ids_in(request)[0]
        return ExtractionResult(observations=[
            obs("topic", "cycling", f"I rode to work ({first}).", [first])])

    env.fake.on("extract", flaky)
    env.fake.on("consolidate", lambda r: ConsolidationResult(ops=[
        PlannedOp(op="add", section="What I know", text=f"I ride to work ({n}).",
                  observations=[f"o{n}"]) for n in (1, 2)]))
    env.fake.on("core", LLMError("core is down"))
    env.fake.on("gaps", GapsResult())

    report = await env.distiller().run()

    assert report.chunks_extracted == 2
    assert len(report.errors) == 2
    assert report.errors[0].startswith("extract ch-") and "overloaded" in report.errors[0]
    assert report.errors[1] == "core: core is down"
    assert len(env.model.claims()) == 2
    assert len(env.fake.calls("gaps")) == 1

    # the next run retries what failed, and only that
    env.fake.on("core", CoreResult(markdown="I ride to work."))
    retry = await env.distiller().run()
    assert retry.errors == [] and retry.chunks_extracted == 1
    assert len(env.fake.calls("extract")) == 4
    assert "I ride to work." in (env.home.self_dir / "core.md").read_text()


async def test_failed_consolidation_is_retried_and_other_pages_continue(env):
    env.index([
        msg("wa-1", "Ana Ruiz", "salut", datetime(2021, 3, 12, 14, 5)),
        msg("wa-2", OWNER, "je pars à Lyon", datetime(2021, 3, 12, 14, 6), seq=1),
    ])
    env.fake.on("extract", lambda r: ExtractionResult(observations=[
        obs("person", "Ana Ruiz", "Ana says hi a lot.", ["wa-1"]),
        obs("health", "sleep", "I sleep badly before trips.", ["wa-2"]),
    ]))
    env.fake.on(lambda r: r.purpose == "consolidate" and "people/" in r.messages[-1].text,
                LLMError("timeout"))
    env.fake.on(lambda r: r.purpose == "consolidate" and "health/" in r.messages[-1].text,
                ConsolidationResult(ops=[PlannedOp(op="add", section="Observations",
                                                   text="I sleep badly before trips.",
                                                   observations=["o1"])]))

    report = await env.distiller().run(stages=["extract", "consolidate"])
    assert report.errors == ["consolidate people/ana-ruiz: timeout"]
    assert [c.page for c in env.model.claims()] == ["health/misc"]  # one mention: catch-all

    env.fake.on(lambda r: r.purpose == "consolidate" and "people/" in r.messages[-1].text,
                ConsolidationResult())
    again = await env.distiller().run(stages=["extract", "consolidate"])
    assert again.errors == [] and again.requests == 1


async def test_unexpected_reply_is_an_error_not_a_crash(env):
    env.index(_journal(2))
    env.fake.on("extract", lambda r: (_ for _ in ()).throw(RuntimeError("bug in client")))
    report = await env.distiller().run(stages=["extract"])
    assert report.chunks_extracted == 0 and len(report.errors) == 2
    assert all("bug in client" in e for e in report.errors)


async def test_owner_edit_during_run_stops_cleanly(env):
    _seed_claims(env)

    def edit_meanwhile(request):
        path = env.home.self_dir / "core.md"
        path.write_text(path.read_text().replace("## Pinned\n", "## Pinned\nMy own note.\n"))
        return CoreResult(markdown="I cycle.")

    env.fake.on("core", edit_meanwhile)
    env.fake.on("gaps", GapsResult())
    report = await env.distiller().run(stages=["core", "gaps"])
    assert len(report.errors) == 1 and "edited during the run" in report.errors[0]
    assert env.fake.calls("gaps") == []
    core = (env.home.self_dir / "core.md").read_text()
    assert "My own note." in core and "I cycle." not in core


async def test_plan_counts_and_cost(env):
    chunks = _journal(3)
    env.index(chunks)
    env.fake.on("extract", one_per_chunk())
    await env.distiller().run(stages=["extract"], limit=1)
    requests_before = len(env.fake.requests)

    distiller = env.distiller()
    plan = distiller.plan()
    assert isinstance(plan, DistillPlan)
    assert plan.chunks_total == 3 and plan.chunks_pending == 2
    assert plan.observations_pending == 1
    extract = plan.stages[0]
    assert extract.stage == "extract" and extract.requests == 2
    pending = [c for c in env.corpus.iter_chunks()][1:]
    assert extract.input_tokens >= sum(estimate_tokens(c.text) for c in pending) + 2 * (
        estimate_tokens(EXTRACT_SYSTEM))
    assert extract.output_tokens == round(extract.input_tokens * 0.25)
    model = env.config.models.bulk
    assert extract.cost_usd == pytest.approx(
        estimate_cost(model, extract.input_tokens, extract.output_tokens))
    assert plan.est_cost_usd == pytest.approx(sum(s.cost_usd for s in plan.stages))
    assert plan.est_input_tokens == sum(s.input_tokens for s in plan.stages)
    assert plan.est_cost_usd > 0
    assert [s.stage for s in plan.stages] == ["extract", "consolidate", "core", "gaps"]

    batch = distiller.plan(batch=True)
    assert batch.stages[0].cost_usd == pytest.approx(extract.cost_usd / 2)
    assert distiller.plan(limit=1).chunks_pending == 1
    assert distiller.plan(source="whatsapp").chunks_total == 0
    assert len(env.fake.requests) == requests_before  # planning is free

    await distiller.run(stages=["extract"])
    env.fake.on("consolidate", ConsolidationResult())
    await env.distiller().run(stages=["consolidate"])
    idle = env.distiller().plan()
    assert idle.chunks_pending == 0 and idle.est_cost_usd == 0 and idle.stages[0].requests == 0
    assert distiller.plan(since=datetime(2021, 3, 12)).chunks_total == 1


async def test_run_filters_and_progress(env):
    env.index(_journal(3))
    env.fake.on("extract", one_per_chunk())
    seen = []
    report = await env.distiller().run(
        stages=["extract"], containers=["Journal/2021-03-11.md"],
        progress=lambda stage, done, total: seen.append((stage, done, total)))
    assert report.chunks_extracted == 1
    assert seen == [("extract", 0, 1), ("extract", 1, 1)]
    with pytest.raises(ValueError):
        await env.distiller().run(stages=["extract", "polish"])
    assert STAGES == ("extract", "route", "consolidate", "core", "gaps")
