"""The command line on the sample persona, with a fake model and the hash embedder."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from repartee.llm import FakeLLM
from repartee.llm.base import LLMRefusal
from repartee.llm.fake import tool_call
from repartee.runtime import open_runtime
from repartee.schema import AddClaim, CreatePage, PageKind
from repartee.selfmodel import SelfModel
from tests.cli.conftest import NO_KEY, one_line_error, set_config

# ------------------------------------------------------------------------- init


def test_init_then_status_on_empty_home(cli, tmp_path: Path) -> None:
    home = tmp_path / "myhome"
    result = cli("init", str(home), "--owner", "Sam Doe")
    assert result.exit_code == 0, result.output
    assert (home / "repartee.yaml").is_file()
    assert "Sam Doe" in (home / "repartee.yaml").read_text(encoding="utf-8")
    assert "repartee.yaml" in result.stdout

    status = cli("status", home=home)
    assert status.exit_code == 0, status.output
    assert "0 items" in status.stdout
    assert "Sam Doe" in status.stdout
    assert "never" in status.stdout  # no ingest yet


def test_init_refuses_a_directory_with_files(cli, tmp_path: Path) -> None:
    target = tmp_path / "old"
    target.mkdir()
    (target / "memory.db").write_text("old data", encoding="utf-8")
    line = one_line_error(cli("init", str(target)))
    assert "already has files" in line and "--force" in line
    assert cli("init", str(target), "--force").exit_code == 0


def test_init_uses_home_option_when_no_path(cli, tmp_path: Path) -> None:
    home = tmp_path / "elsewhere"
    assert cli("init", home=home).exit_code == 0
    assert (home / "repartee.yaml").is_file()


def test_no_home_is_one_line(cli, tmp_path: Path) -> None:
    line = one_line_error(cli("status", home=tmp_path / "nothing-here"))
    assert "repartee init" in line


def test_home_inside_repo_refused_with_one_line(cli, tmp_path: Path, persona: Path) -> None:
    repo = tmp_path / "code"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    line = one_line_error(cli("init", str(repo / "home")))
    assert "Refusing" in line and str(repo) in line

    # An existing home moved inside another repository is refused too.
    inside = repo / "mine"
    persona.rename(inside)
    line = one_line_error(cli("status", home=inside))
    assert "Refusing" in line


# ----------------------------------------------------------------------- ingest


def test_ingest_persona_counts(cli, persona: Path) -> None:
    result = cli("ingest", home=persona)
    assert result.exit_code == 0, result.output
    out = result.stdout
    for source in ("whatsapp", "vault", "tastes", "interviews"):
        assert source in out
    assert "0 unparsed lines" in out
    with open_runtime(persona, llm=None, embedder=None) as rt:
        stats = rt.corpus.stats()
        chats = {i.container for i in rt.corpus.iter_items() if i.source == "whatsapp"}
    assert set(stats["by_source"]) >= {"whatsapp", "vault", "tastes", "interviews"}
    assert len(chats) == 4  # all four chats
    assert f"{stats['items']:,} items" in out
    summary = json.loads((persona / "derived" / "last-ingest.json").read_text("utf-8"))
    assert summary["unparsed_lines"] == 0
    assert summary["embedder"]


def test_status_lists_last_ingest_warnings(cli, persona: Path) -> None:
    (persona / "sources" / "whatsapp" / "broken.txt").write_text(
        "this is not a chat export\nnor is this\n", encoding="utf-8")
    assert cli("ingest", home=persona).exit_code == 0
    summary = json.loads((persona / "derived" / "last-ingest.json").read_text("utf-8"))
    assert summary["unparsed_lines"] == 2 and len(summary["warnings"]) == 2
    status = cli("status", home=persona)
    assert status.exit_code == 0
    assert "2 unparsed lines, 2 warnings" in status.stdout
    assert "sources/whatsapp: 2 unparsed lines" in status.stdout
    for warning in summary["warnings"]:
        assert warning in status.stdout


# ----------------------------------------------------------------------- no key


def _add_claim(home: Path) -> tuple[str, str]:
    """A hand-made claim with evidence from the persona; returns (claim id, item text)."""
    with open_runtime(home, llm=None, embedder=None) as rt:
        item = next(i for i in rt.corpus.iter_items() if i.is_self and len(i.text) > 20)
        model = SelfModel(rt.home.self_dir)
        model.load()
        results = model.apply([
            CreatePage(kind=PageKind.TOPIC, slug="getting-around", title="Getting around"),
            AddClaim(page="topics/getting-around", section="What I think",
                     text="I cycle everywhere.", evidence=[item.id], confidence=0.6),
        ])
        assert all(r.ok for r in results), results
        model.save()
        rt.corpus.sync_claims(model.claims())
        return results[1].claim_id, item.text


def test_commands_without_key(cli, persona: Path) -> None:
    assert cli("ingest", home=persona).exit_code == 0

    status = cli("status", home=persona)
    assert status.exit_code == 0, status.output
    assert "ANTHROPIC_API_KEY" in status.stdout

    claim_id, _ = _add_claim(persona)
    why = cli("why", claim_id, home=persona)
    assert why.exit_code == 0, why.output
    assert "I cycle everywhere." in why.stdout

    voice = cli("voice", home=persona)
    assert voice.exit_code == 0, voice.output
    assert "whatsapp.fr" in voice.stdout
    assert "statistics only" in voice.stdout.lower()

    dry = cli("distill", "--dry-run", home=persona)
    assert dry.exit_code == 0, dry.output

    for args in (("ask", "where do I live?"), ("draft", "say hi", "--to", "Léa"),
                 ("distill", "--yes"), ("eval", "style")):
        line = one_line_error(cli(*args, home=persona))
        assert NO_KEY in line, args


def test_refusal_is_one_line(cli, persona: Path, fake: FakeLLM) -> None:
    assert cli("ingest", home=persona).exit_code == 0
    fake.on("chat", LLMRefusal("The model declined to answer."))
    line = one_line_error(cli("ask", "anything", home=persona))
    assert "declined" in line


def test_no_key_fails_before_the_embedding_model_loads(cli, persona: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(config, *, reasons=None):
        raise AssertionError("loaded the embedding model for a command that cannot run")

    monkeypatch.setattr("repartee.runtime.get_embedder", forbidden)
    assert NO_KEY in one_line_error(cli("ask", "anything", home=persona))


# ---------------------------------------------------------------------- search


def test_first_download_is_announced_on_stderr(cli, persona: Path, isolated_env: Path) -> None:
    models = isolated_env / "xdg_cache_home" / "repartee" / "models"
    (models / "model").rmdir()
    result = cli("ingest", home=persona)
    assert result.exit_code == 0
    assert "220 MB" in result.stderr and "220 MB" not in result.stdout


def test_status_reports_keyword_only_when_no_embedder(cli, persona: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(config, *, reasons=None):
        if reasons is not None:
            reasons.append("cannot load embedding model X: no network")
        return None

    monkeypatch.setattr("repartee.runtime.get_embedder", unavailable)
    ingest = cli("ingest", home=persona)
    assert ingest.exit_code == 0
    assert "keyword search only" in ingest.stdout
    status = cli("status", home=persona)
    assert "keyword search only" in status.stdout
    assert "no network" in status.stdout


def test_status_reports_embeddings_turned_off(cli, persona: Path) -> None:
    set_config(persona, "embeddings:\n  enabled: false\n")
    assert cli("ingest", home=persona).exit_code == 0
    status = cli("status", home=persona)
    assert "keyword search only" in status.stdout
    assert "turned off" in status.stdout


def test_status_reports_embedded_chunks(cli, persona: Path) -> None:
    assert cli("ingest", home=persona).exit_code == 0
    with open_runtime(persona, llm=None, embedder=None) as rt:
        chunks = rt.corpus.stats()["chunks"]
    status = cli("status", home=persona)
    assert f"{chunks:,} of {chunks:,} chunks" in status.stdout
    assert "keyword search only" not in status.stdout


def test_status_does_not_load_the_embedder(cli, persona: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    assert cli("ingest", home=persona).exit_code == 0

    def forbidden(config, *, reasons=None):
        raise AssertionError("status must not load the embedding model")

    monkeypatch.setattr("repartee.runtime.get_embedder", forbidden)
    assert cli("status", home=persona).exit_code == 0
    assert cli("distill", "--dry-run", home=persona).exit_code == 0
    assert cli("voice", home=persona).exit_code == 0


# ---------------------------------------------------------------------- distill


def test_distill_dry_run_prints_plan_and_spends_nothing(cli, persona: Path,
                                                        fake: FakeLLM) -> None:
    assert cli("ingest", home=persona).exit_code == 0
    result = cli("distill", "--dry-run", home=persona)
    assert result.exit_code == 0, result.output
    for stage in ("extract", "consolidate", "core", "gaps"):
        assert stage in result.stdout
    assert "$" in result.stdout
    assert "prompt caching" in result.stdout
    assert fake.requests == []
    assert not (persona / "derived" / "observations").exists()


def test_distill_asks_above_threshold(cli, persona: Path, fake: FakeLLM) -> None:
    assert cli("ingest", home=persona).exit_code == 0
    set_config(persona, "distill:\n  confirm_above_usd: 0.0\n")

    declined = cli("distill", home=persona, input="n\n")
    assert "Continue?" in declined.stderr
    assert fake.requests == []
    assert "Nothing was spent" in declined.output

    accepted = cli("distill", "--yes", home=persona)
    assert accepted.exit_code == 0, accepted.output
    assert "Continue?" not in accepted.output
    assert fake.calls("extract")
    assert "Spent about $" in accepted.stdout
    assert "claim" in accepted.stdout


def test_distill_below_threshold_does_not_ask(cli, persona: Path, fake: FakeLLM) -> None:
    assert cli("ingest", home=persona).exit_code == 0
    set_config(persona, "distill:\n  confirm_above_usd: 100000\n")
    result = cli("distill", "--limit", "3", "--stage", "extract", home=persona)
    assert result.exit_code == 0, result.output
    assert "Continue?" not in result.output
    assert len(fake.calls("extract")) == 3
    assert not fake.calls("consolidate")


# ------------------------------------------------------------------------- sync


def test_sync_end_to_end_with_fake_llm(cli, persona: Path, fake: FakeLLM) -> None:
    result = cli("sync", "--yes", home=persona)
    assert result.exit_code == 0, result.output
    pages = [p for p in (persona / "self").rglob("*.md") if p.name != "core.md"]
    assert any(p.parent.name == "topics" for p in pages)
    assert "Copenhagen" in (persona / "self" / "core.md").read_text(encoding="utf-8")
    assert (persona / "self" / "voice" / "whatsapp.fr.md").is_file()
    assert (persona / "derived" / "last-ingest.json").is_file()
    assert "whatsapp.fr" in result.stdout

    again = cli("sync", "--yes", home=persona)
    assert again.exit_code == 0, again.output
    assert "Nothing new to distill" in again.stdout


def test_sync_without_key_ingests_and_builds_statistics(cli, persona: Path) -> None:
    result = cli("sync", home=persona)
    assert result.exit_code == 0, result.output
    assert "ANTHROPIC_API_KEY" in result.stdout
    assert (persona / "self" / "voice" / "whatsapp.fr.md").is_file()


# -------------------------------------------------------------------- ask, draft


def test_ask_and_draft_print_only_the_answer(cli, persona: Path, fake: FakeLLM) -> None:
    assert cli("ingest", home=persona).exit_code == 0
    fake.add(
        [tool_call("search_memory", query="vélo", person=None, source=None, after=None,
                   before=None, author=None)],
        "I cycle everywhere.",
    )
    ask = cli("ask", "-v", "How do I get around town?", home=persona)
    assert ask.exit_code == 0, ask.output
    assert ask.stdout == "I cycle everywhere.\n"
    assert "search_memory" in ask.stderr

    draft = cli("draft", "dis-lui que je serai en retard", "--to", "Léa", home=persona)
    assert draft.exit_code == 0, draft.output
    assert draft.stdout == "ouais ça me va\n\nje te dis ça demain\n"


def test_ask_reads_the_question_from_stdin(cli, persona: Path, fake: FakeLLM) -> None:
    assert cli("ingest", home=persona).exit_code == 0
    fake.on("chat", "Lyon, then Copenhagen.")
    result = cli("ask", home=persona, input="Where have I lived?\n")
    assert result.exit_code == 0, result.output
    assert result.stdout == "Lyon, then Copenhagen.\n"
    assert "Where have I lived?" in fake.calls("chat")[0].messages[-1].text


# -------------------------------------------------------------------------- why


def test_why_prints_claim_and_excerpts(cli, persona: Path) -> None:
    assert cli("ingest", home=persona).exit_code == 0
    claim_id, item_text = _add_claim(persona)
    result = cli("why", claim_id, home=persona)
    assert result.exit_code == 0, result.output
    assert "I cycle everywhere." in result.stdout
    assert "Evidence" in result.stdout
    assert item_text.split()[0] in result.stdout

    assert cli("why", f"^{claim_id}", home=persona).exit_code == 0
    line = one_line_error(cli("why", "c-nope", home=persona))
    assert "c-nope" in line


# -------------------------------------------------------------------- TUI, evals


def test_no_command_opens_chat(cli, persona: Path, tui_calls) -> None:
    result = cli(home=persona)
    assert result.exit_code == 0, result.output
    assert [mode for _, mode in tui_calls] == ["chat"]


@pytest.mark.parametrize(("args", "mode"), [
    (("interview",), "interview"), (("review",), "review"), (("eval", "voice"), "eval")])
def test_modes_open_the_tui(cli, persona: Path, tui_calls, args, mode) -> None:
    result = cli(*args, home=persona)
    assert result.exit_code == 0, result.output
    assert [m for _, m in tui_calls] == [mode]


def _fake_evals(monkeypatch: pytest.MonkeyPatch, recall_result, style_result) -> dict:
    seen: dict = {}
    package = types.ModuleType("repartee.evals")
    recall = types.ModuleType("repartee.evals.recall")
    style = types.ModuleType("repartee.evals.style_distance")

    async def run_recall(rt, questions_file):
        seen["recall"] = questions_file
        return recall_result

    async def style_distance_report(rt, *, n=20):
        seen["style"] = n
        return style_result

    recall.run_recall = run_recall
    style.style_distance_report = style_distance_report
    for name, module in (("repartee.evals", package), ("repartee.evals.recall", recall),
                         ("repartee.evals.style_distance", style)):
        monkeypatch.setitem(sys.modules, name, module)
    return seen


def test_eval_recall_and_style_print_reports(cli, persona: Path, fake: FakeLLM,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import datetime

    from repartee.evals.recall import FactGrade, QuestionResult, RecallResult

    result = RecallResult(
        created=datetime(2026, 11, 14), questions_file="recall.yaml",
        questions=[
            QuestionResult(question="Where do I live?", expected=["Copenhagen", "since 2021"],
                           answer="Copenhagen.",
                           grades=[FactGrade(fact="Copenhagen", verdict="present"),
                                   FactGrade(fact="since 2021", verdict="absent")]),
            QuestionResult(question="Who is Léa?", expected=["my sister"], error="timed out"),
        ],
        facts=2, present=1, absent=1, failed=1, recall=0.5,
        path=persona / "evals" / "recall-x.json")
    style = {"overall:mirror": 0.3, "overall:owner": 0.1, "whatsapp.fr:mirror": 0.35,
             "whatsapp.fr:owner": 0.12, "whatsapp.en:owner": 0.08}
    seen = _fake_evals(monkeypatch, result, style)

    recall = cli("eval", "recall", str(persona / "recall.yaml"), home=persona)
    assert recall.exit_code == 0, recall.output
    assert seen["recall"] == persona / "recall.yaml"
    out = recall.stdout
    assert out.startswith("Recall 50%: 1 of 2 expected facts present, 1 absent")
    assert "absent: since 2021 (Where do I live?)" in out
    assert "Who is Léa?: timed out" in out
    assert "recall-x.json" in out

    report = cli("eval", "style", "-n", "5", home=persona)
    assert report.exit_code == 0, report.output
    assert seen["style"] == 5
    rows = report.stdout.splitlines()
    assert rows[0].split() == ["register", "mirror", "you"]
    assert rows[1].split() == ["whatsapp.en", "-", "0.08"]
    assert rows[2].split() == ["whatsapp.fr", "0.35", "0.12"]
    assert rows[3].split() == ["overall", "0.30", "0.10"]


def test_eval_style_without_profiles(cli, persona: Path, fake: FakeLLM,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_evals(monkeypatch, None, {})
    result = cli("eval", "style", home=persona)
    assert result.exit_code == 0
    assert "repartee voice" in result.stdout


def test_lazy_imports_name_real_functions() -> None:
    import inspect

    from repartee.evals.recall import run_recall
    from repartee.evals.style_distance import style_distance_report
    from repartee.tui.app import MODES, run

    assert list(inspect.signature(run).parameters) == ["rt", "mode"]
    assert {"chat", "interview", "review", "eval"} <= set(MODES)
    assert list(inspect.signature(run_recall).parameters) == ["rt", "questions_file"]
    assert "n" in inspect.signature(style_distance_report).parameters
