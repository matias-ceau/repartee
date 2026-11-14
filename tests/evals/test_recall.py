"""Recall: the mirror answers with its tools, a scripted judge grades each expected fact."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from repartee.evals import RecallResult, run_recall
from repartee.evals.recall import JUDGE_SYSTEM, JudgeOut, load_questions
from repartee.llm import FakeLLM
from repartee.llm.base import LLMError, LLMRequest
from repartee.runtime import Runtime

QUESTIONS = """\
- question: Where did I grow up?
  expected: ["Nantes", "until I was eleven"]
- question: Who taught me to swim?
  expected: ["Mamie Jeanne"]
- question: When did I move to Bristol?
  expected: ["June 2022", "Easton"]
"""

ANSWERS = {
    "Where did I grow up?": "In Nantes, until I was eleven.",
    "Who taught me to swim?": "I don't know, I can't find anything about that.",
    "When did I move to Bristol?": "In June 2023, to a flat in Easton.",
}


def _answer(request: LLMRequest) -> str:
    for question, answer in ANSWERS.items():
        if question in request.messages[0].text:
            return answer
    raise AssertionError("unexpected question")


def _judge(request: LLMRequest) -> JudgeOut:
    text = request.messages[-1].text
    facts = re.findall(r"^(\d+)\. (.*)$", text, re.MULTILINE)
    answer = re.search(r"<answer>\n(.*?)\n</answer>", text, re.DOTALL).group(1)
    out = []
    for number, fact in facts:
        if "don't know" in answer:
            verdict = "absent"
        elif fact == "June 2022":
            verdict = "contradicted"
        else:
            verdict = "present" if fact.split()[0] in answer else "absent"
        out.append({"fact": int(number), "verdict": verdict, "reason": "scripted"})
    return JudgeOut.model_validate({"facts": out})


@pytest.fixture
def questions(tmp_path: Path) -> Path:
    path = tmp_path / "recall.yaml"
    path.write_text(QUESTIONS, encoding="utf-8")
    return path


@pytest.fixture
def scripted(fake: FakeLLM) -> FakeLLM:
    fake.on("chat", _answer)
    fake.on("judge", _judge)
    return fake


async def test_recall_totals_with_a_scripted_judge(rt: Runtime, scripted: FakeLLM,
                                                   questions: Path) -> None:
    result = await run_recall(rt, questions)
    assert isinstance(result, RecallResult)
    assert [q.question for q in result.questions] == list(ANSWERS)
    grown, swim, bristol = result.questions
    assert grown.answer == ANSWERS["Where did I grow up?"]
    assert [g.verdict for g in grown.grades] == ["present", "present"]
    assert [g.verdict for g in swim.grades] == ["absent"]  # a refusal is absent, not wrong
    assert [g.verdict for g in bristol.grades] == ["contradicted", "present"]
    assert (result.facts, result.present, result.absent, result.contradicted) == (5, 3, 1, 1)
    assert result.failed == 0
    assert result.recall == 3 / 5
    assert grown.grades[0].fact == "Nantes"


async def test_judge_request_holds_everything_and_the_rules(rt: Runtime, scripted: FakeLLM,
                                                            questions: Path) -> None:
    await run_recall(rt, questions)
    judged = scripted.calls("judge")
    assert len(judged) == 3
    for request in judged:
        assert request.tier == "smart" and request.output_schema is JudgeOut
        text = request.messages[-1].text
        assert "<question>" in text and "<answer>" in text and "1. " in text
    rules = JUDGE_SYSTEM.lower()
    assert "absent" in rules and "contradicted" in rules
    assert "hedge" in rules and "refus" in rules
    assert "extra" in rules


async def test_mirror_uses_its_tools_and_writes_no_session(rt: Runtime, scripted: FakeLLM,
                                                           questions: Path) -> None:
    before = sorted(rt.home.conversations_dir.rglob("*"))
    await run_recall(rt, questions)
    chats = scripted.calls("chat")
    assert chats and all(r.tools for r in chats)
    assert sorted(rt.home.conversations_dir.rglob("*")) == before


async def test_a_failure_is_an_entry_not_a_crash(rt: Runtime, scripted: FakeLLM,
                                                 questions: Path) -> None:
    def judge(request: LLMRequest) -> JudgeOut:
        if "Mamie" in request.messages[-1].text:
            raise LLMError("judge overloaded")
        return _judge(request)

    scripted.on("judge", judge)
    result = await run_recall(rt, questions)
    swim = result.questions[1]
    assert swim.error and "judge overloaded" in swim.error
    assert swim.grades == []
    assert result.failed == 1
    assert result.facts == 4  # graded facts only
    assert result.recall == 3 / 4


async def test_a_judge_that_skips_a_fact_fails_the_question(rt: Runtime, scripted: FakeLLM,
                                                            questions: Path) -> None:
    scripted.on("judge", JudgeOut.model_validate({"facts": [
        {"fact": 1, "verdict": "present", "reason": "x"}]}))
    result = await run_recall(rt, questions)
    assert result.questions[1].error is None  # one fact, one grade
    assert result.questions[0].error and "fact 2" in result.questions[0].error


async def test_saved_file_shape(rt: Runtime, scripted: FakeLLM, questions: Path) -> None:
    result = await run_recall(rt, questions)
    assert result.path is not None
    path = Path(result.path)
    assert path.parent == rt.home.evals_dir and path.name.startswith("recall-")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["eval"] == "recall"
    assert data["settings"]["questions_file"] == str(questions)
    assert data["totals"] == {"questions": 3, "failed": 0, "facts": 5, "present": 3,
                              "absent": 1, "contradicted": 1, "recall": 0.6}
    assert data["questions"][2]["grades"][0] == {
        "fact": "June 2022", "verdict": "contradicted", "reason": "scripted"}


def test_load_questions_reads_the_persona_file() -> None:
    persona = Path(__file__).resolve().parents[2] / "examples" / "persona" / "recall.yaml"
    questions = load_questions(persona)
    assert len(questions) >= 10
    assert all(q.question and q.expected for q in questions)


def test_load_questions_rejects_a_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- question: no facts here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected"):
        load_questions(bad)
