"""Recall: does the mirror know what happened? Questions with expected facts, graded by a model.

The mirror answers each question in chat mode with its normal tools, from a fresh session
that is not saved among the owner's conversations. A judge then grades every expected fact
as present, absent or contradicted, with the question, the facts and the answer in front of
it. A hedged or refused answer counts as absent: not knowing is allowed, inventing is not.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from repartee.evals.common import Saver, bounded, settings
from repartee.llm.base import LLMClient, LLMError, LLMRequest, Message, SystemBlock
from repartee.runtime import Runtime

log = logging.getLogger(__name__)

Verdict = Literal["present", "absent", "contradicted"]


class RecallQuestion(BaseModel):
    """One entry of a recall file."""

    question: str
    expected: list[str]
    sources: list[str] = Field(default_factory=list)  # where the facts are, for people


class FactGrade(BaseModel):
    """One expected fact and how the answer stands with it."""

    fact: str
    verdict: Verdict
    reason: str = ""


class QuestionResult(BaseModel):
    """A question, the mirror's answer and the grade of each expected fact."""

    question: str
    expected: list[str]
    sources: list[str] = Field(default_factory=list)
    answer: str | None = None
    grades: list[FactGrade] = Field(default_factory=list)
    #: Why this question has no grades (the mirror or the judge failed); None when graded.
    error: str | None = None


class RecallResult(BaseModel):
    """Every question's result and the totals over graded facts."""

    created: datetime
    questions_file: str
    questions: list[QuestionResult]
    facts: int = 0  # graded facts (facts of failed questions are not counted)
    present: int = 0
    absent: int = 0
    contradicted: int = 0
    failed: int = 0  # questions with an error
    recall: float = 0.0  # present / facts
    path: Path | None = None  # where the result was saved

    def totals(self) -> dict[str, float | int]:
        """The counts, as saved."""
        return {"questions": len(self.questions), "failed": self.failed, "facts": self.facts,
                "present": self.present, "absent": self.absent,
                "contradicted": self.contradicted, "recall": self.recall}


# ----------------------------------------------------------------------- judging


class FactVerdict(BaseModel):
    fact: int = Field(description="The number of the expected fact.")
    verdict: Verdict
    reason: str = Field(description="One sentence.")


class JudgeOut(BaseModel):
    """What a `judge` request returns."""

    facts: list[FactVerdict]


JUDGE_SYSTEM = """\
You grade a memory test. A person's mirror, a system that answers in the first person as \
that person from their own notes and messages, was asked a question about the person's life. \
You get the question, the facts a correct answer contains, and the mirror's answer. "I" in \
the question, the facts and the answer is the same person.

Give every expected fact one verdict:
- present: the answer states the fact or a clear equivalent. Wording, language, date format \
and level of detail may differ ("June 2022" matches "in early summer 2022, in June").
- contradicted: the answer commits to something incompatible with the fact: another place, \
date, person, number, reason or opinion.
- absent: anything else. The answer does not mention the fact, only hints at it, hedges or \
guesses without committing to it, says it does not know, cannot find it or does not \
remember, or refuses to answer. A hedged, uncertain or refused answer is absent, never \
contradicted.

Extra detail beyond the expected facts is fine and changes no verdict; do not judge whether \
it is true. Grade only against the expected facts, not against what seems plausible. Give \
each fact's number, its verdict and a one-sentence reason."""


def judge_prompt(question: str, expected: list[str], answer: str) -> str:
    """The user message of a `judge` request."""
    facts = "\n".join(f"{n}. {fact}" for n, fact in enumerate(expected, 1))
    return (f"<question>\n{question}\n</question>\n\nExpected facts:\n{facts}\n\n"
            f"<answer>\n{answer.strip() or '(no answer)'}\n</answer>\n\n"
            "Grade each expected fact by its number.")


async def judge(llm: LLMClient, question: str, expected: list[str],
                answer: str) -> list[FactGrade]:
    """One grade per expected fact; LLMError when the judge leaves one out."""
    response = await llm.complete(LLMRequest(
        tier="smart", purpose="judge", system=[SystemBlock(text=JUDGE_SYSTEM, cache=True)],
        messages=[Message(role="user", text=judge_prompt(question, expected, answer))],
        output_schema=JudgeOut, max_tokens=4000))
    parsed = response.parsed
    if not isinstance(parsed, JudgeOut):
        raise LLMError("the judge's reply did not match the expected structure")
    by_number = {v.fact: v for v in parsed.facts}
    grades = []
    for number, fact in enumerate(expected, 1):
        verdict = by_number.get(number)
        if verdict is None:
            raise LLMError(f"the judge did not grade fact {number} ({fact!r})")
        grades.append(FactGrade(fact=fact, verdict=verdict.verdict, reason=verdict.reason))
    return grades


# ----------------------------------------------------------------------- running


def load_questions(path: Path) -> list[RecallQuestion]:
    """Questions from a recall YAML file; ValueError naming the entry when one is malformed."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of questions")
    out = []
    for number, entry in enumerate(data, 1):
        if isinstance(entry, dict) and isinstance(entry.get("expected"), str):
            entry = {**entry, "expected": [entry["expected"]]}
        try:
            question = RecallQuestion.model_validate(entry)
        except ValidationError as error:
            raise ValueError(f"{path}: question {number} needs a question and a list of "
                             f"expected facts ({error.error_count()} problems)") from error
        expected = [str(f).strip() for f in question.expected if str(f).strip()]
        if not question.question.strip() or not expected:
            raise ValueError(f"{path}: question {number} needs a question and at least one "
                             "expected fact")
        out.append(question.model_copy(update={"question": question.question.strip(),
                                               "expected": expected}))
    return out


async def run_recall(rt: Runtime, questions_file: Path) -> RecallResult:
    """Ask every question, grade every expected fact, save evals/recall-<timestamp>.json.

    About two model calls per question plus the mirror's lookups; a failed question is an
    entry with an error, not a crash.
    """
    questions = load_questions(questions_file)
    llm = rt.require_llm()
    saver = Saver(rt.home, "recall")

    async def one(q: RecallQuestion) -> QuestionResult:
        result = QuestionResult(question=q.question, expected=q.expected, sources=q.sources)
        try:
            # Unlogged: a logged session would later be ingested as the owner's own writing.
            mirror = rt.mirror("chat", log=False)
            result.answer = await mirror.ask(q.question)
            result.grades = await judge(llm, q.question, q.expected, result.answer)
        except Exception as error:  # one failed question is an entry, never a crash
            log.warning("recall: %r failed: %s", q.question, error)
            result.error = f"{type(error).__name__}: {error}"
        return result

    results = await bounded([lambda q=q: one(q) for q in questions])

    graded = [r for r in results if isinstance(r, QuestionResult)]
    verdicts = [g.verdict for r in graded for g in r.grades]
    facts = len(verdicts)
    present = verdicts.count("present")
    result = RecallResult(
        created=saver.created, questions_file=str(questions_file), questions=graded,
        facts=facts, present=present, absent=verdicts.count("absent"),
        contradicted=verdicts.count("contradicted"),
        failed=sum(1 for r in graded if r.error), recall=present / facts if facts else 0.0)
    result.path = saver.save({
        "eval": "recall",
        "settings": settings(rt, questions_file=str(questions_file)),
        "totals": result.totals(),
        "questions": [r.model_dump() for r in graded],
    })
    return result
