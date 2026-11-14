"""Interviews: the mirror asks the owner questions to fill gaps in the self-model.

Pacing is kept in code, the wording is the model's. Questions come from
``open-questions.md`` (those matching the topic, when one is given), one at a time; the
model may follow up at most twice on a question, then the runner moves on by itself. The
model moves on earlier by calling ``next_question``. ``/skip`` moves on at once. When the
list is empty, the model is shown where the self-model is thin and picks a question.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from repartee.llm.base import ToolSpec
from repartee.mirror import prompts
from repartee.mirror.agent import Mirror, MirrorEvent
from repartee.mirror.context import clip
from repartee.mirror.tools import ToolOutcome
from repartee.schema import PAGE_DIRS, PageKind
from repartee.selfmodel import Question, SelfModel

MAX_FOLLOW_UPS = 2
SKIP = "/skip"
THIN_CLAIMS = 2  # a page with this many live claims or fewer is thin

NEXT_QUESTION_SPEC = ToolSpec(
    name="next_question",
    description=("Move on from the current question: returns the next question on the "
                 "interview list, or, when the list is empty, where the owner's record is thin."),
    input_schema={"type": "object", "properties": {}, "required": [],
                  "additionalProperties": False},
)


class InterviewOutcome(BaseModel):
    """What an interview produced."""

    transcript: Path | None
    claim_ids: list[str] = []  # claims distilled from it, for the closing review
    answered: list[str] = []  # ids of the open questions it resolved


def coverage_hint(model: SelfModel, topic: str | None = None, limit: int = 8) -> str:
    """Where the self-model is thin: pages with few claims, kinds with no pages."""
    thin, empty = [], []
    for kind, directory in PAGE_DIRS.items():
        if kind == PageKind.VOICE:
            continue
        pages = model.pages(kind)
        if not pages:
            empty.append(directory)
        for page in pages:
            count = len(model.claims(page=page.key))
            if count <= THIN_CLAIMS:
                thin.append((count, page.key))
    parts = []
    if thin:
        listed = ", ".join(f"{key} ({n} claims)" for n, key in sorted(thin)[:limit])
        parts.append(f"Thin pages: {listed}.")
    if empty:
        parts.append(f"Nothing at all yet about: {', '.join(empty)}.")
    if not parts:
        parts.append("Every page has some material; look for what is missing within them.")
    if topic:
        parts.append(f"Stay with the topic: {topic}.")
    return " ".join(parts)


class Interview:
    """Runs one interview through a Mirror in interview mode."""

    def __init__(self, mirror: Mirror, *, topic: str | None = None,
                 on_finish: Callable[[Path], Awaitable[list[str]]] | None = None):
        self.mirror = mirror
        self.topic = topic.strip() if topic and topic.strip() else None
        self.on_finish = on_finish
        self.current: Question | None = None
        self._asked: list[str] = []
        self._answered: list[str] = []
        self._follow_ups = 0
        self._moved = False

    # ------------------------------------------------------------- questions

    def _agenda(self) -> list[Question]:
        open_ = [q for q in self.mirror.model.questions() if q.id not in self._asked]
        if self.topic:
            want = self.topic.casefold()
            open_ = [q for q in open_ if want in q.text.casefold()
                     or want in (q.page or "").casefold()]
        return open_

    def _take_next(self) -> Question | None:
        agenda = self._agenda()
        self.current = agenda[0] if agenda else None
        if self.current is not None:
            self._asked.append(self.current.id)
        self._moved = True
        return self.current

    def _coverage(self) -> str:
        return coverage_hint(self.mirror.model, self.topic)

    def _next_question_tool(self, _arguments: dict[str, Any]) -> ToolOutcome:
        question = self._take_next()
        if question is None:
            return ToolOutcome(
                content=prompts.NEXT_QUESTION_NONE.format(coverage=self._coverage()),
                summary="interview: no questions left on the list")
        return ToolOutcome(content=prompts.NEXT_QUESTION.format(question=question.text),
                           refs=[question.id],
                           summary=f"next question: {clip(question.text, 60)}")

    def _move_on(self, why: str) -> str:
        question = self._take_next()
        if question is None:
            return prompts.INTERVIEW_MOVE_ON_FREE.format(why=why, coverage=self._coverage())
        return prompts.INTERVIEW_MOVE_ON.format(question=question.text, why=why)

    # ------------------------------------------------------------- the flow

    async def _run(self, prompt: str, log_text: str | None) -> AsyncIterator[MirrorEvent]:
        async for event in self.mirror.turn(prompt, log_text=log_text):
            yield event
        # The reply either asked a new question (the thread moved) or followed up.
        self._follow_ups = 0 if self._moved else self._follow_ups + 1
        self._moved = False

    async def start(self) -> AsyncIterator[MirrorEvent]:
        """Open the interview: the mirror asks the first question."""
        mirror = self.mirror
        mirror.set_mode("interview")
        mirror.end_session()  # a fresh transcript even if the mirror was already interviewing
        mirror.session_title = self.topic or "Interview"
        mirror.extra_tools["next_question"] = (NEXT_QUESTION_SPEC, self._next_question_tool)
        self._moved = False
        question = self._take_next()
        topic = prompts.INTERVIEW_TOPIC.format(owner=mirror.owner, topic=self.topic) \
            if self.topic else ""
        first = (prompts.INTERVIEW_FIRST.format(question=question.text) if question
                 else prompts.INTERVIEW_NO_LIST.format(coverage=self._coverage()))
        async for event in self._run(prompts.INTERVIEW_OPENING.format(topic=topic, first=first),
                                     None):
            yield event

    async def answer(self, text: str) -> AsyncIterator[MirrorEvent]:
        """The owner's answer (or /skip); the mirror follows up or moves on."""
        text = text.strip()
        self._moved = False
        if text.casefold() == SKIP:
            async for event in self._run(self._move_on(prompts.INTERVIEW_WHY_SKIP), None):
                yield event
            return
        if self.current is not None and self.current.id not in self._answered:
            self._answered.append(self.current.id)
        if self._follow_ups >= MAX_FOLLOW_UPS:
            note = self._move_on(prompts.INTERVIEW_WHY_DONE)
        else:
            left = MAX_FOLLOW_UPS - self._follow_ups
            note = prompts.INTERVIEW_FOLLOW.format(left=left, s="" if left == 1 else "s")
        async for event in self._run(f"{text}\n\n{note}", text):
            yield event

    async def finish(self) -> InterviewOutcome:
        """Save the transcript, distil it (on_finish), resolve the questions answered."""
        mirror = self.mirror
        path = mirror.end_session()
        mirror.extra_tools.pop("next_question", None)
        claim_ids: list[str] = []
        if path is not None and self.on_finish is not None:
            claim_ids = list(await self.on_finish(path))
        resolved = []
        for question_id in self._answered:
            try:
                mirror.model.resolve_question(question_id)
                resolved.append(question_id)
            except KeyError:  # removed from open-questions.md meanwhile
                continue
        if resolved:
            mirror.model.save()
        return InterviewOutcome(transcript=path, claim_ids=claim_ids, answered=resolved)
