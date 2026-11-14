"""The mirror: one agent loop with tools; modes change the instructions, not the machinery.

- chat: answers as the owner, in the first person, from lookups (spec section 13).
- draft: hands the request to the drafter; a plain follow-up message revises the draft.
- interview: the same loop with interviewer instructions; paced by ``Interview``.
- review: walks through unconfirmed claims; no model involved.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, get_args

import regex
from pydantic import BaseModel

from repartee.config import HomeConfig
from repartee.home import Home
from repartee.llm.base import (
    Done,
    LLMClient,
    LLMError,
    LLMRequest,
    Message,
    TextDelta,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from repartee.mirror import prompts
from repartee.mirror.context import build_system, date_note
from repartee.mirror.review import ReviewQueue, render_card
from repartee.mirror.sessions import SessionLog
from repartee.mirror.tools import DrafterLike, Toolbox, ToolOutcome, VoiceLookup, error
from repartee.schema import DraftRequest, DraftResult
from repartee.selfmodel import SelfModel
from repartee.store import Corpus

Mode = Literal["chat", "draft", "interview", "review"]
MAX_TOOL_ROUNDS = 8

LIMIT_NOTE = ("[Lookup limit reached for this answer. Answer now from what you have found; "
              "say plainly what you could not check. No more tool calls.]")
LIMIT_REFUSED = "Lookup limit reached; answer from what you already have."
LIMIT_FALLBACK = ("I couldn't finish looking that up. Could you ask me again, a bit more "
                  "narrowly?")

ToolHandler = Callable[[dict[str, Any]], ToolOutcome | Awaitable[ToolOutcome]]


class TextEvent(BaseModel):
    """A piece of the answer as it streams."""

    type: Literal["text"] = "text"
    text: str


class ToolEvent(BaseModel):
    """A lookup the answer rests on, for the side panel."""

    type: Literal["tool"] = "tool"
    name: str
    summary: str
    refs: list[str]
    is_error: bool = False


class DraftEvent(BaseModel):
    """A draft from the drafter."""

    type: Literal["draft"] = "draft"
    request: DraftRequest
    result: DraftResult


class DoneEvent(BaseModel):
    """End of a turn: the whole answer and every ref it used."""

    type: Literal["done"] = "done"
    text: str
    refs: list[str]


MirrorEvent = TextEvent | ToolEvent | DraftEvent | DoneEvent


class DraftingUnavailable(Exception):
    """The mirror has no drafter."""


@dataclass
class _PendingDraft:
    base: DraftRequest  # the request as the owner first gave it
    request: DraftRequest  # the request behind `result`
    result: DraftResult
    notes: list[str] = field(default_factory=list)  # revision instructions so far


_RECIPIENT = regex.compile(r'^\s*@(?:"([^"]+)"|([^\s"]+))')
_REGISTER = regex.compile(r"^\s*#([\w.-]+)")


def parse_draft_command(text: str) -> tuple[str | None, str | None, str]:
    """'@ana #whatsapp.fr tell her…' -> ('ana', 'whatsapp.fr', 'tell her…')."""
    recipient = register = None
    rest = text
    while True:
        if recipient is None and (m := _RECIPIENT.match(rest)):
            recipient = (m.group(1) or m.group(2)).rstrip(",:;")
        elif register is None and (m := _REGISTER.match(rest)):
            register = m.group(1)
        else:
            break
        rest = rest[m.end():]
    return recipient, register, rest.strip()


def render_draft(result: DraftResult) -> str:
    """A draft as plain text: messages separated by blank lines, variants by '— or —'."""
    return "\n\n— or —\n\n".join("\n\n".join(v) for v in result.variants if v)


class Mirror:
    """The owner's mirror for one session."""

    def __init__(self, home: Home, config: HomeConfig, corpus: Corpus, model: SelfModel,
                 llm: LLMClient, drafter: DrafterLike | None = None, *,
                 voice: VoiceLookup | None = None, clock: Callable[[], datetime] = datetime.now,
                 max_tool_rounds: int = MAX_TOOL_ROUNDS):
        self.home = home
        self.config = config
        self.corpus = corpus
        self.model = model
        self.llm = llm
        self.drafter = drafter
        self.clock = clock
        self.max_tool_rounds = max_tool_rounds
        self.owner = config.owner.name
        self.toolbox = Toolbox(corpus, model, voice, owner_name=self.owner)
        #: Tools added for the current mode by a driver such as Interview.
        self.extra_tools: dict[str, tuple[ToolSpec, ToolHandler]] = {}
        self.mode: Mode = "chat"
        self.history: list[Message] = []
        self.session_title: str | None = None
        self._log: SessionLog | None = None
        self._pending: _PendingDraft | None = None
        self._review: ReviewQueue | None = None

    # ------------------------------------------------------------------ modes

    def set_mode(self, mode: Mode) -> None:
        """Switch mode; the current session is written and a fresh one starts."""
        if mode not in get_args(Mode):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == self.mode:
            return
        self.end_session()
        self.mode = mode
        self.session_title = None
        self.extra_tools.clear()
        self._review = None

    async def send(self, text: str) -> AsyncIterator[MirrorEvent]:
        """One owner message in the current mode; yields events, ending with DoneEvent."""
        if self.mode == "review":
            async for event in self._review_turn(text):
                yield event
            return
        if self.mode == "draft":
            async for event in self._draft_turn(text):
                yield event
            return
        text = text.strip()
        if not text:
            yield DoneEvent(text="", refs=[])
            return
        async for event in self.turn(text, log_text=text):
            yield event

    async def ask(self, text: str) -> str:
        """One answer, as text."""
        answer = ""
        async for event in self.send(text):
            if isinstance(event, DoneEvent):
                answer = event.text
        return answer

    # ------------------------------------------------------------- agent loop

    async def turn(self, prompt: str, *, log_text: str | None) -> AsyncIterator[MirrorEvent]:
        """Run the agent loop on `prompt`; `log_text` is what the transcript records for the
        owner (None for a turn the program writes, such as an interview note)."""
        snapshot = list(self.history)
        content = prompt if self.history else f"{date_note(self.clock())}\n\n{prompt}"
        if self.history and self.history[-1].role == "user":
            # The last turn ended on tool results the model never answered: add to them.
            last = self.history[-1]
            joined = f"{last.text}\n\n{content}" if last.text else content
            self.history[-1] = last.model_copy(update={"text": joined})
        else:
            self.history.append(Message(role="user", text=content))
        try:
            async for event in self._loop(log_text):
                yield event
        except BaseException:
            self.history[:] = snapshot
            raise

    def _specs(self) -> list[ToolSpec]:
        return self.toolbox.specs(self.mode) + [spec for spec, _ in self.extra_tools.values()]

    async def _loop(self, log_text: str | None) -> AsyncIterator[MirrorEvent]:
        system = build_system(self.mode, self.owner, self.model.core_text())
        specs = self._specs()
        purpose = "interview" if self.mode == "interview" else "chat"
        texts: list[str] = []
        refs: list[str] = []
        rounds = 0
        while True:
            request = LLMRequest(tier="chat", purpose=purpose, system=system,
                                 messages=list(self.history), tools=specs)
            pieces: list[str] = []
            done = None
            async for event in self.llm.stream(request):
                if isinstance(event, TextDelta) and event.text:
                    if not pieces and texts:
                        yield TextEvent(text="\n\n")
                    pieces.append(event.text)
                    yield TextEvent(text=event.text)
                elif isinstance(event, Done):
                    done = event.response
            if done is None:
                raise LLMError("the model's stream ended without a final response")
            self.history.append(done.message)  # replayed unchanged, thinking included
            said = "".join(pieces) if pieces else done.text
            if said.strip():
                if not pieces:
                    if texts:
                        yield TextEvent(text="\n\n")
                    yield TextEvent(text=said)
                texts.append(said.strip())
            calls = done.tool_calls
            if not calls:
                break
            if rounds >= self.max_tool_rounds:
                # It called tools after being told to stop: close the calls and give up.
                self.history.append(Message(role="user", tool_results=[
                    ToolResult(tool_call_id=c.id, content=LIMIT_REFUSED, is_error=True)
                    for c in calls]))
                if not texts:
                    yield TextEvent(text=LIMIT_FALLBACK)
                    texts.append(LIMIT_FALLBACK)
                break
            results = []
            for call in calls:
                outcome = await self._call_tool(call)
                refs.extend(outcome.refs)
                yield ToolEvent(name=call.name, summary=outcome.summary, refs=outcome.refs,
                                is_error=outcome.is_error)
                results.append(ToolResult(tool_call_id=call.id, content=outcome.content,
                                          is_error=outcome.is_error))
            rounds += 1
            note = LIMIT_NOTE if rounds >= self.max_tool_rounds else ""
            self.history.append(Message(role="user", tool_results=results, text=note))

        answer = "\n\n".join(texts)
        self._record(log_text, answer)
        yield DoneEvent(text=answer, refs=list(dict.fromkeys(refs)))

    async def _call_tool(self, call: ToolCall) -> ToolOutcome:
        extra = self.extra_tools.get(call.name)
        if extra is None:
            return await self.toolbox.call(call.name, call.input)
        try:
            outcome = extra[1](call.input)
            return await outcome if inspect.isawaitable(outcome) else outcome
        except Exception as exc:  # same contract as Toolbox.call
            return error(call.name, f"{call.name} failed: {type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- drafts

    async def draft(self, request: DraftRequest) -> DraftResult:
        """Draft through the drafter; the result waits for accept_draft."""
        if self.drafter is None:
            raise DraftingUnavailable(prompts.DRAFT_UNAVAILABLE)
        result = await self.drafter.draft(request)
        self._pending = _PendingDraft(base=request, request=request, result=result)
        return result

    def accept_draft(self, final: Sequence[str] | None = None, *,
                     variant: int = 0) -> Path | None:
        """Record the pending draft (as is, or the text the owner used); None if none."""
        if self._pending is None or self.drafter is None:
            return None
        pending, self._pending = self._pending, None
        return self.drafter.record(pending.request, pending.result, variant=variant,
                                   final=final)

    async def _draft_turn(self, text: str) -> AsyncIterator[MirrorEvent]:
        text = text.strip()
        if self.drafter is None:
            yield TextEvent(text=prompts.DRAFT_UNAVAILABLE)
            yield DoneEvent(text=prompts.DRAFT_UNAVAILABLE, refs=[])
            return
        recipient, register, intent = parse_draft_command(text)
        pending = self._pending
        if recipient or register or pending is None:
            if not intent:
                ask = "What should it say?"
                yield TextEvent(text=ask)
                yield DoneEvent(text=ask, refs=[])
                return
            base = DraftRequest(intent=intent, recipient=recipient, register_name=register)
            request, notes = base, []
        else:
            # A plain message while a draft is pending revises it.
            base, notes = pending.base, [*pending.notes, text]
            previous = "\n".join(pending.result.variants[0]) if pending.result.variants else ""
            context = "My previous draft, to revise:\n" + previous
            if base.context:
                context = f"{base.context}\n\n{context}"
            intent_text = base.intent + "".join(f"\nRevision: {n}" for n in notes)
            request = base.model_copy(update={"intent": intent_text, "context": context})
        result = await self.drafter.draft(request)
        self._pending = _PendingDraft(base=base, request=request, result=result, notes=notes)
        rendered = render_draft(result)
        who = f" for {request.recipient}" if request.recipient else ""
        yield ToolEvent(name="draft", refs=list(result.used),
                        summary=f"draft{who} in {result.register_name}, "
                                f"{len(result.used)} sources")
        yield DraftEvent(request=request, result=result)
        self._record(text, rendered)
        yield DoneEvent(text=rendered, refs=list(result.used))

    # ---------------------------------------------------------------- review

    async def _review_turn(self, text: str) -> AsyncIterator[MirrorEvent]:
        if self._review is None:
            self._review = ReviewQueue(self.model, self.corpus)
        queue, command = self._review, text.strip()
        word, _, rest = command.partition(" ")
        word = word.lower()
        message = ""
        try:
            if queue.next() is None:
                message = "Nothing left to review."
            elif word in ("", "next", "show"):
                pass
            elif word in ("y", "yes", "confirm"):
                queue.confirm()
            elif word in ("n", "no", "reject"):
                queue.reject()
            elif word in ("s", "skip"):
                queue.skip()
            elif word in ("e", "edit") and rest.strip():
                queue.edit(rest.strip())
            else:
                message = prompts.REVIEW_HELP
        except ValueError as exc:
            message = f"{exc}. {prompts.REVIEW_HELP}"
        if not message:
            card = queue.next()
            message = (f"{render_card(card)}\n\n{queue.remaining()} left. {prompts.REVIEW_HELP}"
                       if card else "Nothing left to review.")
        yield TextEvent(text=message)
        yield DoneEvent(text=message, refs=[])

    # -------------------------------------------------------------- sessions

    def _record(self, owner_text: str | None, mirror_text: str) -> None:
        if self._log is None:
            directory = (self.home.interviews_dir if self.mode == "interview"
                         else self.home.conversations_dir)
            self._log = SessionLog(directory, self.mode, self.session_title,
                                   started=self.clock())
        if owner_text:
            self._log.add("me", owner_text)
        self._log.add("mirror", mirror_text)
        self._log.write()  # saved after every turn, so nothing is lost on a crash

    def end_session(self) -> Path | None:
        """Write the transcript and start afresh; None when nothing was said."""
        log, self._log = self._log, None
        self.history = []
        return log.write() if log is not None and log.turns else None
