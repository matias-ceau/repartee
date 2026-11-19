"""The terminal app: one screen where the owner talks with, drafts with, and corrects the mirror.

Modes: chat, draft, interview, review, eval. Slash commands switch between them (/help).
Model work runs in workers so the screen never freezes; failures become one line in the
conversation and the turn can be sent again.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import ContentSwitcher, Input

from repartee.llm.base import LLMError, LLMRefusal, LLMUnavailable
from repartee.mirror import (
    DraftEvent,
    DraftingUnavailable,
    Interview,
    Mirror,
    MirrorEvent,
    ReviewCard,
    ReviewQueue,
    TextEvent,
    ToolEvent,
)
from repartee.mirror.context import claim_status, fmt_when
from repartee.runtime import Runtime
from repartee.selfmodel.model import SelfModelConflict
from repartee.sync import distil_file
from repartee.tui.widgets import (
    DraftBlock,
    EvalView,
    HelpBlock,
    Line,
    LookupPanel,
    ModeBar,
    Notice,
    ReviewView,
    Turn,
)

MODES = ("chat", "draft", "interview", "review", "eval")
CONVERSATION_MODES = ("chat", "draft", "interview")
NO_MODEL = "No model client. Set ANTHROPIC_API_KEY or run `ant auth login`, then restart."
EVAL_PAIRS = 10
ERROR_CHARS = 240

HINTS = {
    "chat": "Enter sends · /draft @name … writes for someone · /help",
    "draft": "@name what to say · a plain message revises · /accept [text | text] · /help",
    "interview": "Answer in your own words · /skip moves on · /end finishes · /help",
    "review": "y confirm · e edit · n reject · s skip · / for a command",
    "eval": "a or b: which one did you write? · / for a command",
}

HELP = [
    ("/chat [message]", "talk with the mirror"),
    ("/draft @name what", "draft a message; a plain message after it revises the draft"),
    ("/accept [text]", "save the draft as is, or the text you actually sent "
                       "(separate messages with |)"),
    ("/interview [topic]", "the mirror asks, you answer; /skip moves on, /end finishes"),
    ("/review", "confirm (y), edit (e), reject (n) or skip (s) what it inferred"),
    ("/eval [n]", "blind voice test: pick which of A or B you wrote"),
    ("/sources", "show or hide the looked-up panel"),
    ("/quit", "save the session and leave (also ctrl+q)"),
    ("Esc", "stop an answer that is still coming"),
]

SOURCE_ABBREV = {"whatsapp": "wa"}


def one_line(text: str, limit: int = ERROR_CHARS) -> str:
    """Whitespace folded, trimmed to `limit` characters."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def describe_error(exc: BaseException) -> str:
    """A failure as one short line a person can act on."""
    message = one_line(str(exc)) or type(exc).__name__
    if isinstance(exc, LLMUnavailable):
        return f"Couldn't reach the model: {message}"
    if isinstance(exc, LLMRefusal):
        return f"The model declined: {message}"
    if isinstance(exc, LLMError):
        return f"Model error: {message}"
    if isinstance(exc, SelfModelConflict):
        return ("The self-model changed on disk meanwhile; it has been reloaded. "
                "Nothing was written. Try again.")
    if isinstance(exc, DraftingUnavailable):
        return message
    return f"Something went wrong ({type(exc).__name__}): {message}"


def render_review_card(card: ReviewCard) -> list[tuple[str, str]]:
    """A claim, its status and the material behind it, as (text, style class) lines."""
    claim = card.claim
    lines = [(f"{card.page_title} › {claim.section}", "card-where"),
             (claim.text, "card-claim"),
             (f"{claim_status(claim)} · {claim.id}", "card-status")]
    if card.evidence:
        lines.append(("Evidence", "card-heading"))
        for excerpt in card.evidence:
            lines.append((f"{fmt_when(excerpt.when)} · {excerpt.container} · {excerpt.author}",
                          "excerpt-where"))
            lines.append((excerpt.text, "excerpt"))
    elif claim.evidence:
        lines.append((f"Evidence: {len(claim.evidence)} item(s) no longer in the index.",
                      "card-heading"))
    else:
        lines.append(("Evidence: none (written by you).", "card-heading"))
    return lines


def _load_blind_test() -> Callable[..., Any]:
    from repartee.evals.voice_blind import BlindTest  # written alongside; imported on use

    return BlindTest


class MirrorApp(App):
    """The owner's mirror in a terminal."""

    CSS_PATH = "app.tcss"
    TITLE = "repartee"
    BINDINGS = [
        Binding("escape", "stop", "stop", show=False),
        Binding("ctrl+q", "quit", "quit", show=False, priority=True),
    ]

    def __init__(self, rt: Runtime, mode: str = "chat", *,
                 blind_test: Callable[..., Any] | None = None):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; one of {', '.join(MODES)}")
        super().__init__()
        self.rt = rt
        self.mode = "chat"
        self.busy = False
        self.mirror: Mirror | None = None
        self._start_mode = mode
        self._blind_factory = blind_test
        self._show_sources = True
        self._interview: Interview | None = None
        self._undistilled: Path | None = None
        self._review: ReviewQueue | None = None
        self._review_ids: list[str] | None = None
        self._editing: str | None = None
        self._blind: Any = None
        self._pairs: list[Any] = []
        self._pair_index = 0
        self._right = 0
        self._picked = 0
        self._blind_saved = False
        self._closed = False

    # ---------------------------------------------------------------- layout

    def compose(self) -> ComposeResult:
        yield ModeBar(self.mode, id="modebar")
        with Horizontal(id="main"):
            with ContentSwitcher(initial="conversation", id="views"):
                yield VerticalScroll(id="conversation")
                yield ReviewView(id="review")
                yield EvalView(id="eval")
            yield LookupPanel(id="lookups")
        yield Line(HINTS["chat"], id="hint")
        yield Input(id="prompt", placeholder="Talk to the mirror, or /help",
                    select_on_focus=False)

    @property
    def conversation(self) -> VerticalScroll:
        return self.query_one("#conversation", VerticalScroll)

    @property
    def prompt(self) -> Input:
        return self.query_one("#prompt", Input)

    async def on_mount(self) -> None:
        if self.rt.llm is not None:
            try:
                self.mirror = self.rt.mirror("chat")
            except LLMError as exc:
                self.rt.llm_error = self.rt.llm_error or str(exc)
        self._sync_panel()
        if self.mirror is None:
            await self._say(self._no_model_line(), error=True)
        else:
            await self._say(f"This is {self.rt.config.owner.name}'s mirror. Type to talk; "
                            "/help lists the commands.")
        await self._enter(self._start_mode, "")
        if self.mode in CONVERSATION_MODES:
            self.prompt.focus()

    def _no_model_line(self) -> str:
        reason = one_line(self.rt.llm_error or NO_MODEL)
        return f"No model connected. {reason} Until then, /review still works."

    # ------------------------------------------------------------ messages

    async def _say(self, text: str, *, error: bool = False) -> None:
        """One line from the program, shown in the current view."""
        if self.mode == "review":
            self.query_one(ReviewView).say(text)
        elif self.mode == "eval":
            self.query_one(EvalView).say(text)
        else:
            await self.conversation.mount(Notice(text, error=error))
            self.conversation.scroll_end(animate=False)

    def _at_bottom(self) -> bool:
        """Whether the conversation shows its end (so new text should keep it there)."""
        view = self.conversation
        return view.scroll_y >= view.max_scroll_y - 1

    def _hint(self, text: str | None = None) -> None:
        self.query_one("#hint", Line).set_text(text or HINTS[self.mode])

    def _sync_panel(self) -> None:
        self.query_one(LookupPanel).display = (
            self._show_sources and self.mirror is not None and self.mode in CONVERSATION_MODES)

    def _set_mode(self, mode: str) -> None:
        if self._interview is not None and mode != "interview":
            self._leave_interview()
        self.mode = mode
        self.query_one(ModeBar).set_mode(mode)
        self.query_one(ContentSwitcher).current = (
            "conversation" if mode in CONVERSATION_MODES else mode)
        if self.mirror is not None and mode in ("chat", "draft"):
            self.mirror.set_mode(mode)
        self._sync_panel()
        self._hint()
        if mode in ("review", "eval"):
            self.query_one(ReviewView if mode == "review" else EvalView).focus()
        else:
            self.prompt.focus()

    async def _enter(self, mode: str, rest: str) -> None:
        if mode == "interview":
            await self._start_interview(rest or None)
        elif mode == "review":
            await self._open_review()
        elif mode == "eval":
            await self._start_eval(rest)
        else:
            self._set_mode(mode)
            if mode == "draft" and not rest:
                await self._say("Draft: write @name and what to say. A plain message after "
                                "it revises the draft; /accept saves it.")
            if rest:
                await self._send(rest)

    # ---------------------------------------------------------------- input

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if self._editing is not None:
            await self._finish_edit(text)
            return
        event.input.value = ""
        if not text:
            return
        if text.startswith("/"):
            await self._command(text)
        elif self.mode in ("chat", "draft"):
            await self._send(text)
        elif self.mode == "interview":
            await self._answer(text)
        else:
            await self._say("Use the keys shown below, or type a /command.")

    async def _command(self, text: str) -> None:
        word, _, rest = text[1:].partition(" ")
        word, rest = word.lower(), rest.strip()
        if word in ("help", "h", "?"):
            if self.mode in CONVERSATION_MODES:
                await self.conversation.mount(HelpBlock(HELP))
                self.conversation.scroll_end(animate=False)
            else:
                await self._say("\n".join(f"{cmd}  {what}" for cmd, what in HELP))
        elif word in ("quit", "q", "exit"):
            await self.action_quit()
        elif word == "sources":
            self._show_sources = not self._show_sources
            self._sync_panel()
        elif self.busy:
            self.prompt.value = text
            await self._say("Still working on the last message; wait, or press Esc to stop it.")
        elif word in MODES:
            await self._enter(word, rest)
        elif word == "accept":
            await self._accept(rest)
        elif word == "skip":
            await self._skip()
        elif word == "end":
            await self._end_interview()
        else:
            await self._say(f"Unknown command /{word}. /help lists the commands.")

    def action_stop(self) -> None:
        """Esc: cancel an edit, stop an answer, or go back to the review or test keys."""
        if self._editing is not None:
            self._editing = None
            self.prompt.value = ""
            self._hint()
            self.query_one(ReviewView).focus()
            return
        if self.busy:
            self.workers.cancel_group(self, "turn")
            return
        if self.mode in ("review", "eval"):
            self.query_one(ReviewView if self.mode == "review" else EvalView).focus()

    async def action_quit(self) -> None:
        """Save the session (and a blind test in progress), then leave."""
        self._save_on_exit()
        self.exit()

    def on_unmount(self) -> None:
        self._save_on_exit()

    def _save_on_exit(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.workers.cancel_group(self, "turn")
        if self.mirror is not None:
            try:
                self.mirror.end_session()
            except Exception:  # leaving anyway; the transcript is also written per turn
                pass
        if self._blind is not None and self._picked and not self._blind_saved:
            try:
                self._blind.save()
            except Exception:
                pass

    # ----------------------------------------------------------- model turns

    def _run(self, events: Callable[[], AsyncIterator[MirrorEvent]], *,
             retry: str | None, after: Callable[[], Any] | None = None) -> None:
        """Consume a mirror turn in a worker: text streams in, lookups go to the panel."""
        self.busy = True
        self._hint("thinking… (Esc stops)")
        self.run_worker(self._consume(events, retry, after), group="turn", exclusive=True,
                        exit_on_error=False)

    async def _consume(self, events: Callable[[], AsyncIterator[MirrorEvent]],
                       retry: str | None, after: Callable[[], Any] | None) -> None:
        panel = self.query_one(LookupPanel)
        panel.clear()
        reply: Turn | None = None
        try:
            async for event in events():
                if isinstance(event, TextEvent):
                    if reply is None:
                        reply = Turn("me")
                        await self.conversation.mount(reply)
                    following = self._at_bottom()
                    reply.append(event.text)
                    if following:
                        self.conversation.scroll_end(animate=False)
                elif isinstance(event, ToolEvent):
                    panel.add(event.summary, [self.describe_ref(r) for r in event.refs],
                              error=event.is_error)
                    self._hint(f"looking up · {one_line(event.summary, 60)}")
                elif isinstance(event, DraftEvent):
                    await self.conversation.mount(self._draft_block(event))
                    self.conversation.scroll_end(animate=False)
            if after is not None:
                result = after()
                if asyncio.iscoroutine(result):
                    await result
        except asyncio.CancelledError:
            await self._say("Stopped.")
            if retry and not self.prompt.value:
                self.prompt.value = retry
            raise
        except Exception as exc:
            await self._fail(exc, retry)
        finally:
            self.busy = False
            self._hint()
            panel.scroll_to_top()

    async def _fail(self, exc: BaseException, retry: str | None) -> None:
        if isinstance(exc, SelfModelConflict):
            self.rt.model.load()
        line = describe_error(exc)
        if retry and not self.prompt.value:
            self.prompt.value = retry
            line += " Press Enter to try again."
        await self._say(line, error=True)

    def _draft_block(self, event: DraftEvent) -> DraftBlock:
        result, request = event.result, event.request
        who = f" for {request.recipient}" if request.recipient else ""
        heading = f"draft{who} · {result.register_name}"
        if result.deviations:
            heading += f" · still off: {one_line('; '.join(result.deviations), 80)}"
        return DraftBlock(heading, result.variants)

    def describe_ref(self, ref: str) -> str:
        """A ref as something a person recognises: a page, or where and when a text was."""
        try:
            chunk = self.rt.corpus.get_chunk(ref)
            if chunk is not None:
                return self._where(chunk.source, chunk.kind, chunk.container, chunk.start)
            item = self.rt.corpus.get_item(ref)
            if item is not None:
                return self._where(item.source, item.kind, item.container, item.timestamp)
        except Exception:
            pass
        return ref

    @staticmethod
    def _where(source: str, kind: Any, container: str, when: Any) -> str:
        """'wa · Léa Fontaine · 2024-12-20', 'journal · 2022-03-15', 'note · Bristol'."""
        label = SOURCE_ABBREV.get(source) or str(getattr(kind, "value", kind) or source)
        place = container.removesuffix(".md").rsplit("/", 1)[-1]
        date = f"{when:%Y-%m-%d}" if when is not None else ""
        parts = [label, place] if date in place else [label, place, date]
        return " · ".join(part for part in parts if part)

    async def _send(self, text: str) -> None:
        if self.mirror is None:
            await self._say(self._no_model_line(), error=True)
            return
        if self.busy:
            self.prompt.value = text
            await self._say("Still working on the last message; wait, or press Esc to stop it.")
            return
        await self.conversation.mount(Turn("you", text))
        self.conversation.scroll_end(animate=False)
        mirror = self.mirror
        self._run(lambda: mirror.send(text), retry=text)

    # ---------------------------------------------------------------- drafts

    async def _accept(self, rest: str) -> None:
        if self.mirror is None:
            await self._say(self._no_model_line(), error=True)
            return
        final = [part.strip() for part in rest.split("|") if part.strip()] or None
        try:
            path = self.mirror.accept_draft(final)
        except Exception as exc:
            await self._fail(exc, None)
            return
        if path is None:
            await self._say("No draft to accept. /draft @name what to say writes one.")
            return
        kept = "your text" if final else "the draft as is"
        await self._say(f"Saved {kept} to {self._relative(path)}.")

    def _relative(self, path: Path) -> str:
        try:
            return Path(path).resolve().relative_to(self.rt.home.root.resolve()).as_posix()
        except ValueError:
            return str(path)

    # ------------------------------------------------------------- interview

    async def _start_interview(self, topic: str | None) -> None:
        if self.mirror is None:
            await self._say(self._no_model_line(), error=True)
            return
        if self._interview is not None:
            self._leave_interview()
        self._set_mode("interview")
        self._undistilled = None
        interview = Interview(self.mirror, topic=topic, on_finish=self._distil)
        self._interview = interview
        about = f" about {topic}" if topic else ""
        await self._say(f"Interview{about}. Answer in your own words; /skip moves on, "
                        "/end finishes and reviews what it learnt.")
        self._run(interview.start, retry=f"/interview {topic or ''}".strip())

    async def _answer(self, text: str) -> None:
        interview = self._interview
        if interview is None:
            await self._say("The interview has ended. /interview starts another.")
            return
        if self.busy:
            self.prompt.value = text
            await self._say("Still working on the last message; wait, or press Esc to stop it.")
            return
        await self.conversation.mount(Turn("you", text))
        self.conversation.scroll_end(animate=False)
        self._run(lambda: interview.answer(text), retry=text)

    async def _skip(self) -> None:
        interview = self._interview
        if interview is None or self.mode != "interview":
            await self._say("/skip moves on in an interview. /interview starts one.")
            return
        await self._say("Skipped.")
        self._run(lambda: interview.answer("/skip"), retry="/skip")

    async def _distil(self, path: Path) -> list[str]:
        self._undistilled = path
        claim_ids = await distil_file(self.rt, path)
        self._undistilled = None
        return claim_ids

    def _leave_interview(self) -> None:
        """Leave an interview without /end: its transcript is kept for the next sync."""
        self._interview = None
        if self.mirror is not None and self.mirror.end_session() is not None:
            self.call_later(self._say, "The interview is saved; the next `repartee sync` "
                            "distils it.")

    async def _end_interview(self) -> None:
        if self.mode != "interview" or (self._interview is None and self._undistilled is None):
            await self._say("No interview running. /interview [topic] starts one.")
            return
        self._hint("saving and distilling the interview…")
        await self._say("Saving the interview and distilling what you said…")
        self.busy = True
        self.run_worker(self._finish_interview(), group="turn", exclusive=True,
                        exit_on_error=False)

    async def _finish_interview(self) -> None:
        interview = self._interview
        try:
            if interview is not None:
                outcome = await interview.finish()
                transcript, claim_ids = outcome.transcript, outcome.claim_ids
            else:  # an earlier /end saved the transcript but could not distil it
                transcript = self._undistilled
                claim_ids = await self._distil(transcript)
        except Exception as exc:
            self._interview = None
            self.busy = False
            self._hint()
            if self._undistilled is not None:
                await self._say(f"{describe_error(exc)} The transcript is saved; /end tries "
                                "the distilling again.", error=True)
            else:
                await self._fail(exc, None)
            return
        self._interview = None
        self.busy = False
        if transcript is None:
            self._set_mode("chat")
            await self._say("Nothing was said, so nothing was saved.")
            return
        saved = self._relative(transcript)
        if not claim_ids:
            self._set_mode("chat")
            await self._say(f"Saved {saved}. It produced no new claims to review.")
            return
        await self._say(f"Saved {saved}; {len(claim_ids)} new claims.")
        await self._open_review(claim_ids)

    # ---------------------------------------------------------------- review

    async def _open_review(self, claim_ids: list[str] | None = None) -> None:
        self._set_mode("review")
        self._review_ids = list(claim_ids) if claim_ids is not None else None
        self._review = ReviewQueue(self.rt.model, self.rt.corpus, self._review_ids)
        view = self.query_one(ReviewView)
        view.say("From this interview." if claim_ids is not None else "")
        self._show_card()

    def _show_card(self) -> None:
        view = self.query_one(ReviewView)
        card = self._review.next() if self._review is not None else None
        if card is None:
            view.show([("Nothing left to review. /chat to talk, /interview to add more.",
                        "card-where")], 0)
        else:
            view.show(render_review_card(card), self._review.remaining())

    @on(ReviewView.Act)
    async def _review_key(self, message: ReviewView.Act) -> None:
        view, queue = self.query_one(ReviewView), self._review
        if message.key == "/":
            self._focus_command()
            return
        card = queue.next() if queue is not None else None
        if card is None:
            view.say("Nothing to review.")
            return
        try:
            if message.key == "y":
                queue.confirm()
                view.say("Confirmed.")
            elif message.key == "n":
                queue.reject()
                view.say("Rejected; it will not be inferred again.")
            elif message.key == "s":
                queue.skip()
                view.say("Skipped for now.")
            elif message.key == "e":
                self._editing = card.claim.id
                self.prompt.value = card.claim.text
                self.prompt.cursor_position = len(card.claim.text)
                self.prompt.focus()
                self._hint("Edit the claim · Enter saves · Esc cancels")
                return
        except SelfModelConflict as exc:
            self.rt.model.load()
            self._review = ReviewQueue(self.rt.model, self.rt.corpus, self._review_ids)
            view.say(describe_error(exc))
        except Exception as exc:
            view.say(describe_error(exc))
        self._show_card()

    async def _finish_edit(self, text: str) -> None:
        view, queue = self.query_one(ReviewView), self._review
        claim_id, self._editing = self._editing, None
        self.prompt.value = ""
        self._hint()
        view.focus()
        if not text:
            view.say("Edit cancelled: the claim needs some text.")
            return
        card = queue.next() if queue is not None else None
        if card is None or card.claim.id != claim_id:
            view.say("That claim is no longer the one under review; edit not saved.")
            self._show_card()
            return
        try:
            queue.edit(text)
            view.say("Saved your wording; it counts as confirmed.")
        except SelfModelConflict as exc:
            self.rt.model.load()
            self._review = ReviewQueue(self.rt.model, self.rt.corpus, self._review_ids)
            view.say(describe_error(exc))
        except Exception as exc:
            view.say(describe_error(exc))
        self._show_card()

    def _focus_command(self) -> None:
        self.prompt.value = "/"
        self.prompt.cursor_position = 1
        self.prompt.focus()

    # ------------------------------------------------------------ blind test

    async def _start_eval(self, rest: str) -> None:
        if self.rt.llm is None:
            await self._say(self._no_model_line(), error=True)
            return
        n = int(rest) if rest.isdigit() and int(rest) > 0 else EVAL_PAIRS
        self._set_mode("eval")
        self._blind, self._pairs = None, []
        self._pair_index = self._right = self._picked = 0
        self._blind_saved = False
        self.query_one(EvalView).show_message(
            f"Preparing the blind test: the mirror writes its own version of {n} of your "
            "held-out texts. This takes a moment.")
        self.busy = True
        self._hint("preparing… (Esc stops)")
        self.run_worker(self._prepare_eval(n), group="turn", exclusive=True,
                        exit_on_error=False)

    async def _prepare_eval(self, n: int) -> None:
        view = self.query_one(EvalView)
        try:
            factory = self._blind_factory or _load_blind_test()
            test = factory(self.rt, n=n)
            pairs = list(await test.prepare())
        except asyncio.CancelledError:
            view.show_message("Blind test stopped. /eval starts it again.")
            raise
        except ImportError:
            view.show_message("The blind test is not available in this version.")
            return
        except Exception as exc:
            view.show_message("The blind test could not start.", describe_error(exc))
            return
        finally:
            self.busy = False
            self._hint()
        if not pairs:
            view.show_message("No held-out texts to test with yet. Run `repartee sync` first.")
            return
        self._blind, self._pairs = test, pairs
        self._show_pair()

    def _show_pair(self) -> None:
        pair = self._pairs[self._pair_index]
        a, b = (pair.real, pair.generated) if pair.real_is_a else (pair.generated, pair.real)
        status = (f"Pair {self._pair_index + 1} of {len(self._pairs)} · {pair.register_name}"
                  " · Which one did you write? Press a or b.")
        self.query_one(EvalView).show_pair(status, a, b)

    @on(EvalView.Pick)
    def _eval_key(self, message: EvalView.Pick) -> None:
        view = self.query_one(EvalView)
        if message.key == "/":
            self._focus_command()
            return
        if self._blind is None or self._pair_index >= len(self._pairs):
            view.say("No pair on screen. /eval starts a blind test.")
            return
        pair = self._pairs[self._pair_index]
        picked_a = message.key == "a"
        try:
            self._blind.record(pair, picked_a)
        except Exception as exc:
            view.say(describe_error(exc))
            return
        right = picked_a == pair.real_is_a
        self._picked += 1
        self._right += right
        yours = "A" if pair.real_is_a else "B"
        verdict = (f"Right: {yours} was yours." if right
                   else f"Not this time: {yours} was yours; the mirror passed as you.")
        tally = f"Score: {self._right} of {self._picked} right."
        self._pair_index += 1
        if self._pair_index < len(self._pairs):
            self._show_pair()
            view.say(f"{verdict} {tally}")
            return
        self._finish_eval(f"{verdict} {tally}")

    def _finish_eval(self, last: str) -> None:
        view = self.query_one(EvalView)
        lines = [last]
        try:
            scores = self._blind.score()
            if scores:
                shares = ", ".join(f"{name} {share:.0%}" for name, share in scores.items())
                lines.append(f"How often the mirror passed as you: {shares} "
                             "(50% means it can't be told apart).")
            saved = self._blind.save()
            self._blind_saved = True
            lines.append(f"Saved to {self._relative(Path(saved))}.")
        except Exception as exc:
            lines.append(describe_error(exc))
        view.show_message("Blind test finished. /eval runs another; /chat goes back.",
                          "\n".join(lines))


def run(rt: Runtime, mode: str = "chat") -> None:
    """Open the TUI on a runtime, starting in `mode`."""
    MirrorApp(rt, mode).run()


__all__ = ["MirrorApp", "run"]
