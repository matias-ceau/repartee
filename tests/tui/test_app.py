"""The TUI driven headlessly with Textual's pilot, a fake model and the sample persona."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from textual.widgets import Input

from repartee.llm import FakeLLM
from repartee.llm.base import LLMEvent, LLMRequest, LLMUnavailable, TextDelta
from repartee.llm.fake import tool_call
from repartee.mirror import Mirror
from repartee.runtime import Runtime
from repartee.schema import ClaimStatus
from repartee.tui.app import MirrorApp
from repartee.tui.widgets import DraftBlock, EvalView, LookupPanel, Notice, ReviewView, Turn

from .conftest import BURST, add_claims

SIZE = (100, 30)


async def submit(app: MirrorApp, pilot, text: str) -> None:
    """Type a line into the prompt, send it, and wait for the work it starts."""
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await settle(app, pilot)


async def settle(app: MirrorApp, pilot) -> None:
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


def turns(app: MirrorApp, who: str) -> list[str]:
    return [t.text for t in app.query(Turn) if t.who == who]


def notices(app: MirrorApp) -> list[str]:
    return [n.text for n in app.query(Notice)]


def search(query: str):
    return tool_call("search_memory", query=query, person=None, source=None, after=None,
                     before=None, author=None)


# ---------------------------------------------------------------------- chat


async def test_chat_round_trip_renders_answer_and_lookups(rt: Runtime, fake: FakeLLM) -> None:
    fake.add([search("vélo")], "I cycle everywhere, as far as I can tell.")
    app = MirrorApp(rt)
    async with app.run_test(size=SIZE) as pilot:
        assert app.mode == "chat"
        assert "chat" in app.query_one("#modebar").text
        await submit(app, pilot, "How do I get around town?")
        assert turns(app, "you") == ["How do I get around town?"]
        assert turns(app, "me") == ["I cycle everywhere, as far as I can tell."]
        panel = app.query_one(LookupPanel)
        assert panel.display
        assert len(panel.entries) == 1
        assert "vélo" in panel.entries[0].text
        assert "excerpts" in panel.entries[0].text

        # The next answer has its own lookups.
        fake.add("Still by bike.")
        await submit(app, pilot, "And now?")
        assert panel.entries == []

        await submit(app, pilot, "/sources")
        assert not panel.display
        await submit(app, pilot, "/sources")
        assert panel.display


class GatedLLM(FakeLLM):
    """Streams its first piece, then waits for the test to open the gate."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        first = True
        async for event in super().stream(request):
            if isinstance(event, TextDelta) and first and request.messages[-1].tool_results:
                yield event
                await self.gate.wait()
                first = False
                continue
            yield event


async def test_text_and_lookups_show_while_the_answer_streams(
        rt: Runtime, monkeypatch) -> None:
    gated = GatedLLM()
    gated.add([search("bike")], "I cycle everywhere.")
    rt.llm = gated
    app = MirrorApp(rt)
    async with app.run_test(size=SIZE) as pilot:
        app.query_one("#prompt", Input).value = "How do I get around?"
        await pilot.press("enter")
        for _ in range(50):
            await pilot.pause(0.02)
            if turns(app, "me"):
                break
        # Mid-answer: the lookup is listed and the first words are on screen.
        assert turns(app, "me") == ["I "]
        assert len(app.query_one(LookupPanel).entries) == 1
        assert app.busy
        gated.gate.set()
        await settle(app, pilot)
        assert turns(app, "me") == ["I cycle everywhere."]
        assert not app.busy


# --------------------------------------------------------------------- draft


async def test_draft_shows_burst_and_accept_writes_file(rt: Runtime, fake: FakeLLM) -> None:
    app = MirrorApp(rt)
    async with app.run_test(size=SIZE) as pilot:
        await submit(app, pilot, "/draft @Léa dis-lui que je serai en retard ce soir")
        assert app.mode == "draft"
        assert "draft" in app.query_one("#modebar").text
        block = app.query_one(DraftBlock)
        assert block.messages == [BURST]
        assert len(block.query(".bubble")) == len(BURST)
        assert app.query_one(LookupPanel).entries  # what the draft drew on

        # A plain follow-up revises the pending draft.
        await submit(app, pilot, "plus court")
        assert len(app.query(DraftBlock)) == 2

        await submit(app, pilot, "/accept")
        drafts = list(rt.home.drafts_dir.glob("*.md"))
        assert len(drafts) == 1
        assert "ouais pas de souci" in drafts[0].read_text(encoding="utf-8")
        assert any("Saved" in n for n in notices(app))

        # Nothing pending any more.
        await submit(app, pilot, "/accept")
        assert any("No draft" in n for n in notices(app))

        # Accepting with the text the owner actually used, messages split on " | ".
        await submit(app, pilot, "@Léa dis-lui oui pour samedi")
        await submit(app, pilot, "/accept ok pour samedi | bisous")
        drafts = sorted(rt.home.drafts_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)
        assert len(drafts) == 2
        text = max(drafts, key=lambda p: "bisous" in p.read_text(encoding="utf-8")
                   ).read_text(encoding="utf-8")
        assert "ok pour samedi\n\nbisous" in text


# -------------------------------------------------------------------- review


async def test_review_keys_change_claim_status_and_count(rt: Runtime) -> None:
    ids = add_claims(rt)
    app = MirrorApp(rt)
    async with app.run_test(size=SIZE) as pilot:
        await submit(app, pilot, "/review")
        view = app.query_one(ReviewView)
        assert app.mode == "review"
        assert view.has_focus
        assert view.remaining == 3
        assert "Claim number 0" in view.card_text
        assert "Evidence" in view.card_text

        await pilot.press("y")
        assert rt.model.claim(ids[0]).status == ClaimStatus.CONFIRMED
        assert view.remaining == 2 and "Claim number 1" in view.card_text

        await pilot.press("s")
        assert rt.model.claim(ids[1]).status == ClaimStatus.INFERRED
        assert view.remaining == 1 and "Claim number 2" in view.card_text

        await pilot.press("e")
        prompt = app.query_one("#prompt", Input)
        assert prompt.has_focus and prompt.value == "Claim number 2 about getting around."
        prompt.value = "I mostly walk."
        await pilot.press("enter")
        await pilot.pause()
        edited = rt.model.claim(ids[2])
        assert edited.status == ClaimStatus.CONFIRMED and edited.text == "I mostly walk."
        assert view.remaining == 0
        assert "Nothing left" in view.card_text
        assert view.has_focus

    # A fresh review shows the skipped claim; n rejects it.
    app = MirrorApp(rt, mode="review")
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        view = app.query_one(ReviewView)
        assert view.remaining == 1
        await pilot.press("n")
        assert rt.model.claim(ids[1]) is None
        assert view.remaining == 0


# ----------------------------------------------------------------- interview


async def test_interview_asks_skips_ends_and_reviews_its_claims(
        rt: Runtime, fake: FakeLLM) -> None:
    fake.on("interview", "Where did you grow up?")
    before = set(rt.home.interviews_dir.glob("*.md"))
    app = MirrorApp(rt)
    async with app.run_test(size=SIZE) as pilot:
        await submit(app, pilot, "/interview childhood")
        assert app.mode == "interview"
        assert turns(app, "me") == ["Where did you grow up?"]
        await submit(app, pilot, "In Nantes, until I was eleven.")
        assert turns(app, "you") == ["In Nantes, until I was eleven."]
        assert len(turns(app, "me")) == 2
        await submit(app, pilot, "/skip")
        assert len(turns(app, "me")) == 3

        await submit(app, pilot, "/end")
        new = set(rt.home.interviews_dir.glob("*.md")) - before
        assert len(new) == 1
        assert "In Nantes" in new.pop().read_text(encoding="utf-8")
        # Straight into a review of what the interview produced.
        assert app.mode == "review"
        view = app.query_one(ReviewView)
        assert view.remaining >= 1
        assert "childhood" in view.card_text


async def test_end_outside_an_interview_says_so(rt: Runtime) -> None:
    app = MirrorApp(rt)
    async with app.run_test(size=SIZE) as pilot:
        await submit(app, pilot, "/end")
        assert any("No interview" in n for n in notices(app))


# ---------------------------------------------------------------------- eval


class StubBlindTest:
    """Stands in for evals.voice_blind.BlindTest."""

    instances: list[StubBlindTest] = []

    def __init__(self, rt: Runtime, **_options) -> None:
        from pydantic import BaseModel

        class Pair(BaseModel):
            item_id: str
            register_name: str
            brief: str
            real: list[str]
            generated: list[str]
            real_is_a: bool

        self.rt = rt
        self.pairs = [
            Pair(item_id="wa-1", register_name="whatsapp.fr", brief="b", real=["vrai texte"],
                 generated=["faux texte", "en deux"], real_is_a=True),
            Pair(item_id="wa-2", register_name="whatsapp.fr", brief="b", real=["encore vrai"],
                 generated=["encore faux"], real_is_a=False),
        ]
        self.picks: list[tuple[str, bool]] = []
        self.saved: Path | None = None
        StubBlindTest.instances.append(self)

    async def prepare(self):
        return self.pairs

    def record(self, pair, picked_a: bool) -> None:
        self.picks.append((pair.item_id, picked_a))

    def score(self) -> dict[str, float]:
        fooled = sum(1 for (_, a), p in zip(self.picks, self.pairs, strict=False)
                     if a != p.real_is_a)
        return {"whatsapp.fr": fooled / len(self.picks), "overall": fooled / len(self.picks)}

    def save(self) -> Path:
        self.saved = self.rt.home.root / "evals" / "voice-test.json"
        self.saved.parent.mkdir(exist_ok=True)
        self.saved.write_text("{}", encoding="utf-8")
        return self.saved


async def test_eval_records_picks_and_shows_score(rt: Runtime) -> None:
    StubBlindTest.instances.clear()
    app = MirrorApp(rt, blind_test=StubBlindTest)
    async with app.run_test(size=SIZE) as pilot:
        await submit(app, pilot, "/eval")
        assert app.mode == "eval"
        view = app.query_one(EvalView)
        assert view.has_focus
        assert "vrai texte" in view.text_a and "faux texte" in view.text_b
        await pilot.press("a")  # right: A was real
        assert "Right" in view.result_text and "1 of 1" in view.result_text
        assert "encore faux" in view.text_a and "encore vrai" in view.text_b
        await pilot.press("a")  # wrong: B was real
        await pilot.pause()
        test = StubBlindTest.instances[0]
        assert test.picks == [("wa-1", True), ("wa-2", True)]
        assert "Not this time" in view.result_text and "1 of 2" in view.result_text
        assert test.saved is not None and test.saved.is_file()
        assert "voice-test.json" in view.result_text


# ---------------------------------------------------------- no model, errors


async def test_opens_without_a_model_and_review_still_works(rt_no_llm: Runtime) -> None:
    rt_no_llm.llm_error = "No credentials: set ANTHROPIC_API_KEY."
    add_claims(rt_no_llm)
    app = MirrorApp(rt_no_llm)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        said = notices(app)
        assert any("set ANTHROPIC_API_KEY" in n and "/review" in n for n in said)
        await submit(app, pilot, "hello?")
        assert turns(app, "me") == []
        assert any("No model" in n for n in notices(app)[len(said):])
        await submit(app, pilot, "/review")
        view = app.query_one(ReviewView)
        assert view.remaining == 3
        await pilot.press("y")
        assert view.remaining == 2


async def test_without_a_model_the_fallback_line_names_the_key(rt_no_llm: Runtime) -> None:
    app = MirrorApp(rt_no_llm)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        assert any("ANTHROPIC_API_KEY" in n for n in notices(app))
        for command in ("/draft @Léa salut", "/interview", "/eval"):
            await submit(app, pilot, command)
        assert app.is_running


async def test_model_error_shows_a_line_and_the_turn_can_be_retried(
        rt: Runtime, fake: FakeLLM) -> None:
    fake.add(LLMUnavailable("The network is unreachable."))
    app = MirrorApp(rt)
    async with app.run_test(size=SIZE) as pilot:
        await submit(app, pilot, "Where do I live?")
        errors = [n for n in app.query(Notice) if n.has_class("error")]
        assert len(errors) == 1 and "network is unreachable" in errors[0].text
        assert "\n" not in errors[0].text
        # The failed message is back in the prompt; Enter sends it again.
        prompt = app.query_one("#prompt", Input)
        assert prompt.value == "Where do I live?"
        fake.add("In Copenhagen.")
        await pilot.press("enter")
        await settle(app, pilot)
        assert turns(app, "me") == ["In Copenhagen."]
        assert app.is_running


# ---------------------------------------------------------------- quitting


async def test_quit_saves_the_session(rt: Runtime, fake: FakeLLM, monkeypatch) -> None:
    ended: list[Path | None] = []
    original = Mirror.end_session

    def spy(self: Mirror) -> Path | None:
        path = original(self)
        ended.append(path)
        return path

    monkeypatch.setattr(Mirror, "end_session", spy)
    fake.add("Hello, me.")
    app = MirrorApp(rt)
    async with app.run_test(size=SIZE) as pilot:
        await submit(app, pilot, "hi")
        await submit(app, pilot, "/quit")
    saved = [p for p in ended if p is not None]
    assert len(saved) == 1
    assert "Hello, me." in saved[0].read_text(encoding="utf-8")
    assert saved[0].parent == rt.home.conversations_dir


async def test_help_lists_the_commands(rt: Runtime) -> None:
    app = MirrorApp(rt)
    async with app.run_test(size=SIZE) as pilot:
        await submit(app, pilot, "/help")
        text = "\n".join(notices(app))
        for command in ("/chat", "/draft", "/interview", "/review", "/eval", "/accept",
                        "/sources", "/quit"):
            assert command in text
        await submit(app, pilot, "/nonsense")
        assert any("Unknown command" in n for n in notices(app))
