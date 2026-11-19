"""Interview pacing and the review queue. All data is invented."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from repartee.ingest import parse_transcript
from repartee.llm.fake import tool_call
from repartee.mirror import DoneEvent, Interview, ReviewQueue, explain_claim
from repartee.mirror.review import render_card
from repartee.schema import ClaimOrigin, ClaimStatus
from repartee.selfmodel import SelfModel
from tests.mirror.conftest import ANA, OWNER


async def _text(agen) -> str:
    events = [e async for e in agen]
    assert isinstance(events[-1], DoneEvent)
    return events[-1].text


async def test_interview_flow_asks_follows_up_finishes_and_resolves_questions(
        make_mirror, fake, world):
    q1 = world.model.add_question("Where did I grow up?")
    q2 = world.model.add_question("What did I study at university?", page=ANA)
    seen: list[Path] = []

    async def on_finish(path: Path) -> list[str]:
        seen.append(path)
        return [world.claims["nurse"]]

    fake.add("Where did you grow up?", "What made you leave?", "How did that feel?",
             "What did you study?",
             [tool_call("next_question")], "Thank you, that's all I wanted to ask.",
             purpose="interview")
    interview = Interview(make_mirror(), on_finish=on_finish)

    assert await _text(interview.start()) == "Where did you grow up?"
    first = fake.requests[0]
    assert first.purpose == "interview"
    assert "interview" in first.system[0].text.lower()
    assert q1.text in first.messages[0].text and "14 November 2026" in first.messages[0].text

    assert await _text(interview.answer("In Nantes, until I was eleven.")) == (
        "What made you leave?")
    note = fake.requests[1].messages[-1].text
    assert note.startswith("In Nantes, until I was eleven.")
    assert "follow" in note.lower() and "next_question" in note
    assert "next_question" in {t.name for t in fake.requests[1].tools}

    await _text(interview.answer("My father's job."))
    # Two follow-ups used: the runner moves on by itself and names the next question.
    await _text(interview.answer("Sad, mostly."))
    assert q2.text in fake.requests[3].messages[-1].text

    text = await _text(interview.answer("Geography."))
    assert text == "Thank you, that's all I wanted to ask."
    result = fake.requests[5].messages[-1].tool_results[0].content
    assert "no open questions" in result.lower()

    outcome = await interview.finish()
    assert outcome.transcript is not None and seen == [outcome.transcript]
    assert outcome.transcript.parent == world.home.interviews_dir
    assert outcome.claim_ids == [world.claims["nurse"]]
    assert sorted(outcome.answered) == sorted([q1.id, q2.id])
    assert world.model.questions() == []
    reloaded = SelfModel(world.home.self_dir)
    reloaded.load()
    assert reloaded.questions() == []

    items = parse_transcript(outcome.transcript, world.config.owner)
    assert [i.is_self for i in items] == [False, True, False, True, False, True, False, True,
                                          False]
    assert items[1].text == "In Nantes, until I was eleven."
    assert all(i.kind == "interview" for i in items)
    assert "next_question" not in outcome.transcript.read_text()


async def test_interview_skip_and_topic(make_mirror, fake, world):
    world.model.add_question("Where did I grow up?")
    q2 = world.model.add_question("What did I study at university?")
    q3 = world.model.add_question("Which university friends do I still see?")
    fake.on("interview", "a question")
    interview = Interview(make_mirror(), topic="University")
    await _text(interview.start())
    assert q2.text in fake.requests[0].messages[0].text
    assert "University" in fake.requests[0].messages[0].text

    await _text(interview.answer("/skip"))
    note = fake.requests[1].messages[-1].text
    assert "skip" in note.lower() and q3.text in note

    outcome = await interview.finish()
    assert outcome.answered == []
    assert {q.id for q in world.model.questions()} >= {q2.id, q3.id}
    items = parse_transcript(outcome.transcript, world.config.owner)
    assert not any(i.is_self and "/skip" in i.text for i in items)


async def test_interview_without_questions_uses_coverage(make_mirror, fake):
    fake.on("interview", "Tell me about your work.")
    interview = Interview(make_mirror())
    await _text(interview.start())
    opening = fake.requests[0].messages[0].text
    assert "no open questions" in opening.lower()
    assert "people/ana-ruiz" in opening  # thin pages are named


async def test_interview_finish_with_nothing_said(make_mirror):
    outcome = await Interview(make_mirror()).finish()
    assert outcome.transcript is None and outcome.claim_ids == []


def test_review_queue_order_and_actions(world):
    queue = ReviewQueue(world.model, world.corpus)
    assert queue.remaining() == 3

    card = queue.next()
    assert card.claim.id == world.claims["nurse"] and card.page_title == "Ana Ruiz"
    assert [e.item_id for e in card.evidence] == ["wa-a1"]
    assert card.evidence[0].when == datetime(2021, 3, 12, 14, 5)
    queue.reject()
    assert world.model.claim(world.claims["nurse"]) is None
    assert "night shifts" not in (world.home.self_dir / f"{ANA}.md").read_text()
    assert world.claims["nurse"] not in [c.id for c, _ in world.corpus.search_claims("nurse")]

    assert queue.next().claim.id == world.claims["lyon"]
    queue.edit("Back then I didn't want to leave Lyon.")
    edited = world.model.claim(world.claims["lyon"])
    assert edited.status == ClaimStatus.CONFIRMED and edited.origin == ClaimOrigin.USER

    assert queue.next().claim.id == world.claims["montreal"]
    queue.skip()
    assert queue.remaining() == 0 and queue.next() is None
    assert world.model.claim(world.claims["montreal"]).status == ClaimStatus.INFERRED

    limited = ReviewQueue(world.model, world.corpus, [world.claims["montreal"]])
    assert limited.remaining() == 1
    limited.confirm()
    assert world.model.claim(world.claims["montreal"]).status == ClaimStatus.CONFIRMED
    reloaded = SelfModel(world.home.self_dir)
    reloaded.load()
    assert reloaded.claim(world.claims["montreal"]).status == ClaimStatus.CONFIRMED


def test_explain_claim_returns_excerpts(world):
    card = explain_claim(world.model, world.corpus, world.claims["met"])
    assert [e.item_id for e in card.evidence] == ["wa-a1", "wa-a2"]
    assert card.evidence[1].author == OWNER and card.evidence[1].container == "Ana Ruiz"
    text = render_card(card)
    assert "We met at university" in text and "confirmed" in text and "2021-03-12" in text
    assert explain_claim(world.model, world.corpus, "c-00000000") is None


async def test_review_mode_through_send(make_mirror, fake, world):
    mirror = make_mirror()
    mirror.set_mode("review")
    shown = await _text(mirror.send(""))
    assert "night shifts" in shown and "3 left" in shown
    after = await _text(mirror.send("y"))
    assert world.model.claim(world.claims["nurse"]).status == ClaimStatus.CONFIRMED
    assert "never wanted to leave Lyon" in after
    help_text = await _text(mirror.send("what?"))
    assert "y" in help_text and "edit" in help_text
    assert fake.requests == []
