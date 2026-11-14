"""The agent loop, prompts, draft mode and sessions. All data is invented."""

from __future__ import annotations

from datetime import datetime

import pytest

from repartee.ingest import parse_transcript
from repartee.llm.base import LLMError
from repartee.llm.fake import tool_call
from repartee.mirror import DoneEvent, DraftEvent, TextEvent, ToolEvent
from repartee.mirror.agent import MAX_TOOL_ROUNDS
from repartee.schema import DraftRequest
from tests.mirror.conftest import CORE, MOVING, OWNER, StubDrafter, StubVoice


async def _collect(agen):
    return [event async for event in agen]


async def test_chat_turn_with_tool_round_trip(make_mirror, fake):
    fake.add([tool_call("search_self", query="moving", kind=None)],
             "I went back and forth all spring.")
    mirror = make_mirror()
    events = await _collect(mirror.send("what did I think about moving?"))

    kinds = [type(e).__name__ for e in events]
    assert kinds[0] == "ToolEvent" and kinds[-1] == "DoneEvent"
    assert all(k == "TextEvent" for k in kinds[1:-1]) and len(kinds) > 3
    tool = events[0]
    assert tool.name == "search_self" and MOVING in tool.refs and tool.summary
    done = events[-1]
    assert done.text == "I went back and forth all spring."
    assert "".join(e.text for e in events if isinstance(e, TextEvent)) == done.text
    assert MOVING in done.refs

    first, second = fake.requests
    assert first.purpose == "chat" and first.tier == "chat"
    assert {t.name for t in first.tools} >= {"search_self", "search_memory", "note_gap"}
    # Second request replays the assistant turn unchanged, then one user message of results.
    assert len(second.messages) == 3
    assert second.messages[1].tool_calls[0].name == "search_self"
    results = second.messages[2]
    assert results.role == "user" and len(results.tool_results) == 1
    assert results.tool_results[0].tool_call_id == second.messages[1].tool_calls[0].id
    assert "moving" in results.tool_results[0].content.lower()


async def test_assistant_turn_is_replayed_as_returned(make_mirror, fake):
    fake.add([tool_call("search_self", query="a", kind=None),
              tool_call("search_memory", query="b", person=None, source=None, after=None,
                        before=None, author=None)],
             "done")
    mirror = make_mirror()
    await _collect(mirror.send("hi"))
    second = fake.requests[1]
    assert len(second.messages[2].tool_results) == 2  # all results in one user message
    assert second.messages[1] is mirror.history[1]  # the response's own message object


async def test_system_prompt_has_core_and_cache_flag_and_no_date(make_mirror, fake):
    fake.on("chat", "fine")
    mirror = make_mirror()
    await mirror.ask("hello")
    await mirror.ask("and again")
    first, second = fake.requests
    assert len(first.system) == 2
    instructions, core = first.system
    assert not instructions.cache and core.cache
    assert CORE in core.text and OWNER in instructions.text
    assert "invent" in instructions.text.lower() and "don't know" in instructions.text
    for block in first.system:
        assert "2026" not in block.text and "November" not in block.text
    assert [b.text for b in first.system] == [b.text for b in second.system]
    assert "14 November 2026" in first.messages[0].text
    assert first.messages[0].text.endswith("hello")
    assert second.messages[2].text == "and again"


async def test_core_missing_still_gives_stable_block(make_mirror, fake, world):
    (world.home.self_dir / "core.md").unlink()
    world.model.load()
    fake.on("chat", "I don't know yet.")
    await make_mirror().ask("who am I?")
    assert fake.requests[0].system[1].text.strip()


async def test_tool_round_limit(make_mirror, fake):
    for n in range(MAX_TOOL_ROUNDS + 1):
        fake.add([tool_call("search_self", query=f"q{n}", kind=None)])
    mirror = make_mirror()
    events = await _collect(mirror.send("tell me everything"))
    assert len(fake.requests) == MAX_TOOL_ROUNDS + 1
    last = fake.requests[-1]
    assert "limit" in last.messages[-1].text.lower()
    tool_events = [e for e in events if isinstance(e, ToolEvent)]
    assert len(tool_events) == MAX_TOOL_ROUNDS
    done = events[-1]
    assert isinstance(done, DoneEvent) and done.text  # an honest fallback, never empty
    # History stays valid: every tool call has a result before the next user turn.
    fake.on("chat", "ok")
    await mirror.ask("shorter please")
    final = fake.requests[-1].messages
    assert final[-2].role == "assistant" and final[-1].role == "user"
    assert final[-1].tool_results and final[-1].text == "shorter please"


async def test_llm_error_rolls_back_history(make_mirror, fake):
    fake.add(LLMError("boom"))
    mirror = make_mirror()
    with pytest.raises(LLMError):
        await _collect(mirror.send("hello"))
    assert mirror.history == []
    fake.add("hi")
    assert await mirror.ask("hello") == "hi"
    assert "14 November 2026" in fake.requests[-1].messages[0].text


async def test_get_voice_offered_only_with_voice(make_mirror, fake):
    fake.on("chat", "ok")
    await make_mirror(voice=StubVoice()).ask("hi")
    assert "get_voice" in {t.name for t in fake.requests[-1].tools}


async def test_draft_mode_parses_recipient_and_yields_draft_event(make_mirror, fake, world):
    drafter = StubDrafter(world.home.drafts_dir)
    mirror = make_mirror(drafter=drafter)
    mirror.set_mode("draft")
    events = await _collect(mirror.send("@ana #whatsapp.fr tell her I can't make it Friday"))
    assert fake.requests == []
    request = drafter.requests[0]
    assert request.recipient == "ana" and request.register_name == "whatsapp.fr"
    assert request.intent == "tell her I can't make it Friday"
    draft = next(e for e in events if isinstance(e, DraftEvent))
    assert draft.result.variants == [["désolée", "je peux pas vendredi"]]
    assert any(isinstance(e, ToolEvent) and e.name == "draft" for e in events)
    done = events[-1]
    assert isinstance(done, DoneEvent) and "je peux pas vendredi" in done.text

    await _collect(mirror.send("shorter"))
    again = drafter.requests[1]
    assert again.recipient == "ana" and again.register_name == "whatsapp.fr"
    assert again.intent.startswith("tell her I can't make it Friday") and "shorter" in again.intent
    assert "je peux pas vendredi" in (again.context or "")


async def test_draft_mode_without_drafter_says_so(make_mirror, fake):
    mirror = make_mirror()
    mirror.set_mode("draft")
    events = await _collect(mirror.send("@ana hi"))
    assert "not set up" in events[-1].text and fake.requests == []


async def test_accept_draft_records_final(make_mirror, world):
    drafter = StubDrafter(world.home.drafts_dir)
    mirror = make_mirror(drafter=drafter)
    assert mirror.accept_draft() is None
    result = await mirror.draft(DraftRequest(intent="say hi", recipient="Ana"))
    assert result.variants
    path = mirror.accept_draft(["coucou !"])
    assert path is not None and path.read_text(encoding="utf-8") == "coucou !"
    assert drafter.recorded[0][3] == ["coucou !"]
    assert mirror.accept_draft() is None  # nothing pending any more


async def test_session_log_format_parses_with_parse_transcript(make_mirror, fake, world):
    fake.on("chat", "Line one.\n**me:** not a real turn\n\nLine three.")
    mirror = make_mirror()
    await mirror.ask('what did "I" say: about Lyon?\nsecond line')
    await mirror.ask("ok")
    path = mirror.end_session()
    assert path is not None and path.parent == world.home.conversations_dir
    assert path.name.startswith("2026-11-14-103205-")

    items = parse_transcript(path, world.config.owner)
    assert [i.is_self for i in items] == [True, False, True, False]
    assert items[0].text == 'what did "I" say: about Lyon?\nsecond line'
    assert "not a real turn" in items[1].text and "Line three." in items[1].text
    assert items[0].author == OWNER and items[1].author == "mirror"
    assert items[0].timestamp == datetime(2026, 11, 14, 10, 32, 5)
    assert items[0].meta["mode"] == "chat"
    assert mirror.end_session() is None  # nothing new to write
    assert mirror.history == []


async def test_draft_session_logs_draft_as_mirror_turn(make_mirror, world):
    mirror = make_mirror(drafter=StubDrafter(world.home.drafts_dir))
    mirror.set_mode("draft")
    await _collect(mirror.send("@ana say hi"))
    path = mirror.end_session()
    items = parse_transcript(path, world.config.owner)
    assert items[0].is_self and items[0].text == "@ana say hi"
    assert not items[1].is_self and "je peux pas vendredi" in items[1].text


async def test_set_mode_writes_previous_session(make_mirror, fake, world):
    fake.on("chat", "ok")
    mirror = make_mirror()
    await mirror.ask("hi")
    mirror.set_mode("review")
    assert len(list(world.home.conversations_dir.glob("*.md"))) == 1
    assert mirror.history == []


async def test_log_sessions_off_writes_no_transcript(world, fake):
    from repartee.mirror import Mirror

    fake.add("I went back and forth all spring.")
    mirror = Mirror(world.home, world.config, world.corpus, world.model, fake, log_sessions=False)
    await mirror.ask("what did I think about moving?")
    assert mirror.end_session() is None
    assert not list(world.home.conversations_dir.glob("*.md"))
