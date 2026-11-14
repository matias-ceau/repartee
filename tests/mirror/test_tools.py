"""Tools over the store and the self-model. All data is invented."""

from __future__ import annotations

from repartee.mirror.tools import MAX_CHARS, Toolbox
from tests.mirror.conftest import ANA, MOVING, OWNER, StubVoice


def _box(world, voice=None) -> Toolbox:
    return Toolbox(world.corpus, world.model, voice, owner_name=OWNER)


async def test_tools_search_self_read_page_search_memory_read_memory(world):
    box = _box(world)

    found = await box.call("search_self", {"query": "Montréal move", "kind": None})
    assert not found.is_error
    assert world.claims["montreal"] in found.content
    assert "inferred" in found.content and "from 2022" in found.content
    assert "nt-j1" in found.content  # evidence ids to follow up
    assert MOVING in found.refs
    assert found.summary and "\n" not in found.summary
    assert len(found.content) <= MAX_CHARS

    ana = await box.call("search_self", {"query": "Ana", "kind": "person"})
    assert ANA in ana.content and "confirmed" in ana.content

    page = await box.call("read_page", {"path": "people/ana-ruiz.md"})
    assert "We met at university" in page.content
    assert f"{world.claims['met']} · confirmed" in page.content
    assert "^c-" not in page.content
    assert page.refs == [ANA]

    by_name = await box.call("read_page", {"path": "Ana"})
    assert by_name.refs == [ANA]

    memory = await box.call("search_memory", {
        "query": "Montréal", "person": "Ana", "source": None, "after": "2022",
        "before": None, "author": None})
    assert "ch-b" in memory.content and "ch-a" not in memory.content
    assert "2022-04-02" in memory.content and "whatsapp" in memory.content
    assert memory.refs == ["ch-b"]
    assert len(memory.content) <= MAX_CHARS

    journal = await box.call("search_memory", {
        "query": "moving", "person": None, "source": "journal", "after": None,
        "before": None, "author": "self"})
    assert journal.refs == ["ch-j"]

    read = await box.call("read_memory", {"chunk_id": "ch-a", "around": 1})
    assert "la fac" in read.content and "Montréal" in read.content
    assert read.refs == ["ch-a", "ch-b"]
    by_item = await box.call("read_memory", {"chunk_id": "wa-b2", "around": None})
    assert by_item.refs == ["ch-b"]


async def test_search_self_drops_claims_rejected_since_indexing(world):
    world.model.reject(world.claims["nurse"])
    out = await _box(world).call("search_self", {"query": "nurse night shifts", "kind": None})
    assert world.claims["nurse"] not in out.content


async def test_tool_errors_are_outcomes_not_exceptions(world):
    box = _box(world)
    cases = [
        ("no_such_tool", {}),
        ("search_self", {}),
        ("search_self", {"query": "x", "kind": "planet"}),
        ("search_memory", {"query": "x", "after": "last spring"}),
        ("read_memory", {"chunk_id": "ch-nope"}),
        ("read_page", {"path": "people/nobody"}),
        ("get_voice", {"register": None, "recipient": None, "topic": None}),
        ("note_gap", {"question": "  "}),
    ]
    for name, args in cases:
        out = await box.call(name, args)
        assert out.is_error, (name, args)
        assert out.content and out.summary


async def test_empty_results_say_so_and_suggest_next_step(world):
    out = await _box(world).call("search_memory", {"query": "zzqx", "person": "Nobody"})
    assert not out.is_error and out.refs == []
    assert "Nothing" in out.content


async def test_note_gap_adds_question(world):
    box = _box(world)
    out = await box.call("note_gap", {"question": "Where did I live before Lyon?"})
    assert not out.is_error
    questions = world.model.questions()
    assert [q.text for q in questions] == ["Where did I live before Lyon?"]
    assert out.refs == [questions[0].id]
    assert "Where did I live" in (world.home.self_dir / "open-questions.md").read_text()


async def test_get_voice_and_similar_replies(world):
    box = _box(world, StubVoice())
    voice = await box.call("get_voice", {"register": None, "recipient": "Ana", "topic": None})
    assert "whatsapp.fr" in voice.content and "Short lowercase bursts" in voice.content
    assert "essay.en" in voice.content  # the other registers are listed
    assert "partir me fait peur" in voice.content
    assert "voice/whatsapp.fr" in voice.refs and "wa-b2" in voice.refs

    replies = await box.call("similar_replies", {"context": "alors, Montréal ?",
                                                  "recipient": None})
    assert "partir me fait peur" in replies.content and replies.refs == ["rp-1"]


def test_specs_are_strict_and_depend_on_mode_and_voice(world):
    plain, voiced = _box(world), _box(world, StubVoice())
    chat = {s.name for s in plain.specs("chat")}
    assert chat == {"search_self", "read_page", "search_memory", "read_memory",
                    "similar_replies", "note_gap"}
    assert "get_voice" in {s.name for s in voiced.specs("chat")}
    assert "get_voice" not in {s.name for s in voiced.specs("interview")}
    assert plain.specs("review") == [] and plain.specs("draft") == []
    for spec in voiced.specs("chat"):
        schema = spec.input_schema
        assert schema["additionalProperties"] is False
        assert sorted(schema["required"]) == sorted(schema["properties"])
        assert spec.description
