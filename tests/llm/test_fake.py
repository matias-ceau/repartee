"""FakeLLM: the scripted model every other test relies on."""

import pytest
from pydantic import BaseModel

from repartee.llm import FakeLLM, tool_call
from repartee.llm.base import (
    Done,
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
)


class Verdict(BaseModel):
    real: str
    confidence: float


SEARCH = ToolSpec(
    name="search_self",
    description="Search the self-model.",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
)


def req(text: str = "hi", purpose: str = "chat", **kwargs) -> LLMRequest:
    return LLMRequest(purpose=purpose, messages=[Message(role="user", text=text)], **kwargs)


async def test_fake_queue_and_rules_and_recording():
    fake = FakeLLM(["first"]).add("second")
    fake.on("judge", "standing judge answer")
    fake.on(lambda r: "Ana" in r.messages[-1].text, lambda r: f"echo: {r.messages[-1].text}")

    assert (await fake.complete(req("one"))).text == "first"
    assert (await fake.complete(req("about Ana"))).text == "echo: about Ana"
    assert (await fake.complete(req("x", purpose="judge"))).text == "standing judge answer"
    assert (await fake.complete(req("x", purpose="judge"))).text == "standing judge answer"
    assert (await fake.complete(req("two"))).text == "second"

    assert [r.messages[-1].text for r in fake.requests] == ["one", "about Ana", "x", "x", "two"]
    assert len(fake.calls("judge")) == 2
    fake.assert_exhausted()


async def test_fake_most_recent_rule_wins_and_purpose_queues():
    fake = FakeLLM()
    fake.on("draft", "old rule")
    fake.on("draft", "new rule")
    fake.add("for revise only", purpose="revise")
    fake.add("anyone")

    assert (await fake.complete(req(purpose="draft"))).text == "new rule"
    # "chat" skips the revise-only reply and takes the next general one.
    assert (await fake.complete(req(purpose="chat"))).text == "anyone"
    assert (await fake.complete(req(purpose="revise"))).text == "for revise only"


async def test_fake_records_a_snapshot_of_messages():
    fake = FakeLLM(["a", "b"])
    messages = [Message(role="user", text="hi")]
    await fake.complete(LLMRequest(messages=messages))
    messages.append(Message(role="assistant", text="a"))
    messages.append(Message(role="user", text="again"))
    await fake.complete(LLMRequest(messages=messages))
    assert len(fake.requests[0].messages) == 1
    assert len(fake.requests[1].messages) == 3


async def test_fake_unscripted_request_fails_loudly():
    fake = FakeLLM()
    fake.on("extract", "x")
    fake.add("only for revise", purpose="revise")
    with pytest.raises(AssertionError) as info:
        await fake.complete(req("what did I think about moving?", purpose="consolidate"))
    message = str(info.value)
    assert "consolidate" in message
    assert "what did I think about moving?" in message
    assert "extract" in message  # says what *is* scripted
    assert "revise" in message
    # The request is still recorded, for post-mortems.
    assert fake.requests[-1].purpose == "consolidate"


async def test_fake_assert_exhausted_reports_leftovers():
    fake = FakeLLM(["used", "left over"])
    await fake.complete(req())
    with pytest.raises(AssertionError, match="left over"):
        fake.assert_exhausted()


async def test_fake_stream_yields_deltas_then_done():
    fake = FakeLLM(["I went back and forth all spring."])
    events = [event async for event in fake.stream(req())]
    deltas = [e for e in events if isinstance(e, TextDelta)]
    assert len(deltas) > 1
    assert "".join(d.text for d in deltas) == "I went back and forth all spring."
    assert isinstance(events[-1], Done)
    assert events[-1].response.text == "I went back and forth all spring."
    assert sum(isinstance(e, Done) for e in events) == 1


async def test_fake_stream_tool_calls_after_text():
    fake = FakeLLM([[tool_call("search_self", query="moving")]])
    events = [event async for event in fake.stream(req(tools=[SEARCH]))]
    assert isinstance(events[0], ToolCallEvent)
    assert events[0].call.input == {"query": "moving"}
    assert isinstance(events[-1], Done)


async def test_fake_parsed_reply():
    fake = FakeLLM([Verdict(real="A", confidence=0.7)])
    fake.on("judge", {"real": "B", "confidence": 0.2})
    response = await fake.complete(req(purpose="brief", output_schema=Verdict))
    assert response.parsed == Verdict(real="A", confidence=0.7)
    assert Verdict.model_validate_json(response.text) == response.parsed
    judged = await fake.complete(req(purpose="judge", output_schema=Verdict))
    assert judged.parsed == Verdict(real="B", confidence=0.2)


async def test_fake_parsed_reply_mismatch_fails_loudly():
    class Other(BaseModel):
        x: int

    fake = FakeLLM([Other(x=1), Verdict(real="A", confidence=1.0), "not json"])
    with pytest.raises(AssertionError, match="Verdict"):
        await fake.complete(req(output_schema=Verdict))
    with pytest.raises(AssertionError, match="output_schema"):
        await fake.complete(req())
    with pytest.raises(AssertionError, match="Verdict"):
        await fake.complete(req(output_schema=Verdict))


async def test_fake_tool_calls_reply():
    fake = FakeLLM([[tool_call("search_self", query="Ana")], "done"])
    first = await fake.complete(req(tools=[SEARCH]))
    assert first.stop_reason == "tool_use"
    assert first.tool_calls[0].name == "search_self"
    assert first.message.tool_calls == first.tool_calls
    assert first.message.role == "assistant"
    second = await fake.complete(req(tools=[SEARCH]))
    assert second.stop_reason == "end_turn"
    assert tool_call("x").id != tool_call("x").id


async def test_fake_tool_call_to_unknown_tool_fails_loudly():
    fake = FakeLLM([[ToolCall(id="t1", name="read_page", input={})]])
    with pytest.raises(AssertionError, match="read_page"):
        await fake.complete(req(tools=[SEARCH]))


async def test_fake_exceptions_and_full_responses():
    fake = FakeLLM([LLMError("boom"), LLMResponse(text="raw", stop_reason="max_tokens")])
    with pytest.raises(LLMError, match="boom"):
        await fake.complete(req())
    response = await fake.complete(req())
    assert response.stop_reason == "max_tokens"
    assert response.message.text == "raw"


async def test_fake_reports_usage_and_model():
    fake = FakeLLM(["some words back"])
    response = await fake.complete(req("a question of some length", tier="bulk"))
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.model == "fake-bulk"


async def test_fake_complete_many_returns_errors_in_place():
    fake = FakeLLM(["a", LLMError("bad"), "c"])
    results = await fake.complete_many([req("1"), req("2"), req("3")], concurrency=1)
    assert results[0].text == "a"
    assert isinstance(results[1], LLMError)
    assert results[2].text == "c"
