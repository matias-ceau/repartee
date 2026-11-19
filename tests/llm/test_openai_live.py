"""Real calls to OpenRouter with z-ai/glm-5.3-flash. Skipped by default; run with ``-m live``.

Needs OPENROUTER_API_KEY. Every text sent is invented. About a dozen requests in all,
each a fraction of a cent. Each test prints what the client had to do (schema
fallback, a retry) so ``-s`` shows how the model behaves.
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest

from repartee.config import EffortTiers, HomeConfig, ModelTiers, OwnerConfig
from repartee.distill.extract import ExtractionResult, build_request
from repartee.llm import OpenAIClient, get_client
from repartee.llm.base import (
    Done,
    LLMRequest,
    Message,
    TextDelta,
    ToolCallEvent,
    ToolResult,
)
from repartee.mirror.tools import Toolbox
from repartee.schema import Chunk, Item, Kind

GLM = "z-ai/glm-5.3-flash"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="no OPENROUTER_API_KEY"),
]


def live_client(**effort) -> OpenAIClient:
    config = HomeConfig(
        provider="openrouter",
        models=ModelTiers(bulk=GLM, smart=GLM, chat=GLM),
        effort=EffortTiers(**effort),
    )
    client = get_client(config)
    assert isinstance(client, OpenAIClient)
    return client


def record(client: OpenAIClient) -> list[dict]:
    """Keep the parameters of every request the client sends."""
    sent: list[dict] = []
    completions = client.sdk.chat.completions
    original = completions.create

    async def create(**kwargs):
        sent.append(kwargs)
        return await original(**kwargs)

    completions.create = create
    return sent


def describe(sent: list[dict]) -> str:
    formats = [s.get("response_format", {}).get("type", "-") for s in sent]
    return f"{len(sent)} request(s), response formats {formats}"


SPECS = {spec.name: spec for spec in Toolbox(None, None).specs("chat")}  # type: ignore[arg-type]
TOOLS = [SPECS["search_self"], SPECS["search_memory"]]
SYSTEM_TOOLS = (
    "You are the memory of a fictional person named Jo. You know nothing about Jo except "
    "what the tools return. Always look things up with a tool before answering."
)


async def test_openai_live_plain_completion():
    client = live_client()
    response = await client.complete(
        LLMRequest(
            tier="bulk",
            purpose="live-test",
            max_tokens=2000,
            messages=[Message(role="user", text="Reply with the single word: pong")],
        )
    )
    print(f"\nplain: {response.text!r} {response.usage} model={response.model}")
    assert "pong" in response.text.lower()
    assert response.stop_reason == "end_turn"
    assert response.usage.output_tokens > 0
    assert response.model.startswith("z-ai/")


def invented_chat() -> tuple[Chunk, list[Item]]:
    lines = [
        ("Sam", False, "10:02", "did you sign the lease in Porto in the end?"),
        ("Jo", True, "10:05", "yes! moving on 1 March, the flat near the river"),
        ("Jo", True, "10:06", "still teaching cello on weekends, I'll keep that"),
        ("Sam", False, "10:08", "and the cat?"),
        ("Jo", True, "10:09", "Miso comes with me obviously, she hates trains though"),
    ]
    items = [
        Item(
            id=f"wa-test{i}",
            source="whatsapp",
            kind=Kind.MESSAGE,
            container="Sam",
            author=author,
            is_self=is_self,
            timestamp=datetime.fromisoformat(f"2031-01-14T{clock}"),
            text=text,
            seq=i,
        )
        for i, (author, is_self, clock, text) in enumerate(lines)
    ]
    chunk = Chunk(
        id="ch-test",
        source="whatsapp",
        kind=Kind.MESSAGE,
        container="Sam",
        item_ids=[item.id for item in items],
        start=items[0].timestamp,
        end=items[-1].timestamp,
        participants=["Sam", "Jo"],
        text="\n".join(f"{i.author}: {i.text}" for i in items),
    )
    return chunk, items


async def test_openai_live_structured_extraction():
    client = live_client()
    sent = record(client)
    chunk, items = invented_chat()
    request = build_request(chunk, items, OwnerConfig(name="Jo"))
    response = await client.complete(request)
    result = response.parsed
    print(f"\nextract: {describe(sent)}, json mode models {client.json_mode_models}")
    print(f"extract: {result.model_dump_json(indent=1) if result else None}")
    assert isinstance(result, ExtractionResult)
    assert result.observations
    ids = {item.id for item in items}
    cited = {e for o in result.observations for e in o.evidence}
    assert cited & ids, "no observation cites a real line id"
    assert any("porto" in o.text.lower() for o in result.observations)


async def test_openai_live_tool_round_trip():
    client = live_client()
    sent = record(client)
    question = Message(
        role="user", text="When did I move to Porto? Search the raw material for it."
    )
    first = await client.complete(
        LLMRequest(tier="chat", purpose="chat", system=[], messages=[question], tools=TOOLS)
    )
    print(f"\ntools 1: stop={first.stop_reason} calls={first.tool_calls} text={first.text!r}")
    print(f"tools 1 raw keys: {sorted(first.message.raw)}")
    assert first.stop_reason == "tool_use"
    assert first.tool_calls
    call = first.tool_calls[0]
    assert call.name in {"search_self", "search_memory"}
    assert isinstance(call.input.get("query"), str) and call.input["query"]
    # strict schemas: every property present, optional ones may be null
    spec = SPECS[call.name]
    assert set(call.input) == set(spec.input_schema["properties"])

    results = [
        ToolResult(
            tool_call_id=c.id,
            content="[wa-test1] 2031-01-14 Jo: yes! moving on 1 March, the flat near the river"
            if c.name == "search_memory"
            else "No claims match.",
        )
        for c in first.tool_calls
    ]
    second = await client.complete(
        LLMRequest(
            tier="chat",
            purpose="chat",
            messages=[question, first.message, Message(role="user", tool_results=results)],
            tools=TOOLS,
        )
    )
    print(f"tools 2: stop={second.stop_reason} text={second.text!r} {describe(sent)}")
    replayed = sent[1]["messages"][1]
    assert replayed == first.message.raw
    if second.stop_reason == "tool_use":  # it may want one more lookup; that is fine
        assert second.tool_calls
    else:
        assert "march" in second.text.lower()


async def test_openai_live_stream_with_tool_call():
    client = live_client()
    question = Message(role="user", text="Who is Miso? Look it up in the raw material.")
    request = LLMRequest(
        tier="chat",
        purpose="chat",
        system=[],
        messages=[question],
        tools=TOOLS,
    )
    events = [e async for e in client.stream(request)]
    deltas = [e for e in events if isinstance(e, TextDelta)]
    calls = [e.call for e in events if isinstance(e, ToolCallEvent)]
    assert isinstance(events[-1], Done) and sum(isinstance(e, Done) for e in events) == 1
    done = events[-1].response
    print(f"\nstream 1: {len(deltas)} deltas, calls={calls}, usage={done.usage}")
    assert calls and done.stop_reason == "tool_use"
    assert done.tool_calls == calls
    assert "".join(d.text for d in deltas) == done.text
    assert done.usage.output_tokens > 0

    results = [
        ToolResult(
            tool_call_id=c.id,
            content="[wa-test4] 2031-01-14 Jo: Miso comes with me obviously, she hates trains"
            if c.name == "search_memory"
            else "No claims match.",
        )
        for c in calls
    ]
    follow = request.model_copy(
        update={
            "messages": [question, done.message, Message(role="user", tool_results=results)]
        }
    )
    events = [e async for e in client.stream(follow)]
    deltas = [e for e in events if isinstance(e, TextDelta)]
    final = events[-1].response
    print(f"stream 2: {len(deltas)} deltas, stop={final.stop_reason}, text={final.text!r}")
    if final.stop_reason == "end_turn":
        assert deltas and "".join(d.text for d in deltas) == final.text
        assert "cat" in final.text.lower()
