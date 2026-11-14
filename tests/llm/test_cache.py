"""CachedLLM: the disk cache used by the pipeline stages."""

import json

import pytest
from pydantic import BaseModel

from repartee.llm import CachedLLM, FakeLLM, tool_call
from repartee.llm.base import LLMError, LLMRequest, LLMResponse, Message, SystemBlock, ToolSpec


class Obs(BaseModel):
    items: list[str]


class Other(BaseModel):
    items: list[str]
    extra: int = 0


TOOL = ToolSpec(
    name="search_self",
    description="Search.",
    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
)


def req(text="chunk", **kwargs) -> LLMRequest:
    kwargs.setdefault("purpose", "extract")
    return LLMRequest(
        system=[SystemBlock(text="Extract observations.", cache=True)],
        messages=[Message(role="user", text=text)],
        **kwargs,
    )


async def test_cache_hit_skips_inner_and_restores_parsed(tmp_path):
    fake = FakeLLM([Obs(items=["I moved to Lyon."])])
    cached = CachedLLM(fake, tmp_path)
    first = await cached.complete(req(output_schema=Obs))
    assert (cached.hits, cached.misses) == (0, 1)

    again = CachedLLM(fake, tmp_path)  # a fresh process reading the same directory
    second = await again.complete(req(output_schema=Obs))
    assert (again.hits, again.misses) == (1, 0)
    assert len(fake.requests) == 1
    assert isinstance(second.parsed, Obs)
    assert second.parsed == first.parsed
    assert second.text == first.text
    assert second.usage == first.usage
    assert list(tmp_path.rglob("*.json"))


async def test_cache_keeps_tool_calls_and_raw(tmp_path):
    raw = [{"type": "thinking", "thinking": "", "signature": "s"}]
    calls = [tool_call("search_self", query="x")]
    fake = FakeLLM(
        [
            LLMResponse(
                tool_calls=calls,
                stop_reason="tool_use",
                message=Message(role="assistant", tool_calls=calls, raw=raw),
            )
        ]
    )
    cached = CachedLLM(fake, tmp_path)
    await cached.complete(req(tools=[TOOL]))
    hit = await cached.complete(req(tools=[TOOL]))
    assert cached.hits == 1
    assert hit.stop_reason == "tool_use"
    assert hit.tool_calls[0].input == {"query": "x"}
    assert hit.message.raw == raw


async def test_cache_key_changes_with_messages_schema_tier(tmp_path):
    fake = FakeLLM()
    fake.on("extract", lambda r: Obs(items=[]) if r.output_schema is Obs else "plain")
    fake.on(lambda r: r.output_schema is Other, Other(items=[]))
    cached = CachedLLM(fake, tmp_path)
    base = req()
    variants = [
        base,
        req(text="another chunk"),
        req(output_schema=Obs),
        req(output_schema=Other),
        req(tier="bulk"),
        req(purpose="route"),
        req(tools=[TOOL]),
    ]
    fake.on("route", "routed")
    for request in variants:
        await cached.complete(request)
    assert cached.misses == len(variants)
    await cached.complete(req())
    assert cached.hits == 1


async def test_cache_key_uses_model_for_tier(tmp_path):
    fake = FakeLLM()
    fake.on("extract", "x")
    same_model = CachedLLM(fake, tmp_path, model_for=lambda tier: "claude-opus-5-5")
    await same_model.complete(req(tier="bulk"))
    await same_model.complete(req(tier="smart"))
    assert (same_model.hits, same_model.misses) == (1, 1)


async def test_cache_does_not_store_errors(tmp_path):
    fake = FakeLLM([LLMError("rate limited"), "fine"])
    cached = CachedLLM(fake, tmp_path)
    with pytest.raises(LLMError):
        await cached.complete(req())
    assert not list(tmp_path.rglob("*.json"))
    assert (await cached.complete(req())).text == "fine"
    assert (await cached.complete(req())).text == "fine"
    assert cached.hits == 1


async def test_cache_ignores_corrupt_entries(tmp_path):
    fake = FakeLLM(["one", "two"])
    cached = CachedLLM(fake, tmp_path)
    await cached.complete(req())
    for path in tmp_path.rglob("*.json"):
        path.write_text("{not json", encoding="utf-8")
    assert (await cached.complete(req())).text == "two"


async def test_cache_complete_many_only_sends_misses(tmp_path):
    fake = FakeLLM()
    fake.on("extract", lambda r: f"seen {r.messages[-1].text}")
    cached = CachedLLM(fake, tmp_path)
    await cached.complete(req("a"))
    results = await cached.complete_many([req("a"), req("b"), req("c")], batch=True)
    assert [r.text for r in results] == ["seen a", "seen b", "seen c"]
    assert [r.messages[-1].text for r in fake.requests] == ["a", "b", "c"]
    assert cached.hits == 1


async def test_cache_stream_passes_through(tmp_path):
    fake = FakeLLM(["streamed text"])
    cached = CachedLLM(fake, tmp_path)
    events = [e async for e in cached.stream(req())]
    assert events[-1].response.text == "streamed text"
    assert not list(tmp_path.rglob("*.json"))


def test_cache_files_are_plain_json(tmp_path):
    # Entries must be readable JSON so a person can inspect or delete them.
    import asyncio

    cached = CachedLLM(FakeLLM(["hello"]), tmp_path)
    asyncio.run(cached.complete(req()))
    [path] = list(tmp_path.rglob("*.json"))
    assert json.loads(path.read_text(encoding="utf-8"))["text"] == "hello"
