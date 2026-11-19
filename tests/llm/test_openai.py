"""OpenAIClient against a stub SDK object: request building, response mapping, wiring.

No network and no key: every test passes a StubOpenAI as the client, and the
OpenRouter price fetch is replaced by a function that fails the test if called.
"""

import json

import openai
import pytest
from pydantic import BaseModel, Field

from repartee.config import EffortTiers, HomeConfig, ModelTiers
from repartee.distill.consolidate import ConsolidationResult
from repartee.distill.extract import ExtractionResult
from repartee.llm import CachedLLM, OpenAIClient, get_client, pricing
from repartee.llm.base import (
    Done,
    LLMError,
    LLMRefusal,
    LLMRequest,
    LLMUnavailable,
    Message,
    SystemBlock,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
)
from repartee.llm.openai_client import OPENROUTER_BASE_URL, strict_schema

from .stub_openai import (
    StubOpenAI,
    api_error,
    call,
    chunk,
    completion,
    connection_error,
    timeout_error,
)

GLM = "z-ai/glm-5.3-flash"

SEARCH = ToolSpec(
    name="search_memory",
    description="Search raw material.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "person": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["query", "person"],
        "additionalProperties": False,
    },
)


class Note(BaseModel):
    """A note."""

    text: str
    when: str | None = Field(default=None, description="YYYY.")
    confidence: float = 0.5
    tags: list[str] = Field(default_factory=list)


class Notes(BaseModel):
    notes: list[Note] = Field(default_factory=list)
    title: str


def client_with(*replies, provider="openrouter", models=None, effort=None):
    sdk = StubOpenAI(*replies)
    models = models or ModelTiers(bulk=GLM, smart=GLM, chat=GLM)
    client = OpenAIClient(models, effort or EffortTiers(), client=sdk, provider=provider)
    return client, sdk


def ask(text="hello", **kwargs) -> LLMRequest:
    return LLMRequest(messages=[Message(role="user", text=text)], **kwargs)


# ---------------------------------------------------------------------- requests


async def test_openai_sends_openrouter_routing_preferences():
    from repartee.config import HomeConfig, OpenRouterConfig

    assert OpenRouterConfig().preferences() == {}
    prefs = OpenRouterConfig(zdr=True, data_collection="deny").preferences()
    assert prefs == {"zdr": True, "data_collection": "deny"}
    assert HomeConfig.model_validate({"openrouter": {"zdr": True}}).openrouter.zdr

    sdk = StubOpenAI(completion("hi"))
    client = OpenAIClient(ModelTiers(bulk=GLM, smart=GLM, chat=GLM), EffortTiers(), client=sdk,
                          provider="openrouter", routing=prefs)
    await client.complete(ask())
    assert sdk.calls[0]["extra_body"]["provider"] == {"zdr": True, "data_collection": "deny"}
    client, sdk = client_with(completion("hi"))
    await client.complete(ask())
    assert "provider" not in sdk.calls[0]["extra_body"]


async def test_openai_stalled_host_hits_the_deadline():
    import asyncio

    client, sdk = client_with()

    async def stall(**kwargs):
        await asyncio.sleep(5)

    sdk.chat.completions.create = stall
    client.deadline_s = 0.05
    with pytest.raises(LLMError, match="no reply"):
        await client.complete(ask())


async def test_openai_effort_none_turns_reasoning_off():
    client, sdk = client_with(completion("hi"), effort=EffortTiers(chat="none"))
    await client.complete(ask(tier="chat"))
    assert sdk.calls[0]["extra_body"]["reasoning"] == {"effort": "none"}


async def test_openai_builds_request_from_tier():
    models = ModelTiers(bulk="a/bulk", smart="a/smart", chat="a/chat")
    effort = EffortTiers(bulk="low", smart="xhigh", chat="medium")
    client, sdk = client_with(
        completion("hi", model="a/smart"), completion("ok"), models=models, effort=effort
    )
    response = await client.complete(ask(tier="smart", max_tokens=1234))
    sent = sdk.calls[0]
    assert sent["model"] == "a/smart"
    assert sent["max_tokens"] == 1234
    assert sent["messages"] == [{"role": "user", "content": "hello"}]
    assert sent["extra_body"] == {"reasoning": {"effort": "xhigh"}}
    for absent in ("tools", "response_format", "tool_choice", "temperature", "extra_headers"):
        assert absent not in sent
    assert response.text == "hi"
    assert response.model == "a/smart"
    assert response.stop_reason == "end_turn"
    assert response.usage.input_tokens == 10 and response.usage.output_tokens == 5

    await client.complete(ask(tier="bulk"))
    assert sdk.calls[1]["model"] == "a/bulk"
    assert sdk.calls[1]["extra_body"] == {"reasoning": {"effort": "low"}}


async def test_openai_effort_levels_map_for_openrouter():
    from repartee.llm.openai_client import OPENROUTER_EFFORT

    assert set(OPENROUTER_EFFORT) == {"none", "low", "medium", "high", "xhigh", "max"}
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
    assert set(OPENROUTER_EFFORT.values()) <= allowed


async def test_openai_provider_sends_no_reasoning_and_uses_completion_tokens():
    client, sdk = client_with(completion("hi"), provider="openai")
    await client.complete(ask(max_tokens=999))
    sent = sdk.calls[0]
    assert "extra_body" not in sent
    assert sent["max_completion_tokens"] == 999
    assert "max_tokens" not in sent


async def test_openai_system_blocks_become_one_message():
    client, sdk = client_with(completion("ok"))
    system = [
        SystemBlock(text="You are me."),
        SystemBlock(text="core.md contents", cache=True),
        SystemBlock(text="Today is Tuesday."),
    ]
    await client.complete(ask(system=system))
    messages = sdk.calls[0]["messages"]
    assert messages[0] == {
        "role": "system",
        "content": "You are me.\n\ncore.md contents\n\nToday is Tuesday.",
    }
    assert messages[1] == {"role": "user", "content": "hello"}


async def test_openai_tool_round_trip_replays_raw():
    details = [{"type": "reasoning.text", "text": "Look it up.", "format": "unknown", "index": 0}]
    first = completion(
        "Let me look.",
        tool_calls=[
            call("call_1", "search_memory", {"query": "a", "person": None}),
            call("call_2", "search_memory", {"query": "b", "person": "Ana"}),
        ],
        finish_reason="tool_calls",
        reasoning="Look it up.",
        reasoning_details=details,
    )
    client, sdk = client_with(first, completion("Found it."))
    request = ask("what happened?", tools=[SEARCH], tier="chat")
    response = await client.complete(request)

    assert sdk.calls[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search_memory",
                "description": "Search raw material.",
                "parameters": SEARCH.input_schema,
                "strict": True,
            },
        }
    ]
    assert "tool_choice" not in sdk.calls[0]
    assert response.stop_reason == "tool_use"
    assert [c.id for c in response.tool_calls] == ["call_1", "call_2"]
    assert response.tool_calls[1].input == {"query": "b", "person": "Ana"}
    assert response.message.tool_calls == response.tool_calls
    assert response.text == "Let me look."

    follow_up = LLMRequest(
        tier="chat",
        tools=[SEARCH],
        messages=[
            *request.messages,
            response.message,
            Message(
                role="user",
                text="Keep it short.",
                tool_results=[
                    ToolResult(tool_call_id="call_1", content="chunk A"),
                    ToolResult(tool_call_id="call_2", content="nothing", is_error=True),
                ],
            ),
        ],
    )
    await client.complete(follow_up)
    sent = sdk.calls[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "tool", "tool", "user"]
    # The assistant turn is the provider's message, reasoning details included.
    assert sent[1] == response.message.raw
    assert sent[1]["reasoning_details"] == details
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == json.dumps(
        {"query": "a", "person": None}
    )
    assert "refusal" not in sent[1]
    assert sent[2] == {"role": "tool", "tool_call_id": "call_1", "content": "chunk A"}
    assert sent[3]["tool_call_id"] == "call_2"
    assert sent[3]["content"].startswith("Error")
    assert "nothing" in sent[3]["content"]
    assert sent[4] == {"role": "user", "content": "Keep it short."}
    # raw is plain JSON, so the disk cache can keep it
    json.dumps(response.message.raw)


async def test_openai_assistant_without_raw_is_rebuilt():
    client, sdk = client_with(completion("ok"))
    history = [
        Message(role="user", text="hi"),
        Message(
            role="assistant",
            text="Looking.",
            tool_calls=[ToolCall(id="t1", name="search_memory", input={"query": "x"})],
        ),
        Message(role="user", tool_results=[ToolResult(tool_call_id="t1", content="r")]),
    ]
    await client.complete(LLMRequest(messages=history, tools=[SEARCH]))
    sent = sdk.calls[0]["messages"]
    assert sent[1] == {
        "role": "assistant",
        "content": "Looking.",
        "tool_calls": [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "search_memory", "arguments": '{"query": "x"}'},
            }
        ],
    }
    assert sent[2] == {"role": "tool", "tool_call_id": "t1", "content": "r"}
    assert len(sent) == 3  # no empty user text after the tool results


async def test_openai_unparseable_tool_arguments_raise_llm_error():
    bad = completion(
        None, tool_calls=[call("c1", "search_memory", '{"query": ')], finish_reason="tool_calls"
    )
    not_object = completion(
        None, tool_calls=[call("c1", "search_memory", "[1, 2]")], finish_reason="tool_calls"
    )
    client, _ = client_with(bad, not_object)
    with pytest.raises(LLMError, match="search_memory"):
        await client.complete(ask(tools=[SEARCH]))
    with pytest.raises(LLMError, match="search_memory"):
        await client.complete(ask(tools=[SEARCH]))


async def test_openai_empty_tool_arguments_are_an_empty_object():
    reply = completion(None, tool_calls=[call("c1", "note", "")], finish_reason="tool_calls")
    client, _ = client_with(reply)
    response = await client.complete(ask(tools=[SEARCH]))
    assert response.tool_calls[0].input == {}
    assert response.text == ""


async def test_openai_tool_calls_with_stop_finish_are_still_tool_use():
    reply = completion(None, tool_calls=[call("c1", "search_memory", {"query": "x"})])
    client, _ = client_with(reply)
    response = await client.complete(ask(tools=[SEARCH]))
    assert response.stop_reason == "tool_use"


# ---------------------------------------------------------------------- structured output


def test_strict_schema_requires_everything_and_nulls_optionals():
    schema = strict_schema(Notes)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["notes", "title"]
    assert "$defs" not in schema and "$ref" not in json.dumps(schema)
    note = schema["properties"]["notes"]
    # optional with a default: nullable now
    assert {"type": "null"} in note["anyOf"]
    item = next(s for s in note["anyOf"] if s.get("type") == "array")["items"]
    assert item["additionalProperties"] is False
    assert item["required"] == ["text", "when", "confidence", "tags"]
    assert item["properties"]["text"] == {"type": "string"}
    assert {"type": "null"} in item["properties"]["confidence"]["anyOf"]
    assert item["properties"]["when"]["description"] == "YYYY."
    def no_dropped_keywords(node):
        if isinstance(node, dict):
            assert "default" not in node
            assert not isinstance(node.get("title"), str)  # a title keyword, not a property
            for value in node.values():
                no_dropped_keywords(value)
        elif isinstance(node, list):
            for value in node:
                no_dropped_keywords(value)

    no_dropped_keywords(schema)


def test_strict_schema_on_pipeline_models():
    for model in (ExtractionResult, ConsolidationResult):
        schema = strict_schema(model)
        text = json.dumps(schema)
        assert '"$ref"' not in text and '"default"' not in text

        def check(node):
            if isinstance(node, dict):
                if node.get("type") == "object" and "properties" in node:
                    assert node["additionalProperties"] is False
                    assert node["required"] == list(node["properties"])
                for value in node.values():
                    check(value)
            elif isinstance(node, list):
                for value in node:
                    check(value)

        check(schema)


async def test_openai_structured_output_parsed_with_null_defaults():
    reply = completion(
        '{"title": "t", "notes": [{"text": "I moved.", "when": null, "confidence": null, '
        '"tags": null}]}'
    )
    client, sdk = client_with(reply)
    response = await client.complete(ask(output_schema=Notes, tier="bulk"))
    fmt = sdk.calls[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "Notes"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == strict_schema(Notes)
    assert response.parsed == Notes(title="t", notes=[Note(text="I moved.")])
    assert response.parsed.notes[0].confidence == 0.5
    assert response.parsed.notes[0].tags == []


async def test_openai_structured_output_strips_code_fence():
    reply = completion('Here it is:\n```json\n{"title": "t", "notes": null}\n```\n')
    client, _ = client_with(reply)
    response = await client.complete(ask(output_schema=Notes))
    assert response.parsed == Notes(title="t")


async def test_openai_structured_output_retries_once_with_the_error():
    client, sdk = client_with(
        completion('{"notes": []}', usage={"prompt_tokens": 7, "completion_tokens": 3,
                                           "total_tokens": 10}),
        completion('{"title": "fixed", "notes": []}'),
    )
    response = await client.complete(ask("extract", output_schema=Notes))
    assert response.parsed == Notes(title="fixed")
    assert len(sdk.calls) == 2
    retry = sdk.calls[1]["messages"]
    assert retry[0] == {"role": "user", "content": "extract"}
    assert retry[1]["role"] == "assistant" and retry[1]["content"] == '{"notes": []}'
    assert retry[2]["role"] == "user"
    assert "title" in retry[2]["content"]  # the validation error names the field
    assert "JSON" in retry[2]["content"]
    assert sdk.calls[1]["response_format"] == sdk.calls[0]["response_format"]
    # usage covers both attempts
    assert response.usage.input_tokens == 17
    assert response.usage.output_tokens == 8


async def test_openai_structured_output_fails_after_retry():
    client, sdk = client_with(completion("not json"), completion("still not json"))
    with pytest.raises(LLMError, match="Notes"):
        await client.complete(ask(output_schema=Notes))
    assert len(sdk.calls) == 2


async def test_openai_schema_rejected_falls_back_to_json_mode():
    client, sdk = client_with(
        api_error(openai.BadRequestError, 400, "response_format json_schema not supported"),
        completion('{"title": "a"}'),
        completion('{"title": "b"}'),
    )
    response = await client.complete(ask(output_schema=Notes, system=[SystemBlock(text="Sys.")]))
    assert response.parsed == Notes(title="a")
    fallback = sdk.calls[1]
    assert fallback["response_format"] == {"type": "json_object"}
    system = fallback["messages"][0]
    assert system["role"] == "system"
    assert system["content"].startswith("Sys.")
    assert json.dumps(strict_schema(Notes)) in system["content"]
    # The model is remembered: the next structured request goes straight to JSON mode.
    await client.complete(ask(output_schema=Notes))
    assert sdk.calls[2]["response_format"] == {"type": "json_object"}
    assert len(sdk.calls) == 3


async def test_openai_non_schema_bad_request_is_an_error():
    client, _ = client_with(api_error(openai.BadRequestError, 400, "context too long"))
    with pytest.raises(LLMError, match="context too long") as info:
        await client.complete(ask())
    assert not isinstance(info.value, LLMUnavailable)


async def test_openai_structured_cut_off_twice_raises():
    cut = completion('{"title": ', finish_reason="length")
    client, sdk = client_with(cut, cut)
    with pytest.raises(LLMError, match="max_tokens"):
        await client.complete(ask(output_schema=Notes, max_tokens=1000))
    assert [c["max_tokens"] for c in sdk.calls] == [1000, 2000]


async def test_openai_structured_cut_off_retries_with_more_room_and_counts_both():
    priced = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.25}
    client, sdk = client_with(
        completion('{"title": ', finish_reason="length", usage=priced),
        completion('{"title": "done"}', usage=priced),
    )
    response = await client.complete(ask(output_schema=Notes, max_tokens=40_000))
    assert response.parsed == Notes(title="done")
    assert [c["max_tokens"] for c in sdk.calls] == [40_000, 64_000]
    assert response.usage.output_tokens == 10
    assert response.usage.cost_usd == pytest.approx(0.5)


async def test_openai_usage_cost_is_read_when_given():
    client, _ = client_with(completion("hi", usage={"prompt_tokens": 3, "completion_tokens": 1,
                                                    "total_tokens": 4, "cost": 0.0002}))
    assert (await client.complete(ask())).usage.cost_usd == pytest.approx(0.0002)
    client, _ = client_with(completion("hi"))
    assert (await client.complete(ask())).usage.cost_usd is None


# ---------------------------------------------------------------------- stop reasons, usage


async def test_openai_length_is_max_tokens_and_drops_cut_tool_calls():
    cut = completion(
        "Partial",
        tool_calls=[call("c9", "search_memory", '{"query": "x"')],
        finish_reason="length",
    )
    client, _ = client_with(cut)
    response = await client.complete(ask(tools=[SEARCH]))
    assert response.stop_reason == "max_tokens"
    assert response.tool_calls == []
    assert "tool_calls" not in response.message.raw
    assert response.text == "Partial"


async def test_openai_content_filter_and_refusal_raise():
    client, _ = client_with(
        completion("", finish_reason="content_filter"),
        completion(None, refusal="I can't help with that."),
        [chunk({"role": "assistant", "content": ""}), chunk({}, finish_reason="content_filter")],
    )
    with pytest.raises(LLMRefusal):
        await client.complete(ask())
    with pytest.raises(LLMRefusal, match="can't help"):
        await client.complete(ask())
    with pytest.raises(LLMRefusal):
        [e async for e in client.stream(ask())]


async def test_openai_usage_maps_cached_tokens():
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 50,
        "total_tokens": 1050,
        "prompt_tokens_details": {"cached_tokens": 800, "cache_write_tokens": 100},
    }
    client, _ = client_with(completion("ok", usage=usage))
    response = await client.complete(ask())
    assert response.usage.input_tokens == 100
    assert response.usage.cache_read_tokens == 800
    assert response.usage.cache_write_tokens == 100
    assert response.usage.output_tokens == 50


async def test_openai_no_choices_is_an_error():
    from openai.types.chat import ChatCompletion

    empty = ChatCompletion.model_validate(
        {"id": "x", "object": "chat.completion", "created": 0, "model": GLM, "choices": [],
         "error": {"message": "upstream exploded", "code": 502}}
    )
    client, _ = client_with(empty)
    with pytest.raises(LLMError, match="upstream exploded"):
        await client.complete(ask())


# ---------------------------------------------------------------------- errors


async def test_openai_errors_map():
    client, _ = client_with(
        api_error(openai.AuthenticationError, 401),
        api_error(openai.APIStatusError, 402, "Insufficient credits"),
        connection_error(),
        timeout_error(),
        api_error(openai.RateLimitError, 429, "slow down"),
        api_error(openai.InternalServerError, 500),
        api_error(openai.PermissionDeniedError, 403, "key disabled"),
    )
    with pytest.raises(LLMUnavailable, match="OPENROUTER_API_KEY") as info:
        await client.complete(ask())
    assert "\n" not in str(info.value)
    with pytest.raises(LLMUnavailable, match="credit"):
        await client.complete(ask())
    with pytest.raises(LLMUnavailable, match="network"):
        await client.complete(ask())
    with pytest.raises(LLMUnavailable, match="timed out"):
        await client.complete(ask())
    with pytest.raises(LLMError, match="429") as info:
        await client.complete(ask())
    assert not isinstance(info.value, LLMUnavailable)
    with pytest.raises(LLMError, match="500") as info:
        await client.complete(ask())
    assert not isinstance(info.value, LLMUnavailable)
    with pytest.raises(LLMUnavailable, match="OPENROUTER_API_KEY"):
        await client.complete(ask())


async def test_openai_provider_errors_name_its_variable():
    client, _ = client_with(api_error(openai.AuthenticationError, 401), provider="openai")
    with pytest.raises(LLMUnavailable, match="OPENAI_API_KEY"):
        await client.complete(ask())


async def test_openai_auth_error_in_stream_maps_to_unavailable():
    client, _ = client_with(api_error(openai.AuthenticationError, 401))
    with pytest.raises(LLMUnavailable):
        [e async for e in client.stream(ask())]


# ---------------------------------------------------------------------- streaming


def tool_stream() -> list:
    return [
        chunk({"role": "assistant", "content": "", "reasoning_details": [
            {"type": "reasoning.text", "text": "Need ", "format": "unknown", "index": 0}]}),
        chunk({"content": "", "reasoning_details": [
            {"type": "reasoning.text", "text": "a search.", "format": "unknown", "index": 0}]}),
        chunk({"content": "I went "}),
        chunk({"content": "back and forth."}),
        chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                               "function": {"name": "search_memory", "arguments": ""}}]}),
        chunk({"tool_calls": [{"index": 1, "id": "call_2", "type": "function",
                               "function": {"name": "search_memory", "arguments": '{"qu'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"query": "a", '}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"person": null}'}}]}),
        chunk({"tool_calls": [{"index": 1, "function": {"arguments": 'ery": "b", "person": "Ana"}'}}
                              ]}),
        chunk({}, finish_reason="tool_calls"),
        chunk(None, usage={"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52}),
    ]


async def test_openai_stream_yields_deltas_tools_done():
    client, sdk = client_with(tool_stream())
    events = [e async for e in client.stream(ask(tools=[SEARCH], tier="chat"))]
    sent = sdk.calls[0]
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    deltas = [e for e in events if isinstance(e, TextDelta)]
    assert [d.text for d in deltas] == ["I went ", "back and forth."]
    tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
    assert [e.call.id for e in tool_events] == ["call_1", "call_2"]
    assert tool_events[0].call.input == {"query": "a", "person": None}
    assert tool_events[1].call.input == {"query": "b", "person": "Ana"}
    assert isinstance(events[-1], Done)
    assert sum(isinstance(e, Done) for e in events) == 1
    assert events.index(tool_events[0]) > events.index(deltas[-1])
    response = events[-1].response
    assert response.stop_reason == "tool_use"
    assert response.text == "I went back and forth."
    assert response.usage.input_tokens == 40 and response.usage.output_tokens == 12
    assert response.model == GLM
    raw = response.message.raw
    assert raw["reasoning_details"] == [
        {"type": "reasoning.text", "text": "Need a search.", "format": "unknown", "index": 0}
    ]
    assert raw["tool_calls"][1]["function"]["arguments"] == '{"query": "b", "person": "Ana"}'


async def test_openai_stream_done_matches_complete():
    whole = completion(
        "I went back and forth.",
        tool_calls=[
            call("call_1", "search_memory", '{"query": "a", "person": null}'),
            call("call_2", "search_memory", '{"query": "b", "person": "Ana"}'),
        ],
        finish_reason="tool_calls",
        usage={"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
        reasoning_details=[
            {"type": "reasoning.text", "text": "Need a search.", "format": "unknown", "index": 0}
        ],
    )
    client, _ = client_with(whole, tool_stream())
    completed = await client.complete(ask(tools=[SEARCH]))
    streamed = [e async for e in client.stream(ask(tools=[SEARCH]))][-1].response
    assert streamed.model_dump() == completed.model_dump()
    assert streamed.message.raw == completed.message.raw


async def test_openai_stream_with_output_schema_still_parses():
    client, _ = client_with(completion('{"title": "t"}'))
    events = [e async for e in client.stream(ask(output_schema=Notes))]
    assert events[-1].response.parsed == Notes(title="t")


async def test_openai_stream_error_chunk_raises():
    client, _ = client_with(
        [chunk({"content": "Hal"}), chunk({}, finish_reason="error",
                                          error={"message": "provider died", "code": 502})]
    )
    with pytest.raises(LLMError, match="provider died"):
        [e async for e in client.stream(ask())]


# ---------------------------------------------------------------------- get_client


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    import os

    for name in list(os.environ):
        if name.startswith(("OPENAI_", "OPENROUTER_", "ANTHROPIC_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))

    def no_network(*args, **kwargs):
        raise AssertionError("the default test run must not fetch prices")

    monkeypatch.setattr(pricing, "fetch_openrouter_models", no_network)


def openrouter_config(**kwargs) -> HomeConfig:
    return HomeConfig(provider="openrouter", models=ModelTiers(bulk=GLM, smart=GLM, chat=GLM),
                      **kwargs)


def test_get_client_openrouter_without_key_names_the_variable(clean_env):
    with pytest.raises(LLMUnavailable) as info:
        get_client(openrouter_config())
    assert "OPENROUTER_API_KEY" in str(info.value)
    assert "\n" not in str(info.value)


def test_get_client_openrouter_with_key(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-not-real")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://elsewhere.invalid/v1")
    client = get_client(openrouter_config())
    assert isinstance(client, OpenAIClient)
    assert client.provider == "openrouter"
    assert str(client.sdk.base_url).rstrip("/") == OPENROUTER_BASE_URL
    assert client.sdk.api_key == "sk-or-v1-not-real"
    for header in ("HTTP-Referer", "X-Title"):
        assert header not in client.sdk.default_headers


def test_get_client_openrouter_rejects_anthropic_model_ids(clean_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-not-real")
    config = HomeConfig(provider="openrouter")  # tiers still claude-opus-5-5
    with pytest.raises(LLMUnavailable) as info:
        get_client(config)
    text = str(info.value)
    assert "models:" in text and "repartee.yaml" in text and GLM in text
    assert "claude-opus-5-5" in text
    assert "\n" not in text


def test_get_client_openai_reads_its_own_environment(clean_env, monkeypatch):
    config = HomeConfig(provider="openai", models=ModelTiers(bulk="m", smart="m", chat="m"))
    with pytest.raises(LLMUnavailable) as info:
        get_client(config)
    assert "OPENAI_API_KEY" in str(info.value)
    assert "\n" not in str(info.value)
    monkeypatch.setenv("OPENAI_API_KEY", "local")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
    client = get_client(config)
    assert isinstance(client, OpenAIClient)
    assert client.provider == "openai"
    assert str(client.sdk.base_url).startswith("http://127.0.0.1:8080/v1")


def test_get_client_openrouter_cached_and_registers_prices(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-not-real")
    monkeypatch.setattr(pricing, "PRICES", dict(pricing.PRICES))
    calls = []

    def fake_fetch(timeout):
        calls.append(timeout)
        prices = {"prompt": "0.00000004", "completion": "0.00000008"}
        return {"data": [{"id": GLM, "pricing": prices}]}

    monkeypatch.setattr(pricing, "fetch_openrouter_models", fake_fetch)
    cached = get_client(openrouter_config(), cache_dir=tmp_path / "cache")
    assert isinstance(cached, CachedLLM)
    assert isinstance(cached.inner, OpenAIClient)
    assert pricing.PRICES[GLM] == pytest.approx((0.04, 0.08))
    assert cached.model_for("bulk") != cached.model_for("smart")  # effort differs per tier
    assert len(calls) == 1


def test_get_client_openrouter_registers_prices_even_without_key(clean_env, monkeypatch,
                                                                  tmp_path):
    # `distill --dry-run` needs the estimate before anyone has a key.
    monkeypatch.setattr(pricing, "PRICES", dict(pricing.PRICES))
    monkeypatch.setattr(
        pricing, "fetch_openrouter_models",
        lambda timeout: {"data": [{"id": GLM, "pricing": {"prompt": "1e-7", "completion": "0"}}]},
    )
    with pytest.raises(LLMUnavailable):
        get_client(openrouter_config(), cache_dir=tmp_path / "cache")
    assert pricing.is_priced(GLM)


def test_get_client_anthropic_unchanged(clean_env, monkeypatch, tmp_path):
    from repartee.llm import AnthropicClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    cached = get_client(HomeConfig(), cache_dir=tmp_path / "cache")
    assert isinstance(cached.inner, AnthropicClient)
    assert cached.model_for("bulk") == "claude-opus-5-5@low"
