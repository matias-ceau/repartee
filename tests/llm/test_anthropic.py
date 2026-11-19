"""AnthropicClient against a stub SDK object: request building and response mapping.

No network and no key: every test passes a StubSDK as the client.
"""

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from repartee.config import EffortTiers, HomeConfig, ModelTiers
from repartee.llm import AnthropicClient, CachedLLM, get_client
from repartee.llm.anthropic_client import FALLBACK_BETA
from repartee.llm.base import (
    Done,
    LLMError,
    LLMRefusal,
    LLMRequest,
    LLMUnavailable,
    Message,
    SystemBlock,
    TextDelta,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
)

from .stub_sdk import StubSDK, message, text_message

SEARCH = ToolSpec(
    name="search_memory",
    description="Search raw material.",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
)


class Observations(BaseModel):
    items: list[str]


def client_with(*replies, models=None, effort=None) -> tuple[AnthropicClient, StubSDK]:
    sdk = StubSDK(*replies)
    client = AnthropicClient(
        models or ModelTiers(), effort or EffortTiers(), client=sdk, poll_interval=0
    )
    return client, sdk


def ask(text="hello", **kwargs) -> LLMRequest:
    return LLMRequest(messages=[Message(role="user", text=text)], **kwargs)


def api_error(cls, status: int, text: str = "nope"):
    request = httpx2.Request("POST", "https://api.invalid/v1/messages")
    response = httpx2.Response(status, request=request)
    return cls(text, response=response, body=None)


async def test_anthropic_builds_request_from_tier():
    models = ModelTiers(bulk="claude-sonnet-5-5", smart="claude-opus-5-5", chat="claude-opus-5-5")
    effort = EffortTiers(bulk="low", smart="high", chat="medium")
    client, sdk = client_with(text_message("hi"), text_message("hi"), models=models, effort=effort)

    response = await client.complete(ask(tier="smart", max_tokens=1234))
    sent = sdk.calls[0]
    assert sent["model"] == "claude-opus-5-5"
    assert sent["max_tokens"] == 1234
    assert sent["output_config"] == {"effort": "high"}
    assert sent["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    for absent in ("thinking", "temperature", "top_p", "top_k", "tool_choice", "system", "tools"):
        assert absent not in sent
    # Server-side refusal fallback, as prescribed for claude-opus-5-5.
    assert sent["betas"] == [FALLBACK_BETA]
    assert sent["fallbacks"] == "default"
    assert response.text == "hi"
    assert response.model == "claude-opus-5-5"
    assert response.usage.input_tokens == 10

    await client.complete(ask(tier="bulk"))
    bulk = sdk.calls[1]
    assert bulk["model"] == "claude-sonnet-5-5"
    assert bulk["output_config"] == {"effort": "low"}


async def test_anthropic_complete_streams_large_max_tokens():
    # The SDK refuses non-streaming calls that may run past ten minutes.
    client, sdk = client_with(text_message("short"), text_message("long"))
    assert (await client.complete(ask(max_tokens=16000))).text == "short"
    assert "_streamed" not in sdk.calls[0]
    assert (await client.complete(ask(max_tokens=64000))).text == "long"
    assert sdk.calls[1]["_streamed"] is True


async def test_anthropic_no_fallback_for_models_without_it():
    models = ModelTiers(bulk="claude-haiku-5-5")
    client, sdk = client_with(text_message("ok"), models=models)
    await client.complete(ask(tier="bulk"))
    assert "fallbacks" not in sdk.calls[0]
    assert "betas" not in sdk.calls[0]


async def test_anthropic_system_cache_blocks():
    client, sdk = client_with(text_message("ok"), text_message("ok"))
    system = [
        SystemBlock(text="You are me.", cache=False),
        SystemBlock(text="core.md contents", cache=True),
        SystemBlock(text="Today is Tuesday."),
    ]
    await client.complete(ask(system=system))
    assert sdk.calls[0]["system"] == [
        {"type": "text", "text": "You are me."},
        {"type": "text", "text": "core.md contents", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Today is Tuesday."},
    ]
    # A single-turn request does not pay for a conversation cache write.
    assert "cache_control" not in sdk.calls[0]

    conversation = [
        Message(role="user", text="hi"),
        Message(role="assistant", text="hello"),
        Message(role="user", text="and?"),
    ]
    await client.complete(LLMRequest(system=system, messages=conversation))
    assert sdk.calls[1]["cache_control"] == {"type": "ephemeral"}


async def test_anthropic_caps_cache_breakpoints():
    client, sdk = client_with(text_message("ok"))
    system = [SystemBlock(text=f"block {i}", cache=True) for i in range(6)]
    conversation = [
        Message(role="user", text="hi"),
        Message(role="assistant", text="hello"),
        Message(role="user", text="and?"),
    ]
    await client.complete(LLMRequest(system=system, messages=conversation))
    sent = sdk.calls[0]
    marked = [b for b in sent["system"] if "cache_control" in b]
    assert len(marked) + 1 <= 4  # plus the conversation breakpoint
    assert sent["system"][-1].get("cache_control")  # the latest stable prefix is kept


async def test_anthropic_tool_round_trip_replays_raw():
    first = message(
        [
            {"type": "thinking", "thinking": "", "signature": "sig-1"},
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "id": "toolu_1", "name": "search_memory", "input": {"query": "a"}},
            {"type": "tool_use", "id": "toolu_2", "name": "search_memory", "input": {"query": "b"}},
        ],
        stop_reason="tool_use",
    )
    client, sdk = client_with(first, text_message("Found it."))
    request = ask("what happened?", tools=[SEARCH], tier="chat")
    response = await client.complete(request)

    assert sdk.calls[0]["tools"] == [
        {
            "name": "search_memory",
            "description": "Search raw material.",
            "input_schema": SEARCH.input_schema,
            "strict": True,
        }
    ]
    assert "tool_choice" not in sdk.calls[0]
    assert response.stop_reason == "tool_use"
    assert [c.id for c in response.tool_calls] == ["toolu_1", "toolu_2"]
    assert response.tool_calls[1].input == {"query": "b"}
    assert response.message.role == "assistant"
    assert response.message.tool_calls == response.tool_calls

    follow_up = LLMRequest(
        tier="chat",
        tools=[SEARCH],
        messages=[
            *request.messages,
            response.message,
            Message(
                role="user",
                tool_results=[
                    ToolResult(tool_call_id="toolu_1", content="chunk A"),
                    ToolResult(tool_call_id="toolu_2", content="nothing", is_error=True),
                ],
            ),
        ],
    )
    await client.complete(follow_up)
    sent = sdk.calls[1]["messages"]
    assert len(sent) == 3
    # The assistant turn is the provider's content, thinking block and signature included.
    assert sent[1]["role"] == "assistant"
    assert sent[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-1"}
    assert [b["type"] for b in sent[1]["content"]] == ["thinking", "text", "tool_use", "tool_use"]
    # Both tool results travel together in one user message.
    assert sent[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "chunk A"},
            {
                "type": "tool_result",
                "tool_use_id": "toolu_2",
                "content": "nothing",
                "is_error": True,
            },
        ],
    }


async def test_anthropic_assistant_without_raw_is_rebuilt():
    client, sdk = client_with(text_message("ok"))
    from repartee.llm.base import ToolCall

    history = [
        Message(role="user", text="hi"),
        Message(
            role="assistant",
            text="Looking.",
            tool_calls=[ToolCall(id="t1", name="search_memory", input={"query": "x"})],
        ),
        Message(
            role="user", tool_results=[ToolResult(tool_call_id="t1", content="r")], text="go on"
        ),
    ]
    await client.complete(LLMRequest(messages=history, tools=[SEARCH]))
    sent = sdk.calls[0]["messages"]
    assert sent[1]["content"] == [
        {"type": "text", "text": "Looking."},
        {"type": "tool_use", "id": "t1", "name": "search_memory", "input": {"query": "x"}},
    ]
    assert sent[2]["content"][0]["type"] == "tool_result"
    assert sent[2]["content"][1] == {"type": "text", "text": "go on"}


async def test_anthropic_structured_output_parsed():
    reply = message(
        [
            {"type": "thinking", "thinking": "", "signature": "s"},
            {"type": "text", "text": '{"items": ["I moved to Lyon in 2014."]}'},
        ]
    )
    client, sdk = client_with(reply)
    response = await client.complete(ask(output_schema=Observations, tier="bulk"))
    sent = sdk.calls[0]
    assert sent["output_config"]["effort"] == "low"
    fmt = sent["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["type"] == "object"
    assert fmt["schema"]["additionalProperties"] is False
    assert "items" in fmt["schema"]["required"]
    assert response.parsed == Observations(items=["I moved to Lyon in 2014."])


async def test_anthropic_structured_output_invalid_json_raises():
    client, _ = client_with(text_message("not json at all"))
    with pytest.raises(LLMError, match="Observations"):
        await client.complete(ask(output_schema=Observations))


async def test_anthropic_max_tokens_is_reported_and_drops_cut_tool_calls():
    cut = message(
        [
            {"type": "text", "text": "Partial"},
            {"type": "tool_use", "id": "toolu_9", "name": "search_memory", "input": {}},
        ],
        stop_reason="max_tokens",
    )
    client, _ = client_with(cut, text_message('{"items": [', stop_reason="max_tokens"))
    response = await client.complete(ask(tools=[SEARCH]))
    assert response.stop_reason == "max_tokens"
    assert response.tool_calls == []
    assert all(b["type"] != "tool_use" for b in response.message.raw)
    with pytest.raises(LLMError, match="max_tokens"):
        await client.complete(ask(output_schema=Observations))


async def test_anthropic_refusal_raises():
    refused = message(
        [],
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "bio", "explanation": None},
    )
    client, _ = client_with(refused, refused)
    with pytest.raises(LLMRefusal, match="bio"):
        await client.complete(ask())
    with pytest.raises(LLMRefusal):
        [e async for e in client.stream(ask())]


async def test_anthropic_fallback_turn_is_kept_for_replay():
    # A mid-output decline answered by the fallback model: the boundary block stays,
    # the thinking and tool calls before it are dropped, the text keeps flowing.
    reply = message(
        [
            {"type": "thinking", "thinking": "", "signature": "a"},
            {"type": "text", "text": "Part one. "},
            {
                "type": "fallback",
                "from": {"model": "claude-opus-5-5"},
                "to": {"model": "claude-opus-4-8"},
                "trigger": {"type": "refusal", "category": "cyber"},
            },
            {"type": "text", "text": "Part two."},
        ],
        model="claude-opus-4-8",
    )
    client, _ = client_with(reply)
    response = await client.complete(ask())
    assert response.text == "Part one. Part two."
    assert response.model == "claude-opus-4-8"
    assert [b["type"] for b in response.message.raw] == ["text", "fallback", "text"]
    assert response.message.raw[1]["from"] == {"model": "claude-opus-5-5"}


async def test_anthropic_stream_yields_deltas_tools_done():
    reply = message(
        [
            {"type": "text", "text": "I went back and forth."},
            {"type": "tool_use", "id": "toolu_1", "name": "search_memory", "input": {"query": "a"}},
        ],
        stop_reason="tool_use",
    )
    client, sdk = client_with(reply)
    events = [e async for e in client.stream(ask(tools=[SEARCH], tier="chat"))]
    deltas = [e for e in events if isinstance(e, TextDelta)]
    assert len(deltas) > 1
    assert "".join(d.text for d in deltas) == "I went back and forth."
    tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
    assert tool_events[0].call.id == "toolu_1"
    assert isinstance(events[-1], Done)
    assert events.index(tool_events[0]) > events.index(deltas[-1])
    assert events[-1].response.stop_reason == "tool_use"
    assert sdk.calls[0]["fallbacks"] == "default"


async def test_anthropic_auth_error_maps_to_unavailable():
    client, _ = client_with(
        api_error(anthropic.AuthenticationError, 401),
        anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.invalid")),
        api_error(anthropic.BadRequestError, 400, "bad schema"),
        api_error(anthropic.InternalServerError, 500),
    )
    with pytest.raises(LLMUnavailable, match="ANTHROPIC_API_KEY"):
        await client.complete(ask())
    with pytest.raises(LLMUnavailable, match="network"):
        await client.complete(ask())
    with pytest.raises(LLMError, match="bad schema") as info:
        await client.complete(ask())
    assert not isinstance(info.value, LLMUnavailable)
    with pytest.raises(LLMError, match="500"):
        await client.complete(ask())


async def test_anthropic_auth_error_in_stream_maps_to_unavailable():
    client, _ = client_with(api_error(anthropic.AuthenticationError, 401))
    with pytest.raises(LLMUnavailable):
        [e async for e in client.stream(ask())]


async def test_anthropic_batch_uses_batches_api_and_maps_by_custom_id():
    client, sdk = client_with()
    sdk.beta.messages.batch_replies.extend(
        [
            text_message('{"items": ["a"]}'),
            "invalid request: something",
            text_message('{"items": ["c"]}'),
            "expired",
        ]
    )
    requests = [ask(f"chunk {i}", output_schema=Observations, tier="bulk") for i in range(4)]
    results = await client.complete_many(requests, batch=True)
    created = sdk.beta.messages.batches.created[0]
    assert len(created) == 4
    params = created[0]["params"]
    assert params["model"] == "claude-opus-5-5"
    assert params["output_config"]["effort"] == "low"
    assert "fallbacks" not in params and "betas" not in params  # rejected by the Batches API
    assert results[0].parsed == Observations(items=["a"])
    assert isinstance(results[1], LLMError) and "something" in str(results[1])
    assert results[2].parsed == Observations(items=["c"])
    assert isinstance(results[3], LLMError) and "expired" in str(results[3])


async def test_anthropic_batch_refusals_retry_with_fallback():
    refused = message(
        [], stop_reason="refusal", stop_details={"type": "refusal", "category": "bio"}
    )
    client, sdk = client_with(text_message('{"items": ["rescued"]}', model="claude-opus-4-8"))
    sdk.beta.messages.batch_replies.extend([refused, text_message('{"items": ["fine"]}')])
    results = await client.complete_many(
        [ask("a", output_schema=Observations), ask("b", output_schema=Observations)], batch=True
    )
    assert results[0].parsed == Observations(items=["rescued"])
    assert results[1].parsed == Observations(items=["fine"])
    assert sdk.calls[0]["fallbacks"] == "default"


async def test_anthropic_complete_many_without_batch_uses_messages():
    client, sdk = client_with(text_message("1"), text_message("2"))
    results = await client.complete_many([ask("a"), ask("b")], concurrency=1)
    assert [r.text for r in results] == ["1", "2"]
    assert not sdk.beta.messages.batches.created


ANTHROPIC_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_CONFIG_DIR",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
    "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
)


@pytest.fixture
def no_credentials(monkeypatch, tmp_path):
    import os

    for name in list(os.environ):
        if name.startswith("ANTHROPIC_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.config/anthropic profile either
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))


def test_get_client_without_credentials_raises_unavailable_with_fix(no_credentials):
    with pytest.raises(LLMUnavailable) as info:
        get_client(HomeConfig())
    message_text = str(info.value)
    assert "ANTHROPIC_API_KEY" in message_text
    assert "ant auth login" in message_text
    assert "\n" not in message_text


def test_get_client_with_key_builds_cached_client(no_credentials, monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    plain = get_client(HomeConfig())
    assert isinstance(plain, AnthropicClient)
    cached = get_client(HomeConfig(), cache_dir=tmp_path / "cache")
    assert isinstance(cached, CachedLLM)
    assert isinstance(cached.inner, AnthropicClient)
