"""A stand-in for ``openai.AsyncOpenAI`` that records calls and never touches the network.

Only ``chat.completions.create`` is modelled, plain and streamed. Replies are real SDK
``ChatCompletion`` / ``ChatCompletionChunk`` objects, built without validation as the
SDK builds them, so the mapping code sees the shapes (extra provider fields and
values outside the SDK's literals included) it sees in production.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
from openai.types.chat import ChatCompletion, ChatCompletionChunk


def completion(
    content: str | None = "",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    model: str = "z-ai/glm-5.3-flash",
    usage: dict[str, Any] | None = None,
    refusal: str | None = None,
    **message_extra: Any,
) -> ChatCompletion:
    """A ChatCompletion as the API would return it; extra kwargs land on the message."""
    message: dict[str, Any] = {"role": "assistant", "content": content, "refusal": refusal}
    if tool_calls:
        message["tool_calls"] = tool_calls
    message.update(message_extra)
    return ChatCompletion.construct(
        **{
            "id": "gen-stub",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


def call(id: str, name: str, arguments: Any) -> dict[str, Any]:
    """One tool call as the API returns it (arguments as a JSON string)."""
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": id, "type": "function", "function": {"name": name, "arguments": text}}


def chunk(
    delta: dict[str, Any] | None = None,
    *,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
    model: str = "z-ai/glm-5.3-flash",
    **extra: Any,
) -> ChatCompletionChunk:
    choices = [] if delta is None else [
        {"index": 0, "delta": delta, "finish_reason": finish_reason}
    ]
    return ChatCompletionChunk.construct(
        **{
            "id": "gen-stub",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": choices,
            "usage": usage,
            **extra,
        }
    )


class StubStream:
    """Quacks like openai.AsyncStream: async iterable and async context manager."""

    def __init__(self, chunks: list[ChatCompletionChunk | Exception]) -> None:
        self.chunks = chunks
        self.closed = False

    async def __aenter__(self) -> StubStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed = True

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for item in self.chunks:
            if isinstance(item, Exception):
                raise item
            yield item


class StubCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.replies: list[Any] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(json.loads(json.dumps(kwargs)))  # a snapshot, as sent
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if kwargs.get("stream"):
            assert isinstance(reply, list), "a streamed call needs a list of chunks"
            return StubStream(reply)
        return reply


class StubOpenAI:
    """Quacks like AsyncOpenAI for the calls OpenAIClient makes."""

    def __init__(self, *replies: Any, base_url: str = "https://openrouter.ai/api/v1") -> None:
        self.base_url = base_url
        self.chat = SimpleNamespace(completions=StubCompletions())
        self.chat.completions.replies.extend(replies)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.chat.completions.calls


def api_error(cls: type, status: int, text: str = "nope") -> Exception:
    request = httpx2.Request("POST", "https://openrouter.invalid/api/v1/chat/completions")
    response = httpx2.Response(status, request=request)
    return cls(text, response=response, body=None)


def connection_error() -> Exception:
    return openai.APIConnectionError(request=httpx2.Request("POST", "https://openrouter.invalid"))


def timeout_error() -> Exception:
    return openai.APITimeoutError(request=httpx2.Request("POST", "https://openrouter.invalid"))
