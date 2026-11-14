"""A scripted model for tests: no network, no key, deterministic.

Script it by purpose with standing rules, or with an ordered queue::

    fake = FakeLLM()
    fake.on("extract", Observations(items=[...]))            # every "extract" request
    fake.on(lambda r: "Ana" in r.messages[-1].text, "She works nights.")
    fake.add([tool_call("search_self", query="moving")], "I went back and forth.")
    fake.add("only for revise", purpose="revise")             # queued, purpose-scoped

Rules are tried first (most recently added first), then the queue in order. A request
nothing answers fails with an AssertionError that names its purpose, quotes the
last message and lists what is scripted. Every request is kept in ``requests``.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import regex
from pydantic import BaseModel, ValidationError

from repartee.config import estimate_tokens
from repartee.llm.base import (
    Done,
    LLMClient,
    LLMEvent,
    LLMRequest,
    LLMResponse,
    Message,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    Usage,
)

Reply = str | BaseModel | dict | list[ToolCall] | LLMResponse | Exception | type[Exception]
Matcher = str | Callable[[LLMRequest], bool]
Responder = Reply | Callable[[LLMRequest], Reply]

_call_ids = itertools.count(1)


def tool_call(name: str, *, id: str | None = None, **input: Any) -> ToolCall:
    """A ToolCall with a fresh unique id, for scripting tool-using replies."""
    return ToolCall(id=id or f"toolu_fake_{next(_call_ids)}", name=name, input=input)


@dataclass
class _Rule:
    matcher: Matcher
    reply: Responder

    def matches(self, request: LLMRequest) -> bool:
        if isinstance(self.matcher, str):
            return request.purpose == self.matcher
        return bool(self.matcher(request))

    def describe(self) -> str:
        if isinstance(self.matcher, str):
            return f"rule for purpose {self.matcher!r}"
        return f"rule {getattr(self.matcher, '__name__', 'predicate')}"


@dataclass
class _Queued:
    reply: Responder
    purpose: str | None = None

    def describe(self) -> str:
        scope = f" (purpose {self.purpose!r})" if self.purpose else ""
        return f"queued {_short(self.reply)}{scope}"


class FakeLLM(LLMClient):
    """Scripted LLMClient: rules by purpose or predicate, an ordered queue, full recording."""

    def __init__(self, script: Sequence[Responder] = ()) -> None:
        self.requests: list[LLMRequest] = []
        self._rules: list[_Rule] = []
        self._queue: list[_Queued] = [_Queued(reply) for reply in script]

    # ------------------------------------------------------------------ scripting

    def add(self, *replies: Responder, purpose: str | None = None) -> FakeLLM:
        """Queue replies, consumed in order (only by requests of ``purpose`` when given)."""
        self._queue.extend(_Queued(reply, purpose) for reply in replies)
        return self

    def on(self, purpose: Matcher, reply: Responder) -> FakeLLM:
        """Answer every matching request with ``reply`` (a callable gets the request)."""
        self._rules.append(_Rule(purpose, reply))
        return self

    def calls(self, purpose: str) -> list[LLMRequest]:
        """The recorded requests with this purpose."""
        return [r for r in self.requests if r.purpose == purpose]

    def assert_exhausted(self) -> None:
        """Fail if queued replies were never used."""
        if self._queue:
            left = "; ".join(q.describe() for q in self._queue)
            raise AssertionError(f"FakeLLM: {len(self._queue)} queued replies never used: {left}")

    # ------------------------------------------------------------------ protocol

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Record the request and answer it from the script."""
        self.requests.append(request.model_copy(update={"messages": list(request.messages)}))
        return self.respond(self._pick(request), request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        """Like ``complete``, with the text split into several deltas."""
        response = await self.complete(request)
        for piece in regex.findall(r"\s*\S+\s*", response.text) or (
            [response.text] if response.text else []
        ):
            yield TextDelta(text=piece)
        for call in response.tool_calls:
            yield ToolCallEvent(call=call)
        yield Done(response=response)

    # ------------------------------------------------------------------ internals

    def _pick(self, request: LLMRequest) -> Responder:
        for rule in reversed(self._rules):
            if rule.matches(request):
                return rule.reply
        for index, queued in enumerate(self._queue):
            if queued.purpose is None or queued.purpose == request.purpose:
                return self._queue.pop(index).reply
        raise AssertionError(self._unscripted_message(request))

    def _unscripted_message(self, request: LLMRequest) -> str:
        last = request.messages[-1] if request.messages else None
        said = ""
        if last is not None:
            said = last.text or "; ".join(r.content for r in last.tool_results)
        scripted = [rule.describe() for rule in reversed(self._rules)]
        scripted += [queued.describe() for queued in self._queue]
        return (
            f"FakeLLM got an unscripted request: purpose={request.purpose!r} "
            f"tier={request.tier!r}, last message {_clip(said)!r}. "
            f"Scripted: {', '.join(scripted) or 'nothing'}. "
            "Add a reply with fake.on(purpose, reply) or fake.add(reply)."
        )

    @staticmethod
    def respond(reply: Responder, request: LLMRequest) -> LLMResponse:
        """Turn a scripted reply into the LLMResponse a real client would return."""
        if callable(reply) and not isinstance(reply, type):
            reply = reply(request)
        if isinstance(reply, type) and issubclass(reply, Exception):
            raise reply(f"scripted {reply.__name__} for purpose {request.purpose!r}")
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, LLMResponse):
            if not reply.message.text and not reply.message.tool_calls:
                reply.message.text = reply.text
                reply.message.tool_calls = list(reply.tool_calls)
            return reply

        text, parsed, calls = "", None, []
        schema = request.output_schema
        if isinstance(reply, list):
            calls = _check_tool_calls(reply, request)
        elif isinstance(reply, BaseModel):
            if schema is None:
                raise AssertionError(
                    f"FakeLLM: scripted a {type(reply).__name__} for purpose "
                    f"{request.purpose!r}, but the request has no output_schema"
                )
            if not isinstance(reply, schema):
                raise AssertionError(
                    f"FakeLLM: purpose {request.purpose!r} asks for {schema.__name__}, "
                    f"scripted reply is a {type(reply).__name__}"
                )
            parsed, text = reply, reply.model_dump_json()
        elif isinstance(reply, dict | str):
            if schema is None:
                if isinstance(reply, dict):
                    raise AssertionError(
                        f"FakeLLM: scripted a dict for purpose {request.purpose!r}, "
                        "but the request has no output_schema"
                    )
                text = reply
            else:
                try:
                    if isinstance(reply, dict):
                        parsed = schema.model_validate(reply)
                    else:
                        parsed = schema.model_validate_json(reply)
                except ValidationError as error:
                    raise AssertionError(
                        f"FakeLLM: scripted reply for purpose {request.purpose!r} "
                        f"is not a valid {schema.__name__}: {error}"
                    ) from error
                text = parsed.model_dump_json()
        else:
            raise AssertionError(f"FakeLLM: cannot use {reply!r} as a reply")

        return LLMResponse(
            text=text,
            tool_calls=calls,
            parsed=parsed,
            stop_reason="tool_use" if calls else "end_turn",
            usage=Usage(
                input_tokens=_request_tokens(request),
                output_tokens=estimate_tokens(text + json.dumps([c.input for c in calls])),
            ),
            model=f"fake-{request.tier}",
            message=Message(role="assistant", text=text, tool_calls=calls),
        )


def _check_tool_calls(calls: list[Any], request: LLMRequest) -> list[ToolCall]:
    if not calls or not all(isinstance(c, ToolCall) for c in calls):
        raise AssertionError(f"FakeLLM: a list reply must hold ToolCalls, got {calls!r}")
    offered = {tool.name for tool in request.tools}
    for call in calls:
        if call.name not in offered:
            raise AssertionError(
                f"FakeLLM: scripted a call to {call.name!r} for purpose {request.purpose!r}, "
                f"but the request offers {sorted(offered) or 'no tools'}"
            )
    return list(calls)


def _request_tokens(request: LLMRequest) -> int:
    parts = [block.text for block in request.system]
    for message in request.messages:
        parts.append(message.text)
        parts.extend(result.content for result in message.tool_results)
    return estimate_tokens("\n".join(parts))


def _clip(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _short(reply: Responder) -> str:
    if isinstance(reply, str):
        return repr(_clip(reply, 40))
    if callable(reply) and not isinstance(reply, type | BaseModel):
        return getattr(reply, "__name__", "callable")
    return type(reply).__name__ if not isinstance(reply, type) else reply.__name__
