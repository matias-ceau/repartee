"""The LLM protocol the rest of Repartee speaks.

Callers name a *tier* and a *purpose*, never a model. Implementations map tiers
to models and effort levels from config. Everything is async.

Purposes in use (also how FakeLLM routes replies):
    extract, route, consolidate, core, gaps        distill pipeline
    style_card, draft, revise                      voice
    chat, interview                                mirror
    brief, judge                                   evals
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from repartee.config import Tier


class LLMError(Exception):
    """Any failure talking to the model."""


class LLMUnavailable(LLMError):
    """No credentials or no network. The message says how to fix it."""


class LLMRefusal(LLMError):
    """The model declined the request (after any fallback)."""


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema, additionalProperties false, all required listed


class ToolCall(BaseModel):
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_call_id: str
    content: str
    is_error: bool = False


class Message(BaseModel):
    """One turn. Assistant turns may carry tool calls; user turns may carry tool results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: Literal["user", "assistant"]
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    #: The provider's own content for this assistant turn. Implementations replay it
    #: unchanged (thinking blocks included) instead of rebuilding from text/tool_calls.
    raw: Any | None = Field(default=None, exclude=True)


class SystemBlock(BaseModel):
    text: str
    cache: bool = False  # mark the end of a stable prefix worth caching


class LLMRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tier: Tier = "smart"
    purpose: str = ""
    system: list[SystemBlock] = Field(default_factory=list)
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    #: When set, the reply must be JSON matching this model; see LLMResponse.parsed.
    output_schema: type[BaseModel] | None = None
    max_tokens: int = 16000


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    #: What the provider says it charged, when it says (OpenRouter does). Preferred over
    #: an estimate from token counts, since the price depends on who served the request.
    cost_usd: float | None = None


StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal"]


class LLMResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    parsed: BaseModel | None = None  # instance of request.output_schema when one was given
    stop_reason: StopReason = "end_turn"
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    #: The assistant turn to append to the conversation, with ``raw`` filled in.
    message: Message = Field(default_factory=lambda: Message(role="assistant"))


class TextDelta(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    call: ToolCall


class Done(BaseModel):
    type: Literal["done"] = "done"
    response: LLMResponse


LLMEvent = TextDelta | ToolCallEvent | Done


class LLMClient(ABC):
    """Implement ``complete``; override ``stream`` and ``complete_many`` when the backend can."""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """One request, one full response. Raises LLMError subclasses."""

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        """Text deltas and tool calls as they arrive, then exactly one Done."""
        response = await self.complete(request)
        if response.text:
            yield TextDelta(text=response.text)
        for call in response.tool_calls:
            yield ToolCallEvent(call=call)
        yield Done(response=response)

    async def complete_many(
        self,
        requests: Sequence[LLMRequest],
        *,
        concurrency: int = 8,
        batch: bool = False,
    ) -> list[LLMResponse | LLMError]:
        """Run many requests; results are in request order, failures returned in place.

        ``batch`` asks for the provider's cheaper asynchronous batch route when it
        has one; the default implementation ignores it.
        """
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def one(request: LLMRequest) -> LLMResponse | LLMError:
            async with semaphore:
                try:
                    return await self.complete(request)
                except LLMError as error:
                    return error

        return list(await asyncio.gather(*(one(r) for r in requests)))
