"""LLMClient on the official Anthropic SDK (written against anthropic 2.8.0).

What every request carries:

- ``model`` and ``output_config.effort`` from the request's tier (config).
- No ``thinking`` parameter: claude-opus-5-5 runs adaptive thinking when it is
  omitted and rejects every other setting. No sampling parameters, no
  ``tool_choice`` (forced tool choice is a 400 on this model; ``auto`` is the default).
- Tools with ``strict: true``; structured output through ``output_config.format``
  with a JSON schema made by the SDK's ``transform_schema`` from the pydantic model.
- ``cache_control`` on system blocks marked ``cache=True``, plus top-level automatic
  caching of the conversation prefix for multi-turn requests.
- On models that support it, the server-side refusal fallback:
  beta ``server-side-fallback-2026-07-01`` with ``fallbacks="default"``, so a
  classifier decline is re-run on the model Anthropic recommends for that category
  inside the same call. The Batches API rejects ``fallbacks``; batch items that come
  back refused are retried once through the regular route, which has it.

Assistant turns keep the provider's content blocks in ``Message.raw`` and are
replayed verbatim, so thinking blocks and fallback boundaries survive tool loops.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import anthropic

from repartee.config import EffortTiers, ModelTiers
from repartee.llm.base import (
    Done,
    LLMClient,
    LLMError,
    LLMEvent,
    LLMRefusal,
    LLMRequest,
    LLMResponse,
    LLMUnavailable,
    Message,
    StopReason,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    Usage,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

CREDENTIALS_FIX = "No Anthropic credentials found. Set ANTHROPIC_API_KEY or run `ant auth login`."
FALLBACK_BETA = "server-side-fallback-2026-07-01"
#: Models that accept ``fallbacks``; the skill prescribes opting in on all of them.
FALLBACK_MODELS = frozenset(
    {
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5-5",
        "claude-fable-5-2",
        "claude-fable-5-1",
        "claude-mythos-5-2",
        "claude-mythos-5-1",
    }
)
MAX_CACHE_BREAKPOINTS = 4
#: The SDK refuses a non-streaming call above about 21K max_tokens (ten-minute guard).
NONSTREAMING_MAX_TOKENS = 20_000
BATCH_SIZE = 5000  # well under the API's 100,000 requests / 256 MB per batch
#: Blocks that must not be echoed from before a fallback boundary.
_DROP_BEFORE_FALLBACK = {"thinking", "redacted_thinking", "tool_use", "server_tool_use"}


def make_sdk_client() -> anthropic.AsyncAnthropic:
    """An AsyncAnthropic from the environment, or LLMUnavailable saying how to fix it."""
    try:
        client = anthropic.AsyncAnthropic()
    except anthropic.AnthropicError as error:
        raise LLMUnavailable(f"{CREDENTIALS_FIX} ({_one_line(error)})") from error
    if client.api_key is None and client.auth_token is None and client.credentials is None:
        raise LLMUnavailable(CREDENTIALS_FIX)
    return client


class AnthropicClient(LLMClient):
    """Anthropic Messages API with tiers, structured output, caching and refusal fallback."""

    def __init__(
        self,
        models: ModelTiers,
        effort: EffortTiers,
        client: anthropic.AsyncAnthropic | None = None,
        *,
        poll_interval: float = 30.0,
    ) -> None:
        self.models = models
        self.effort = effort
        self._client = client if client is not None else make_sdk_client()
        self.poll_interval = poll_interval

    # ------------------------------------------------------------------ protocol

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """One request; streamed under the hood when max_tokens is too large for a plain call."""
        params = self.build_params(request)
        with _mapped_errors():
            if request.max_tokens > NONSTREAMING_MAX_TOKENS:
                async with self._client.beta.messages.stream(**params) as stream:
                    message = await stream.get_final_message()
            else:
                message = await self._client.beta.messages.create(**params)
        return self.to_response(message, request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        """Text deltas as they arrive, then tool calls, then Done."""
        with _mapped_errors():
            async with self._client.beta.messages.stream(**self.build_params(request)) as stream:
                async for event in stream:
                    if event.type == "text" and event.text:
                        yield TextDelta(text=event.text)
                message = await stream.get_final_message()
        response = self.to_response(message, request)
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
        """Concurrent requests, or the Message Batches API (half price) when ``batch``."""
        if not batch:
            return await super().complete_many(requests, concurrency=concurrency)
        results: list[LLMResponse | LLMError] = []
        for start in range(0, len(requests), BATCH_SIZE):
            results.extend(await self._run_batch(requests[start : start + BATCH_SIZE]))
        refused = [i for i, result in enumerate(results) if isinstance(result, LLMRefusal)]
        if refused:  # batches cannot fall back; the regular route can
            retried = await super().complete_many(
                [requests[i] for i in refused], concurrency=concurrency
            )
            for index, result in zip(refused, retried, strict=True):
                results[index] = result
        return results

    # ------------------------------------------------------------------ requests

    def build_params(self, request: LLMRequest, *, for_batch: bool = False) -> dict[str, Any]:
        """Keyword arguments for ``beta.messages.create`` (or one batch item's params)."""
        model = getattr(self.models, request.tier)
        output_config: dict[str, Any] = {"effort": getattr(self.effort, request.tier)}
        if request.output_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": anthropic.transform_schema(request.output_schema),
            }
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": [_message_param(m) for m in request.messages],
            "output_config": output_config,
        }
        conversation_breakpoint = len(request.messages) > 1
        if request.system:
            budget = MAX_CACHE_BREAKPOINTS - int(conversation_breakpoint)
            cached = [i for i, block in enumerate(request.system) if block.cache][-budget:]
            params["system"] = [
                {"type": "text", "text": block.text}
                | ({"cache_control": {"type": "ephemeral"}} if i in cached else {})
                for i, block in enumerate(request.system)
            ]
        if conversation_breakpoint:
            params["cache_control"] = {"type": "ephemeral"}
        if request.tools:
            params["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "strict": True,
                }
                for tool in request.tools
            ]
        if not for_batch and model in FALLBACK_MODELS:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    # ------------------------------------------------------------------ responses

    def to_response(self, message: Any, request: LLMRequest) -> LLMResponse:
        """Map an SDK message to LLMResponse, checking the stop reason first."""
        stop = message.stop_reason
        model = getattr(message, "model", "") or ""
        if stop == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise LLMRefusal(
                f"{model or 'the model'} declined the request "
                f"(purpose {request.purpose!r}, category {category}), after any fallback"
            )
        blocks = list(message.content)
        boundary = max((i for i, b in enumerate(blocks) if b.type == "fallback"), default=-1)
        truncated = stop in ("max_tokens", "model_context_window_exceeded")
        raw: list[dict[str, Any]] = []
        for index, block in enumerate(blocks):
            if index < boundary and block.type in _DROP_BEFORE_FALLBACK:
                continue
            if truncated and block.type == "tool_use":
                continue  # a cut-off call must not be replayed or run
            raw.append(block.to_dict() if hasattr(block, "to_dict") else dict(block))

        text = "".join(block.text for block in blocks if block.type == "text")
        calls = [
            ToolCall(id=block.id, name=block.name, input=dict(block.input or {}))
            for block in blocks[boundary + 1 :]
            if block.type == "tool_use" and not truncated
        ]
        stop_reason: StopReason
        if truncated:
            stop_reason = "max_tokens"
        elif stop == "tool_use":
            stop_reason = "tool_use"
        elif stop in ("end_turn", "stop_sequence"):
            stop_reason = "end_turn"
        else:  # pause_turn and compaction need server tools or betas this client never sends
            raise LLMError(f"unexpected stop reason {stop!r} from {model}")

        parsed = None
        if request.output_schema is not None:
            if truncated:
                raise LLMError(
                    f"structured reply for purpose {request.purpose!r} was cut off at "
                    f"max_tokens={request.max_tokens}; raise max_tokens"
                )
            parsed = _parse(text, request.output_schema, request.purpose)

        usage = message.usage
        return LLMResponse(
            text=text,
            tool_calls=calls,
            parsed=parsed,
            stop_reason=stop_reason,
            usage=Usage(
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
            ),
            model=model,
            message=Message(role="assistant", text=text, tool_calls=calls, raw=raw),
        )

    # ------------------------------------------------------------------ batches

    async def _run_batch(self, requests: Sequence[LLMRequest]) -> list[LLMResponse | LLMError]:
        batches = self._client.beta.messages.batches
        items = [
            {"custom_id": f"r{i}", "params": self.build_params(r, for_batch=True)}
            for i, r in enumerate(requests)
        ]
        try:
            with _mapped_errors():
                job = await batches.create(requests=items)
                while job.processing_status != "ended":
                    await asyncio.sleep(self.poll_interval)
                    job = await batches.retrieve(job.id)
                results: list[LLMResponse | LLMError | None] = [None] * len(requests)
                async for item in await batches.results(job.id):
                    index = int(item.custom_id[1:])
                    results[index] = self._batch_result(item.result, requests[index])
        except LLMError as error:
            return [error] * len(requests)
        return [r if r is not None else LLMError("missing from batch results") for r in results]

    def _batch_result(self, result: Any, request: LLMRequest) -> LLMResponse | LLMError:
        if result.type == "succeeded":
            try:
                return self.to_response(result.message, request)
            except LLMError as error:
                return error
        if result.type == "errored":
            error = getattr(result.error, "error", result.error)
            return LLMError(f"batch item failed: {getattr(error, 'message', error)}")
        return LLMError(f"batch item {result.type}")


# ---------------------------------------------------------------------- helpers


def _message_param(message: Message) -> dict[str, Any]:
    if message.role == "assistant":
        if message.raw is not None:
            return {"role": "assistant", "content": message.raw}
        content: list[dict[str, Any]] = []
        if message.text:
            content.append({"type": "text", "text": message.text})
        content.extend(
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input}
            for c in message.tool_calls
        )
        return {"role": "assistant", "content": content}
    content = [
        {"type": "tool_result", "tool_use_id": r.tool_call_id, "content": r.content}
        | ({"is_error": True} if r.is_error else {})
        for r in message.tool_results
    ]
    if message.text or not content:
        content.append({"type": "text", "text": message.text})
    return {"role": "user", "content": content}


def _parse(text: str, schema: type[BaseModel], purpose: str) -> BaseModel:
    try:
        return schema.model_validate_json(text)
    except ValueError as error:
        raise LLMError(
            f"reply for purpose {purpose!r} is not a valid {schema.__name__}: {_one_line(error)}"
        ) from error


def _one_line(error: BaseException) -> str:
    return " ".join(str(error).split())[:300]


@contextlib.contextmanager
def _mapped_errors() -> Iterator[None]:
    """Translate SDK exceptions into the LLMError family (the SDK has already retried)."""
    try:
        yield
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as error:
        raise LLMUnavailable(
            f"The Anthropic API refused the credentials ({error.status_code}). "
            "Set ANTHROPIC_API_KEY or run `ant auth login`."
        ) from error
    except anthropic.APIConnectionError as error:
        raise LLMUnavailable(
            f"Cannot reach the Anthropic API ({_one_line(error)}). Check the network connection."
        ) from error
    except anthropic.APIStatusError as error:
        raise LLMError(f"Anthropic API error {error.status_code}: {_one_line(error)}") from error
    except anthropic.AnthropicError as error:
        raise LLMError(f"Anthropic SDK error: {_one_line(error)}") from error
    except TypeError as error:
        if "Could not resolve authentication method" in str(error):
            raise LLMUnavailable(CREDENTIALS_FIX) from error
        raise
