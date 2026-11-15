"""LLMClient on the official OpenAI SDK's chat completions API (written against openai 3.51.0).

One client for every server that speaks that API: ``provider="openrouter"`` (base URL
and key fixed here) and ``provider="openai"`` (the SDK reads OPENAI_API_KEY and
OPENAI_BASE_URL, so any compatible server, local ones included).

What every request carries:

- ``model`` from the request's tier. Under OpenRouter, ``reasoning.effort`` from the
  tier (through ``extra_body``) and ``max_tokens``; other servers get
  ``max_completion_tokens`` and no reasoning setting, which they may not accept.
- The system blocks joined into one system message (OpenAI-style providers cache a
  stable prefix by themselves; there is nothing to mark).
- Tools as strict function tools. No ``tool_choice``, no sampling parameters.
- Structured output as a strict JSON-schema response format made by
  :func:`strict_schema`. A provider that rejects the schema gets plain JSON mode with
  the schema in the system message instead (remembered per model). A reply that does
  not parse or validate is answered once, in the same conversation, with the error;
  a second failure is an LLMError.

Assistant turns keep the provider's message (content, tool calls, OpenRouter's
``reasoning_details``) as a plain dict in ``Message.raw`` and are replayed verbatim.
No app-identifying headers are sent.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import types
from collections.abc import AsyncIterator, Iterator
from typing import Any, Union, get_args, get_origin

import openai
from pydantic import BaseModel

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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_FIX = "Set OPENROUTER_API_KEY to an OpenRouter key (https://openrouter.ai/keys)."
OPENAI_FIX = (
    "Set OPENAI_API_KEY (any value for a local server without auth), and OPENAI_BASE_URL "
    "for a server other than OpenAI."
)
#: Our effort levels in OpenRouter's vocabulary (it also has "minimal" and "none").
#: OpenRouter moves a level a model lacks to the nearest one it has.
OPENROUTER_EFFORT: dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}
#: Schema keywords kept for strict mode; validation happens in pydantic anyway.
_SCHEMA_KEYWORDS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "anyOf",
    "description",
}
_FENCE = re.compile(r"```[A-Za-z0-9_-]*\s*\n?(.*?)```", re.DOTALL)
RETRY_PROMPT = (
    "That reply could not be used: {error}. Reply again with only the JSON object that "
    "matches the schema: no other text, no code fence."
)
JSON_MODE_PROMPT = (
    "Reply with only a JSON object, no other text, matching this JSON schema:\n{schema}"
)


def make_sdk_client(provider: str) -> openai.AsyncOpenAI:
    """An AsyncOpenAI for the provider, or LLMUnavailable saying how to fix it."""
    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise LLMUnavailable(f"No OpenRouter key found. {OPENROUTER_FIX}")
        return openai.AsyncOpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)
    try:
        return openai.AsyncOpenAI()
    except openai.OpenAIError as error:
        raise LLMUnavailable(f"No OpenAI-compatible credentials found. {OPENAI_FIX}") from error


class OpenAIClient(LLMClient):
    """OpenAI chat completions (OpenRouter, OpenAI, compatible servers) behind our protocol."""

    def __init__(
        self,
        models: ModelTiers,
        effort: EffortTiers,
        client: Any | None = None,
        *,
        provider: str = "openai",
    ) -> None:
        self.models = models
        self.effort = effort
        self.provider = provider
        self.sdk = client if client is not None else make_sdk_client(provider)
        #: Models whose provider rejected a JSON-schema response format: JSON mode instead.
        self.json_mode_models: set[str] = set()

    # ------------------------------------------------------------------ protocol

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """One request; structured requests are validated and retried once."""
        if request.output_schema is not None:
            return await self._structured(request)
        return self.to_response(await self._create(self.build_params(request)), request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        """Text deltas as they arrive, then tool calls, then Done."""
        if request.output_schema is not None:  # validated JSON is only usable whole
            async for event in super().stream(request):
                yield event
            return
        params = self.build_params(request)
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
        accumulated = _StreamAccumulator()
        with self._mapped_errors():
            stream = await self.sdk.chat.completions.create(**params)
            async with stream:
                async for chunk in stream:
                    text = accumulated.add(_as_dict(chunk))
                    if text:
                        yield TextDelta(text=text)
        response = self.to_response(accumulated.result(), request)
        for call in response.tool_calls:
            yield ToolCallEvent(call=call)
        yield Done(response=response)

    # ------------------------------------------------------------------ requests

    def model_for(self, request: LLMRequest) -> str:
        return str(getattr(self.models, request.tier))

    def build_params(self, request: LLMRequest, *, json_mode: bool = False) -> dict[str, Any]:
        """Keyword arguments for ``chat.completions.create``."""
        schema = request.output_schema
        instruction = None
        if schema is not None and json_mode:
            instruction = JSON_MODE_PROMPT.format(schema=json.dumps(strict_schema(schema)))
        params: dict[str, Any] = {
            "model": self.model_for(request),
            "messages": _messages(request, instruction),
        }
        if self.provider == "openrouter":
            params["max_tokens"] = request.max_tokens
            effort = OPENROUTER_EFFORT.get(str(getattr(self.effort, request.tier)), "medium")
            params["extra_body"] = {"reasoning": {"effort": effort}}
        else:
            params["max_completion_tokens"] = request.max_tokens
        if request.tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                        "strict": True,
                    },
                }
                for tool in request.tools
            ]
        if schema is not None:
            if json_mode:
                params["response_format"] = {"type": "json_object"}
            else:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": strict_schema(schema),
                        "strict": True,
                    },
                }
        return params

    async def _create(self, params: dict[str, Any], *, schema_attempt: bool = False) -> dict:
        with self._mapped_errors(schema_attempt=schema_attempt):
            completion = await self.sdk.chat.completions.create(**params)
        return _as_dict(completion)

    async def _structured(self, request: LLMRequest) -> LLMResponse:
        assert request.output_schema is not None
        model = self.model_for(request)
        json_mode = model in self.json_mode_models
        try:
            data = await self._create(
                self.build_params(request, json_mode=json_mode), schema_attempt=not json_mode
            )
        except _SchemaRejected:
            json_mode = True
            data = await self._create(self.build_params(request, json_mode=True))
            self.json_mode_models.add(model)

        response = self.to_response(data, request)
        try:
            response.parsed = parse_reply(response.text, request.output_schema)
            return response
        except ValueError as error:
            problem = _one_line(error)

        retry = request.model_copy(
            update={
                "messages": [
                    *request.messages,
                    response.message,
                    Message(role="user", text=RETRY_PROMPT.format(error=problem)),
                ]
            }
        )
        second = self.to_response(
            await self._create(self.build_params(retry, json_mode=json_mode)), retry
        )
        second.usage = _add_usage(response.usage, second.usage)
        try:
            second.parsed = parse_reply(second.text, request.output_schema)
        except ValueError as error:
            raise LLMError(
                f"reply for purpose {request.purpose!r} is not a valid "
                f"{request.output_schema.__name__}, even when asked again: {_one_line(error)}"
            ) from error
        return second

    # ------------------------------------------------------------------ responses

    def to_response(self, data: dict[str, Any], request: LLMRequest) -> LLMResponse:
        """Map a chat completion (as a dict) to LLMResponse, checking the finish first."""
        model = data.get("model") or self.model_for(request)
        choices = data.get("choices") or []
        error = data.get("error") or (choices[0].get("error") if choices else None)
        finish = choices[0].get("finish_reason") if choices else None
        if error or finish == "error":
            raise LLMError(f"{self._label()} reported an error from {model}: {_error_text(error)}")
        if not choices:
            raise LLMError(f"{self._label()} returned no reply from {model}")
        message = choices[0].get("message") or {}
        if finish == "content_filter":
            raise LLMRefusal(
                f"{model} declined the request (purpose {request.purpose!r}, content filter)"
            )
        if message.get("refusal"):
            raise LLMRefusal(
                f"{model} declined the request (purpose {request.purpose!r}): "
                f"{_one_line(message['refusal'])}"
            )

        truncated = finish == "length"
        text = _content_text(message.get("content"))
        raw_calls = [] if truncated else list(message.get("tool_calls") or [])
        calls: list[ToolCall] = []
        replay_calls: list[dict[str, Any]] = []
        for index, raw_call in enumerate(raw_calls):
            call = _tool_call(raw_call, index, model)
            calls.append(call)
            function = raw_call.get("function") or {}
            arguments = function.get("arguments")
            replay_calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": arguments
                        if isinstance(arguments, str) and arguments.strip()
                        else json.dumps(call.input),
                    },
                }
            )

        stop_reason: StopReason
        if truncated:
            stop_reason = "max_tokens"
        elif calls:
            stop_reason = "tool_use"
        else:  # stop, or a server's own name for it
            stop_reason = "end_turn"

        raw: dict[str, Any] = {"role": "assistant", "content": text if text or not calls else None}
        if replay_calls:
            raw["tool_calls"] = replay_calls
        if message.get("reasoning_details"):
            raw["reasoning_details"] = message["reasoning_details"]

        if request.output_schema is not None and truncated:
            raise LLMError(
                f"structured reply for purpose {request.purpose!r} was cut off at "
                f"max_tokens={request.max_tokens}; raise max_tokens"
            )
        return LLMResponse(
            text=text,
            tool_calls=calls,
            stop_reason=stop_reason,
            usage=_usage(data.get("usage")),
            model=model,
            message=Message(role="assistant", text=text, tool_calls=calls, raw=raw),
        )

    # ------------------------------------------------------------------ errors

    def _label(self) -> str:
        return "OpenRouter" if self.provider == "openrouter" else "The OpenAI-compatible API"

    @contextlib.contextmanager
    def _mapped_errors(self, *, schema_attempt: bool = False) -> Iterator[None]:
        """Translate SDK exceptions into the LLMError family (the SDK has already retried)."""
        label = self._label()
        fix = OPENROUTER_FIX if self.provider == "openrouter" else OPENAI_FIX
        try:
            yield
        except (openai.AuthenticationError, openai.PermissionDeniedError) as error:
            text = _one_line(error)
            if "moderation" in text.lower() or "flagged" in text.lower():
                raise LLMRefusal(f"{label} moderation declined the request: {text}") from error
            raise LLMUnavailable(
                f"{label} refused the credentials ({error.status_code}: {text}). {fix}"
            ) from error
        except openai.APITimeoutError as error:
            raise LLMUnavailable(
                f"{label} timed out. Check the network connection or try again later."
            ) from error
        except openai.APIConnectionError as error:
            where = "" if self.provider == "openrouter" else " and OPENAI_BASE_URL"
            raise LLMUnavailable(
                f"Cannot reach {label} ({_one_line(error)}). Check the network connection{where}."
            ) from error
        except openai.APIStatusError as error:
            text = _one_line(error)
            if error.status_code == 402 or (
                error.status_code == 429 and "quota" in text.lower()
            ):
                where = (
                    " at https://openrouter.ai/settings/credits"
                    if self.provider == "openrouter"
                    else ""
                )
                raise LLMUnavailable(
                    f"{label} says the account is out of credit ({error.status_code}): add "
                    f"credit{where} or choose cheaper models in repartee.yaml."
                ) from error
            if schema_attempt and error.status_code in (400, 404, 422):
                raise _SchemaRejected(text) from error
            raise LLMError(f"{label} API error {error.status_code}: {text}") from error
        except openai.OpenAIError as error:
            raise LLMError(f"{label} SDK error: {_one_line(error)}") from error


class _SchemaRejected(Exception):
    """The provider would not take the JSON-schema response format."""


# ---------------------------------------------------------------------- messages


def _messages(request: LLMRequest, instruction: str | None) -> list[dict[str, Any]]:
    system = "\n\n".join(block.text for block in request.system)
    if instruction:
        system = f"{system}\n\n{instruction}" if system else instruction
    out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    for message in request.messages:
        out.extend(_message_params(message))
    return out


def _message_params(message: Message) -> list[dict[str, Any]]:
    if message.role == "assistant":
        if isinstance(message.raw, dict):
            return [dict(message.raw)]
        param: dict[str, Any] = {
            "role": "assistant",
            "content": message.text if message.text or not message.tool_calls else None,
        }
        if message.tool_calls:
            param["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.input)},
                }
                for c in message.tool_calls
            ]
        return [param]
    # Tool results answer the assistant turn just before, so they come first.
    out: list[dict[str, Any]] = [
        {
            "role": "tool",
            "tool_call_id": r.tool_call_id,
            "content": f"Error: {r.content}" if r.is_error else r.content,
        }
        for r in message.tool_results
    ]
    if message.text or not out:
        out.append({"role": "user", "content": message.text})
    return out


# ---------------------------------------------------------------------- responses


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return obj.model_dump()


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some servers answer with content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _tool_call(raw_call: dict[str, Any], index: int, model: str) -> ToolCall:
    function = raw_call.get("function") or {}
    name = str(function.get("name") or "")
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        parsed: Any = arguments
    elif arguments is None or not str(arguments).strip():
        parsed = {}
    else:
        try:
            parsed = json.loads(arguments)
        except ValueError as error:
            raise LLMError(
                f"{model} called {name!r} with arguments that are not valid JSON: "
                f"{_one_line(error)} in {str(arguments)[:200]!r}"
            ) from error
    if not isinstance(parsed, dict):
        raise LLMError(f"{model} called {name!r} with arguments that are not a JSON object")
    return ToolCall(id=raw_call.get("id") or f"call_{index}", name=name, input=parsed)


def _usage(usage: dict[str, Any] | None) -> Usage:
    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    written = details.get("cache_write_tokens") or 0
    prompt = usage.get("prompt_tokens") or 0
    return Usage(
        input_tokens=max(0, prompt - cached - written),  # prompt_tokens counts them all
        output_tokens=usage.get("completion_tokens") or 0,
        cache_read_tokens=cached,
        cache_write_tokens=written,
    )


def _add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_read_tokens=a.cache_read_tokens + b.cache_read_tokens,
        cache_write_tokens=a.cache_write_tokens + b.cache_write_tokens,
    )


def _error_text(error: Any) -> str:
    if isinstance(error, dict):
        message = error.get("message") or json.dumps(error)
        code = error.get("code")
        return f"{message} ({code})" if code else str(message)
    return str(error) if error else "unspecified error"


def _one_line(error: BaseException | str) -> str:
    return " ".join(str(error).split())[:300]


class _StreamAccumulator:
    """Folds streamed chunks into the dict ``to_response`` reads for a whole completion."""

    def __init__(self) -> None:
        self.model = ""
        self.text: list[str] = []
        self.refusal: list[str] = []
        self.calls: dict[int, dict[str, str]] = {}
        self.reasoning: list[dict[str, Any]] = []
        self.finish: str | None = None
        self.usage: dict[str, Any] | None = None
        self.error: Any = None

    def add(self, chunk: dict[str, Any]) -> str:
        """Take one chunk; returns its new text."""
        self.model = chunk.get("model") or self.model
        self.usage = chunk.get("usage") or self.usage
        self.error = chunk.get("error") or self.error
        text = ""
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            self.finish = choice.get("finish_reason") or self.finish
            self.error = choice.get("error") or self.error
            delta = choice.get("delta") or {}
            if delta.get("content"):
                text += delta["content"]
            if delta.get("refusal"):
                self.refusal.append(delta["refusal"])
            for fragment in delta.get("tool_calls") or []:
                slot = self.calls.setdefault(
                    fragment.get("index", len(self.calls)), {"id": "", "name": "", "arguments": ""}
                )
                slot["id"] = fragment.get("id") or slot["id"]
                function = fragment.get("function") or {}
                if function.get("name") and not slot["name"]:
                    slot["name"] = function["name"]
                slot["arguments"] += function.get("arguments") or ""
            for detail in delta.get("reasoning_details") or []:
                self._add_reasoning(detail)
        self.text.append(text)
        return text

    def _add_reasoning(self, detail: dict[str, Any]) -> None:
        key = (detail.get("type"), detail.get("index", 0))
        for existing in self.reasoning:
            if (existing.get("type"), existing.get("index", 0)) == key:
                for field, value in detail.items():
                    if field in ("text", "summary", "data") and isinstance(value, str):
                        existing[field] = (existing.get(field) or "") + value
                    elif value is not None:
                        existing[field] = value
                return
        self.reasoning.append(dict(detail))

    def result(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(self.text) or None,
            "refusal": "".join(self.refusal) or None,
        }
        if self.calls:
            message["tool_calls"] = [
                {
                    "id": slot["id"],
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["arguments"]},
                }
                for _, slot in sorted(self.calls.items())
            ]
        if self.reasoning:
            message["reasoning_details"] = self.reasoning
        choice = {"index": 0, "finish_reason": self.finish, "message": message}
        return {
            "model": self.model,
            "choices": [choice],
            "usage": self.usage,
            "error": self.error,
        }


# ---------------------------------------------------------------------- schemas


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON schema as strict mode wants it.

    Every property required (optional ones made nullable), ``additionalProperties:
    false`` on every object, references inlined, and keywords strict mode does not
    take (defaults, titles, length and range limits) dropped: pydantic still checks
    them on the reply.
    """
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})
    return _strict(schema, definitions, ())


def _strict(node: Any, definitions: dict[str, Any], seen: tuple[str, ...]) -> Any:
    if isinstance(node, list):
        return [_strict(item, definitions, seen) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        name = str(node["$ref"]).rsplit("/", 1)[-1]
        if name in seen or name not in definitions:
            raise ValueError(f"cannot inline the schema reference {node['$ref']!r}")
        resolved = _strict(definitions[name], definitions, (*seen, name))
        extra = _strict({k: v for k, v in node.items() if k != "$ref"}, definitions, seen)
        return {**resolved, **extra}
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key not in _SCHEMA_KEYWORDS:
            continue
        if key == "properties":
            out[key] = {name: _strict(sub, definitions, seen) for name, sub in value.items()}
        elif key in ("items", "anyOf") or isinstance(value, dict):
            out[key] = _strict(value, definitions, seen)
        else:
            out[key] = value
    if "properties" in out:
        required = set(node.get("required", []))
        out["properties"] = {
            name: sub if name in required else _nullable(sub)
            for name, sub in out["properties"].items()
        }
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    return out


def _allows_null(schema: dict[str, Any]) -> bool:
    kind = schema.get("type")
    if kind == "null" or (isinstance(kind, list) and "null" in kind):
        return True
    return any(isinstance(s, dict) and _allows_null(s) for s in schema.get("anyOf", []))


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    if _allows_null(schema):
        return schema
    if "anyOf" in schema:
        return {**schema, "anyOf": [*schema["anyOf"], {"type": "null"}]}
    description = schema.get("description")
    inner = {k: v for k, v in schema.items() if k != "description"}
    out: dict[str, Any] = {"anyOf": [inner, {"type": "null"}]}
    if description:
        out["description"] = description
    return out


# ---------------------------------------------------------------------- parsing


def parse_reply(text: str, schema: type[BaseModel]) -> BaseModel:
    """The reply as an instance of ``schema``; ValueError saying what is wrong otherwise.

    Tolerates a markdown code fence or chatter around the JSON object, and a null
    where the field has a non-null default (the default is used).
    """
    if not text.strip():
        raise ValueError("the reply was empty")
    data = _extract_json(text)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    return schema.model_validate(_drop_nulls(data, schema))


def _extract_json(text: str) -> Any:
    body = text.strip()
    fenced = _FENCE.search(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        return json.loads(body)
    except ValueError:
        start, end = body.find("{"), body.rfind("}")
        if 0 <= start < end:
            return json.loads(body[start : end + 1])
        raise


def _drop_nulls(data: dict[str, Any], model: type[BaseModel]) -> dict[str, Any]:
    out = dict(data)
    for name, field in model.model_fields.items():
        key = field.alias or name
        if key not in out:
            continue
        value = out[key]
        if value is None:
            if not field.is_required() and not _allows_none(field.annotation):
                del out[key]  # strict mode made it nullable; the default applies
            continue
        nested = _model_in(field.annotation)
        if nested is not None:
            if isinstance(value, dict):
                out[key] = _drop_nulls(value, nested)
            elif isinstance(value, list):
                out[key] = [_drop_nulls(v, nested) if isinstance(v, dict) else v for v in value]
    return out


def _model_in(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for argument in get_args(annotation):
        found = _model_in(argument)
        if found is not None:
            return found
    return None


def _allows_none(annotation: Any) -> bool:
    if annotation is None or annotation is type(None) or annotation is Any:
        return True
    if get_origin(annotation) in (Union, types.UnionType):
        return any(_allows_none(a) for a in get_args(annotation))
    return False
