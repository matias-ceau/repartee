"""A stand-in for ``anthropic.AsyncAnthropic`` that records calls and never touches the network.

Only the surface AnthropicClient uses is modelled: ``beta.messages.create``,
``beta.messages.stream`` and ``beta.messages.batches``. Responses are real SDK
``BetaMessage`` objects so the mapping code sees the shapes it sees in production.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from anthropic.types.beta import BetaMessage


def message(
    content: list[dict[str, Any]],
    *,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    usage: dict[str, Any] | None = None,
    stop_details: dict[str, Any] | None = None,
) -> BetaMessage:
    """A BetaMessage as the API would return it."""
    return BetaMessage.model_validate(
        {
            "id": "msg_stub",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": stop_details,
            "usage": usage or {"input_tokens": 10, "output_tokens": 5},
        }
    )


def text_message(text: str, **kwargs: Any) -> BetaMessage:
    return message([{"type": "text", "text": text}], **kwargs)


class StubStream:
    def __init__(self, reply: BetaMessage) -> None:
        self.reply = reply

    async def __aenter__(self) -> StubStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def __aiter__(self):
        return self._events()

    async def _events(self):
        yield SimpleNamespace(type="message_start")
        for block in self.reply.content:
            if block.type == "text":
                words = block.text.split(" ")
                for i, word in enumerate(words):
                    yield SimpleNamespace(
                        type="text", text=word + (" " if i < len(words) - 1 else "")
                    )
        yield SimpleNamespace(type="message_stop")

    async def get_final_message(self) -> BetaMessage:
        return self.reply


class StubBatches:
    def __init__(self, owner: StubMessages) -> None:
        self.owner = owner
        self.created: list[list[dict[str, Any]]] = []
        self.results_by_custom_id: dict[str, Any] = {}
        self.polls = 0

    async def create(self, *, requests: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.created.append(list(requests))
        for item in requests:
            reply = self.owner.batch_replies.pop(0)
            self.results_by_custom_id[item["custom_id"]] = reply
        return SimpleNamespace(id=f"batch_{len(self.created)}", processing_status="in_progress")

    async def retrieve(self, batch_id: str, **kwargs: Any) -> Any:
        self.polls += 1
        return SimpleNamespace(id=batch_id, processing_status="ended")

    async def results(self, batch_id: str, **kwargs: Any) -> Any:
        items = list(self.results_by_custom_id.items())
        self.results_by_custom_id = {}

        async def iterate():
            for custom_id, reply in reversed(items):  # results come back in any order
                if isinstance(reply, BetaMessage):
                    result = SimpleNamespace(type="succeeded", message=reply)
                elif reply in ("canceled", "expired"):
                    result = SimpleNamespace(type=reply)
                else:
                    error = SimpleNamespace(type="invalid_request_error", message=str(reply))
                    result = SimpleNamespace(type="errored", error=SimpleNamespace(error=error))
                yield SimpleNamespace(custom_id=custom_id, result=result)

        return iterate()


class StubMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.replies: list[BetaMessage | Exception] = []
        self.batch_replies: list[Any] = []
        self.batches = StubBatches(self)

    def _next(self) -> BetaMessage:
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def create(self, **kwargs: Any) -> BetaMessage:
        self.calls.append(kwargs)
        return self._next()

    def stream(self, **kwargs: Any) -> StubStream:
        self.calls.append({**kwargs, "_streamed": True})
        return StubStream(self._next())


class StubSDK:
    """Quacks like AsyncAnthropic for the calls AnthropicClient makes."""

    def __init__(self, *replies: BetaMessage | Exception) -> None:
        self.beta = SimpleNamespace(messages=StubMessages())
        self.beta.messages.replies.extend(replies)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.beta.messages.calls
