"""The default complete_many in the shared protocol."""

import asyncio

import pytest

from repartee.llm.base import LLMClient, LLMError, LLMRequest, LLMResponse, Message


class SlowEcho(LLMClient):
    """Answers in reverse order of arrival and fails on 'fail'."""

    def __init__(self) -> None:
        self.running = 0
        self.peak = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        text = request.messages[-1].text
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0.01 * (5 - int(text[-1])))
        self.running -= 1
        if text.startswith("fail"):
            raise LLMError(text)
        return LLMResponse(text=text)


async def test_base_complete_many_preserves_order_and_returns_errors_in_place():
    client = SlowEcho()
    texts = ["ok0", "fail1", "ok2", "ok3", "fail4"]
    requests = [LLMRequest(messages=[Message(role="user", text=t)]) for t in texts]
    results = await client.complete_many(requests, concurrency=2)
    assert [r.text if isinstance(r, LLMResponse) else f"E:{r}" for r in results] == [
        "ok0",
        "E:fail1",
        "ok2",
        "ok3",
        "E:fail4",
    ]
    assert client.peak <= 2


async def test_base_complete_many_lets_non_llm_errors_through():
    class Broken(LLMClient):
        async def complete(self, request):
            raise RuntimeError("bug")

    with pytest.raises(RuntimeError):
        await Broken().complete_many([LLMRequest(messages=[Message(role="user", text="x")])])
