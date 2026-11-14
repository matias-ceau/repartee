"""One real call to the Anthropic API. Skipped by default; run with ``-m live``."""

import pytest

from repartee.config import HomeConfig
from repartee.llm import get_client
from repartee.llm.base import LLMRequest, Message


@pytest.mark.live
async def test_live_one_line_call():
    client = get_client(HomeConfig())
    response = await client.complete(
        LLMRequest(
            tier="bulk",
            purpose="live-test",
            max_tokens=2000,
            messages=[Message(role="user", text="Reply with the single word: pong")],
        )
    )
    assert "pong" in response.text.lower()
    assert response.usage.output_tokens > 0
