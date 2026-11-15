"""Spend: the provider's own cost when it reports one, else an estimate from tokens."""

from __future__ import annotations

import pytest

from repartee.config import ModelTiers
from repartee.distill.cost import Spend
from repartee.llm.base import LLMResponse, Usage
from repartee.llm.pricing import estimate_cost


def test_spend_prefers_the_reported_cost():
    spend = Spend(ModelTiers())
    spend.add("bulk", LLMResponse(usage=Usage(input_tokens=1000, output_tokens=500,
                                              cost_usd=0.01), model="z-ai/glm-5.3-flash"))
    assert spend.cost_usd == pytest.approx(0.01)
    assert (spend.requests, spend.input_tokens, spend.output_tokens) == (1, 1000, 500)


def test_spend_estimates_without_a_reported_cost():
    spend = Spend(ModelTiers())
    spend.add("bulk", LLMResponse(usage=Usage(input_tokens=1000, output_tokens=500),
                                  model="claude-opus-5-5"))
    assert spend.cost_usd == pytest.approx(estimate_cost("claude-opus-5-5", 1000, 500))
    assert spend.cost_usd > 0
