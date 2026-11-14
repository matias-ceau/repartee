"""Price table and cost estimates."""

import pytest

from repartee.llm.pricing import PRICES, estimate_cost


def test_estimate_cost_known_unknown_batch():
    assert PRICES["claude-opus-5-5"] == (4.0, 20.0)
    cost = estimate_cost("claude-opus-5-5", 1_000_000, 100_000)
    assert cost == pytest.approx(4.0 + 2.0)
    assert estimate_cost("claude-opus-5-5", 1_000_000, 100_000, batch=True) == pytest.approx(3.0)
    assert estimate_cost("some-unknown-model", 10**9, 10**9) == 0.0
    assert estimate_cost("claude-opus-5-5", 0, 0) == 0.0


def test_every_default_model_is_priced():
    from repartee.config import DEFAULT_MODEL

    assert DEFAULT_MODEL in PRICES
    for model in ("claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2"):
        assert model in PRICES
