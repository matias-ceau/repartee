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


# ---------------------------------------------------------------------- OpenRouter prices

GLM = "z-ai/glm-5.3-flash"
LISTING = {
    "data": [
        {"id": GLM, "pricing": {"prompt": "0.0000000399", "completion": "0.0000000784"}},
        {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
        {"id": "broken/model", "pricing": {"prompt": "n/a"}},
        {"id": "free/model", "pricing": {"prompt": "0", "completion": "0"}},
        {"name": "no id at all"},
    ]
}


@pytest.fixture
def prices(monkeypatch):
    from repartee.llm import pricing

    monkeypatch.setattr(pricing, "PRICES", dict(pricing.PRICES))
    calls = []

    def fetch(timeout):
        calls.append(timeout)
        return LISTING

    monkeypatch.setattr(pricing, "fetch_openrouter_models", fetch)
    return pricing, calls


def test_openrouter_prices_are_fetched_cached_and_registered(prices, tmp_path):
    pricing, calls = prices
    registered = pricing.register_openrouter_prices([GLM, GLM, "nobody/else"], tmp_path)
    assert registered == [GLM]
    assert pricing.PRICES[GLM] == pytest.approx((0.0399, 0.0784))
    assert pricing.is_priced(GLM)
    assert not pricing.is_priced("nobody/else")
    assert estimate_cost(GLM, 1_000_000, 1_000_000) == pytest.approx(0.0399 + 0.0784)
    assert calls and calls[0] <= 5  # a short timeout

    table = pricing.openrouter_prices(tmp_path)  # within a day: from disk
    assert len(calls) == 1
    assert table[GLM] == pytest.approx((0.0399, 0.0784))
    assert table["free/model"] == (0.0, 0.0)
    assert "openrouter/auto" not in table and "broken/model" not in table


def test_openrouter_prices_refresh_after_a_day(prices, tmp_path):
    import os
    import time

    pricing, calls = prices
    pricing.openrouter_prices(tmp_path)
    [cache_file] = list(tmp_path.rglob("*.json"))
    old = time.time() - 2 * 86400
    os.utime(cache_file, (old, old))
    pricing.openrouter_prices(tmp_path)
    assert len(calls) == 2


def test_openrouter_price_failures_never_raise(prices, tmp_path, monkeypatch):
    import os
    import time

    pricing, _ = prices

    def offline(timeout):
        raise OSError("no network")

    monkeypatch.setattr(pricing, "fetch_openrouter_models", offline)
    assert pricing.register_openrouter_prices([GLM], tmp_path) == []
    assert not pricing.is_priced(GLM)
    assert estimate_cost(GLM, 10, 10) == 0.0

    # A garbage reply is no better than none.
    monkeypatch.setattr(pricing, "fetch_openrouter_models", lambda timeout: ["not", "a", "dict"])
    assert pricing.register_openrouter_prices([GLM], tmp_path) == []

    # An unwritable cache directory still registers what was fetched.
    monkeypatch.setattr(pricing, "fetch_openrouter_models", lambda timeout: LISTING)
    blocked = tmp_path / "file"
    blocked.write_text("x")
    assert pricing.register_openrouter_prices([GLM], blocked) == [GLM]

    # A stale cache beats no prices when the fetch fails.
    monkeypatch.setattr(pricing, "PRICES", dict(PRICES))
    good = tmp_path / "good"
    pricing.openrouter_prices(good)
    [cache_file] = list(good.rglob("*.json"))
    old = time.time() - 5 * 86400
    os.utime(cache_file, (old, old))
    monkeypatch.setattr(pricing, "fetch_openrouter_models", offline)
    assert pricing.register_openrouter_prices([GLM], good) == [GLM]

    # A corrupt cache file is a miss.
    cache_file.write_text("{not json")
    assert pricing.register_openrouter_prices([GLM], good) == []
