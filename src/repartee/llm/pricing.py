"""Model prices for cost estimates (``repartee distill --dry-run`` and the spend check).

First-party API list prices in USD per million tokens, from the claude-api skill's
model table cached 2026-11-03. Estimates only: cache reads/writes and long-context
tiers are not modelled.
"""

from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
    # model id: (input, output)
    "claude-fable-5-2": (8.00, 40.00),
    "claude-mythos-5-2": (8.00, 40.00),
    "claude-fable-5-1": (10.00, 50.00),
    "claude-mythos-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}

BATCH_DISCOUNT = 0.5


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, *, batch: bool = False
) -> float:
    """Estimated USD cost of a call; 0.0 for a model not in the table."""
    if model not in PRICES:
        return 0.0
    input_price, output_price = PRICES[model]
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return cost * BATCH_DISCOUNT if batch else cost
