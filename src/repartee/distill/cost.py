"""What a run spends, measured from the usage each response reports."""

from __future__ import annotations

from repartee.config import ModelTiers
from repartee.llm.base import LLMResponse
from repartee.llm.pricing import PRICES, estimate_cost


class Spend:
    """Running totals of requests, tokens and estimated USD for one run."""

    def __init__(self, models: ModelTiers):
        self.models = models
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0

    def model_for(self, tier: str) -> str:
        """The configured model of a tier."""
        return str(getattr(self.models, tier))

    def add(self, tier: str, response: LLMResponse, *, batch: bool = False) -> None:
        """Count one response."""
        usage = response.usage
        model = response.model if response.model in PRICES else self.model_for(tier)
        tokens_in = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
        self.requests += 1
        self.input_tokens += tokens_in
        self.output_tokens += usage.output_tokens
        if usage.cost_usd is not None:  # the provider's own figure
            self.cost_usd += usage.cost_usd
            return
        # prompt-cache reads bill at about a tenth of input, writes at about 1.25 times
        billed_in = round(usage.input_tokens + 0.1 * usage.cache_read_tokens
                          + 1.25 * usage.cache_write_tokens)
        self.cost_usd += estimate_cost(model, billed_in, usage.output_tokens, batch=batch)
