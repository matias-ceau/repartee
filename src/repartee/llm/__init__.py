"""The LLM layer: protocol, Anthropic and OpenAI-style clients, fake, disk cache, prices."""

from __future__ import annotations

from pathlib import Path

from repartee.config import HomeConfig
from repartee.llm import pricing
from repartee.llm.anthropic_client import AnthropicClient
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
    SystemBlock,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    Usage,
)
from repartee.llm.cache import CachedLLM
from repartee.llm.fake import FakeLLM, tool_call
from repartee.llm.openai_client import OpenAIClient
from repartee.llm.pricing import PRICES, estimate_cost, is_priced

__all__ = [
    "PRICES",
    "AnthropicClient",
    "CachedLLM",
    "Done",
    "FakeLLM",
    "LLMClient",
    "LLMError",
    "LLMEvent",
    "LLMRefusal",
    "LLMRequest",
    "LLMResponse",
    "LLMUnavailable",
    "Message",
    "OpenAIClient",
    "SystemBlock",
    "TextDelta",
    "ToolCall",
    "ToolCallEvent",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "estimate_cost",
    "get_client",
    "is_priced",
    "pricing",
    "tool_call",
]


def get_client(config: HomeConfig, cache_dir: Path | None = None) -> LLMClient:
    """The configured provider's client; cached under ``cache_dir/llm`` when a cache dir is given.

    Raises LLMUnavailable with a one-line fix when no credentials are found, or when
    the models in repartee.yaml cannot belong to the provider. Under OpenRouter, the
    prices of the configured models are registered for cost estimates first (best
    effort, from a list cached in ``cache_dir`` for a day), so an estimate works
    even before there is a key.
    """
    tiers = ("bulk", "smart", "chat")
    client: LLMClient
    if config.provider == "anthropic":
        client = AnthropicClient(config.models, config.effort)

        def model_for(tier: str) -> str:
            return f"{getattr(config.models, tier)}@{getattr(config.effort, tier)}"

    else:
        if config.provider == "openrouter":
            _check_openrouter_models(config)
            if cache_dir is not None:
                pricing.register_openrouter_prices(
                    [getattr(config.models, tier) for tier in tiers], Path(cache_dir)
                )
        routing = config.openrouter.preferences() if config.provider == "openrouter" else {}
        client = OpenAIClient(config.models, config.effort, provider=config.provider,
                              routing=routing)

        def model_for(tier: str) -> str:
            model = f"{config.provider}:{getattr(config.models, tier)}"
            if config.provider == "openrouter":  # effort only reaches OpenRouter
                model += f"@{getattr(config.effort, tier)}"
            return model

    if cache_dir is None:
        return client
    return CachedLLM(client, Path(cache_dir) / "llm", model_for=model_for)


def _check_openrouter_models(config: HomeConfig) -> None:
    wrong = sorted({m for m in config.models.model_dump().values() if "/" not in m})
    if wrong:
        raise LLMUnavailable(
            f"provider is openrouter but models: in repartee.yaml names {', '.join(wrong)}, "
            "which is not an OpenRouter id; set bulk, smart and chat under models: to "
            "OpenRouter ids such as z-ai/glm-5.3-flash."
        )
