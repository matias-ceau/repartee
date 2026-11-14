"""The LLM layer: protocol, Anthropic client, scripted fake, disk cache, prices."""

from __future__ import annotations

from pathlib import Path

from repartee.config import HomeConfig
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
from repartee.llm.pricing import PRICES, estimate_cost

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
    "SystemBlock",
    "TextDelta",
    "ToolCall",
    "ToolCallEvent",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "estimate_cost",
    "get_client",
    "tool_call",
]


def get_client(config: HomeConfig, cache_dir: Path | None = None) -> LLMClient:
    """The configured client; cached under ``cache_dir/llm`` when a cache dir is given.

    Raises LLMUnavailable with a one-line fix when no credentials are found.
    """
    client = AnthropicClient(config.models, config.effort)
    if cache_dir is None:
        return client

    def model_for(tier: str) -> str:
        return f"{getattr(config.models, tier)}@{getattr(config.effort, tier)}"

    return CachedLLM(client, Path(cache_dir) / "llm", model_for=model_for)
