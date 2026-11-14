"""A disk cache in front of any LLMClient, for the pipeline stages.

One JSON file per request under ``directory/<2 hex>/<key>.json``. The key is a hash
of everything that shapes the reply: the model the tier maps to, purpose, system
blocks, messages (including replayed provider content), tool specs, output schema
and max_tokens. Only successful ``complete`` results are stored; ``stream`` passes
through untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from repartee.llm.base import LLMClient, LLMError, LLMEvent, LLMRequest, LLMResponse

CACHE_VERSION = 1


class CachedLLM(LLMClient):
    """Wraps an LLMClient; identical ``complete`` requests are answered from disk."""

    def __init__(
        self,
        inner: LLMClient,
        directory: Path,
        model_for: Callable[[str], str] | None = None,
    ) -> None:
        self.inner = inner
        self.directory = Path(directory)
        self.model_for = model_for
        self.hits = 0
        self.misses = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Answer from disk when possible, else ask the inner client and store the reply."""
        key = self.key(request)
        cached = self._load(key, request)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        response = await self.inner.complete(request)
        self._store(key, response)
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        """Not cached: streaming is for live conversation."""
        async for event in self.inner.stream(request):
            yield event

    async def complete_many(
        self,
        requests: Sequence[LLMRequest],
        *,
        concurrency: int = 8,
        batch: bool = False,
    ) -> list[LLMResponse | LLMError]:
        """Serve hits from disk and send only the misses to the inner client."""
        results: list[LLMResponse | LLMError | None] = [None] * len(requests)
        keys = [self.key(request) for request in requests]
        missing: list[int] = []
        for index, (request, key) in enumerate(zip(requests, keys, strict=True)):
            cached = self._load(key, request)
            if cached is None:
                missing.append(index)
            else:
                self.hits += 1
                results[index] = cached
        if missing:
            self.misses += len(missing)
            fresh = await self.inner.complete_many(
                [requests[i] for i in missing], concurrency=concurrency, batch=batch
            )
            for index, result in zip(missing, fresh, strict=True):
                results[index] = result
                if isinstance(result, LLMResponse):
                    self._store(keys[index], result)
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------ keys and files

    def key(self, request: LLMRequest) -> str:
        """Stable hash of everything in the request that shapes the reply."""
        schema = request.output_schema
        payload: dict[str, Any] = {
            "v": CACHE_VERSION,
            "model": self.model_for(request.tier) if self.model_for else request.tier,
            "purpose": request.purpose,
            "system": [[block.text, block.cache] for block in request.system],
            "messages": [
                {**message.model_dump(mode="json"), "raw": message.raw}
                for message in request.messages
            ],
            "tools": [tool.model_dump(mode="json") for tool in request.tools],
            "schema": schema.model_json_schema() if schema else None,
            "max_tokens": request.max_tokens,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=repr)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def _load(self, key: str, request: LLMRequest) -> LLMResponse | None:
        path = self._path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            parsed_data = data.pop("parsed", None)
            raw = data.pop("raw", None)
            response = LLMResponse.model_validate(data)
            if parsed_data is not None and request.output_schema is not None:
                response.parsed = request.output_schema.model_validate(parsed_data)
            response.message.raw = raw
            return response
        except (OSError, ValueError, ValidationError, AttributeError, TypeError):
            return None  # missing or unreadable entries are simply misses

    def _store(self, key: str, response: LLMResponse) -> None:
        data = response.model_dump(mode="json", exclude={"parsed"})
        data["parsed"] = response.parsed.model_dump(mode="json") if response.parsed else None
        data["raw"] = response.message.raw
        try:
            text = json.dumps(data, ensure_ascii=False, indent=1)
        except (TypeError, ValueError):
            return  # provider content that is not plain JSON is not worth caching
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".tmp{os.getpid()}")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
