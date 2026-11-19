"""Model prices for cost estimates (``repartee distill --dry-run`` and the spend check).

First-party API list prices in USD per million tokens, from the claude-api skill's
model table cached 2026-11-03. Estimates only: cache reads/writes and long-context
tiers are not modelled.

OpenRouter models are priced from its public model list
(``register_openrouter_prices``), fetched at most once a day, best effort: when
it cannot be had, those models stay unpriced and ``is_priced`` says so.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_CACHE_NAME = "openrouter-prices.json"
PRICE_MAX_AGE = 86_400.0  # seconds: refetch the list once a day
FETCH_TIMEOUT = 3.0


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, *, batch: bool = False
) -> float:
    """Estimated USD cost of a call; 0.0 for a model not in the table."""
    if model not in PRICES:
        return 0.0
    input_price, output_price = PRICES[model]
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return cost * BATCH_DISCOUNT if batch else cost


def is_priced(model: str) -> bool:
    """True when ``estimate_cost`` knows this model's price (else it reports 0.0)."""
    return model in PRICES


# ---------------------------------------------------------------------- OpenRouter


def fetch_openrouter_models(timeout: float) -> Any:
    """The public OpenRouter model list (no key needed), as parsed JSON."""
    request = urllib.request.Request(OPENROUTER_MODELS_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as reply:  # noqa: S310 (fixed https URL)
        return json.load(reply)


def _per_million(value: Any) -> float | None:
    try:
        price = float(value) * 1_000_000
    except (TypeError, ValueError):
        return None
    return price if price >= 0 else None  # routers list -1: the price depends on the route


def _parse_listing(listing: Any) -> dict[str, tuple[float, float]]:
    table: dict[str, tuple[float, float]] = {}
    entries = listing.get("data") if isinstance(listing, dict) else None
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        pricing = entry.get("pricing")
        if not isinstance(pricing, dict):
            continue
        prompt = _per_million(pricing.get("prompt"))
        completion = _per_million(pricing.get("completion"))
        if prompt is not None and completion is not None:
            table[entry["id"]] = (round(prompt, 9), round(completion, 9))
    return table


def _read_cache(path: Path) -> dict[str, tuple[float, float]] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): (float(v[0]), float(v[1])) for k, v in data["prices"].items()}
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        return None


def _write_cache(path: Path, table: dict[str, tuple[float, float]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".tmp{os.getpid()}")
        temporary.write_text(json.dumps({"prices": table}), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        pass  # an unwritable cache only means fetching again next time


def openrouter_prices(
    cache_dir: Path, *, max_age: float = PRICE_MAX_AGE, timeout: float = FETCH_TIMEOUT
) -> dict[str, tuple[float, float]]:
    """OpenRouter prices in USD per million tokens, from a day-old disk copy or the network.

    Never raises: on any failure it returns the last copy on disk, however old, or {}.
    """
    path = Path(cache_dir) / OPENROUTER_CACHE_NAME
    cached = _read_cache(path)
    try:
        fresh = cached is not None and time.time() - path.stat().st_mtime < max_age
    except OSError:
        fresh = False
    if fresh and cached is not None:
        return cached
    try:
        table = _parse_listing(fetch_openrouter_models(timeout))
    except Exception:  # noqa: BLE001 (best effort: a price list must never stop a command)
        table = {}
    if not table:
        return cached or {}
    _write_cache(path, table)
    return table


def register_openrouter_prices(models: Iterable[str], cache_dir: Path) -> list[str]:
    """Add the listed prices of ``models`` to PRICES; returns the models now priced."""
    table = openrouter_prices(cache_dir)
    registered: list[str] = []
    for model in dict.fromkeys(models):
        if model in table:
            PRICES[model] = table[model]
            registered.append(model)
    return registered
