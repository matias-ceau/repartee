"""Settings and path resolution.

Two files hold settings:

- ``<home>/repartee.yaml``: who the owner is, where the sources are, model and
  embedding choices. Lives in the private home. Loaded as :class:`HomeConfig`.
- ``$XDG_CONFIG_HOME/repartee/config.yaml``: optional, only says where the home is.

API keys are read from the environment by the SDKs and never stored here.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from repartee.schema import Kind

APP = "repartee"
HOME_CONFIG_NAME = "repartee.yaml"
DEFAULT_MODEL = "claude-opus-5-5"

SourceType = Literal["whatsapp", "vault", "writing", "tastes", "assistant"]
Tier = Literal["bulk", "smart", "chat"]
#: Who serves the models. "openrouter" and "openai" speak the OpenAI chat API;
#: "openai" takes its endpoint and key from OPENAI_BASE_URL / OPENAI_API_KEY, so it
#: also covers any compatible server, including a local one.
Provider = Literal["anthropic", "openrouter", "openai"]
#: "none" turns reasoning off where the provider allows it (OpenRouter); Claude gets "low".
Effort = Literal["none", "low", "medium", "high", "xhigh", "max"]


class OwnerConfig(BaseModel):
    name: str = "Me"
    aliases: list[str] = Field(default_factory=list)  # author names that mean "me"

    def is_self(self, author: str) -> bool:
        """True when a display name found in a source is the owner."""
        wanted = {_normalize_name(n) for n in [self.name, *self.aliases]}
        return _normalize_name(author) in wanted


def _normalize_name(name: str) -> str:
    # Exports decorate names with direction marks and a leading "~" for unsaved contacts.
    cleaned = "".join(ch for ch in name if ch.isprintable() and ch not in "‎‏‪‬")
    return cleaned.strip().lstrip("~").strip().casefold()


class SourceConfig(BaseModel):
    type: SourceType
    path: str  # absolute, ~-relative, or relative to the home
    exclude: list[str] = Field(default_factory=list)  # globs relative to path
    kinds: dict[str, Kind] = Field(default_factory=dict)  # glob -> kind (markdown sources)
    options: dict[str, Any] = Field(default_factory=dict)  # adapter-specific, e.g. date_order


class ModelTiers(BaseModel):
    bulk: str = DEFAULT_MODEL  # extraction over the whole corpus
    smart: str = DEFAULT_MODEL  # consolidation, style cards, grading
    chat: str = DEFAULT_MODEL  # the mirror itself and drafting


class EffortTiers(BaseModel):
    bulk: Effort = "low"
    smart: Effort = "high"
    chat: Effort = "medium"


class OpenRouterConfig(BaseModel):
    """Which OpenRouter hosts may serve a request (its provider preferences)."""

    zdr: bool = False  # only hosts with zero data retention: nothing kept after the reply
    data_collection: Literal["allow", "deny"] = "allow"  # "deny": no host that stores or trains
    sort: Literal["price", "throughput", "latency"] | None = None

    def preferences(self) -> dict[str, Any]:
        """The ``provider`` object of an OpenRouter request; empty when nothing is set."""
        prefs: dict[str, Any] = {}
        if self.zdr:
            prefs["zdr"] = True
        if self.data_collection == "deny":
            prefs["data_collection"] = "deny"
        if self.sort:
            prefs["sort"] = self.sort
        return prefs


class EmbeddingConfig(BaseModel):
    enabled: bool = True
    model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class ChunkingConfig(BaseModel):
    chat_gap_hours: float = 3.0  # silence that starts a new chat chunk
    chat_max_tokens: int = 1500
    note_target_tokens: int = 800


class DistillConfig(BaseModel):
    min_observations_for_page: int = 3
    confirm_above_usd: float = 5.0  # ask before a run estimated to cost more
    concurrency: int = 8


class GitConfig(BaseModel):
    autocommit: bool = True


class HomeConfig(BaseModel):
    owner: OwnerConfig = Field(default_factory=OwnerConfig)
    sources: list[SourceConfig] = Field(default_factory=list)
    provider: Provider = "anthropic"
    models: ModelTiers = Field(default_factory=ModelTiers)  # ids as the provider names them
    effort: EffortTiers = Field(default_factory=EffortTiers)
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    distill: DistillConfig = Field(default_factory=DistillConfig)
    git: GitConfig = Field(default_factory=GitConfig)
    holdout_percent: int = 5

    @classmethod
    def load(cls, path: Path) -> HomeConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def dump(self, path: Path) -> None:
        data = self.model_dump(mode="json", exclude_defaults=False)
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


# ------------------------------------------------------------------------- paths


def _xdg(var: str, fallback: str) -> Path:
    value = os.environ.get(var, "").strip()
    return Path(value).expanduser() if value else Path(fallback).expanduser()


def xdg_config_home() -> Path:
    return _xdg("XDG_CONFIG_HOME", "~/.config")


def xdg_data_home() -> Path:
    return _xdg("XDG_DATA_HOME", "~/.local/share")


def xdg_cache_home() -> Path:
    return _xdg("XDG_CACHE_HOME", "~/.cache")


def resolve_home(explicit: str | Path | None = None) -> Path:
    """Where the private home is: explicit, $REPARTEE_HOME, user config, then XDG data."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("REPARTEE_HOME", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    user_config = xdg_config_home() / APP / "config.yaml"
    if user_config.is_file():
        data = yaml.safe_load(user_config.read_text(encoding="utf-8")) or {}
        if data.get("home"):
            return Path(str(data["home"])).expanduser().resolve()
    return (xdg_data_home() / APP).resolve()


def cache_dir_for(home_root: Path) -> Path:
    """Per-home cache directory under $XDG_CACHE_HOME. Rebuildable content only."""
    digest = hashlib.sha1(str(home_root.resolve()).encode("utf-8")).hexdigest()[:12]
    return xdg_cache_home() / APP / digest


def estimate_tokens(text: str) -> int:
    """Rough token count for budgeting and chunk sizing. Not for billing."""
    return max(1, round(len(text) / 3.5))
