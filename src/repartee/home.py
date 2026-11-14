"""The private home: where one owner's material and self-model live.

This module only knows the layout. Creating a home is in ``scaffold.py``,
safety checks are in ``guard.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from repartee.config import (
    HOME_CONFIG_NAME,
    HomeConfig,
    SourceConfig,
    cache_dir_for,
    resolve_home,
)


class HomeNotFound(Exception):
    """No home at the resolved location. Run ``repartee init``."""


@dataclass(frozen=True)
class Home:
    root: Path

    # ------------------------------------------------------------ private, durable
    @property
    def config_path(self) -> Path:
        return self.root / HOME_CONFIG_NAME

    @property
    def sources_dir(self) -> Path:
        return self.root / "sources"

    @property
    def interviews_dir(self) -> Path:
        return self.root / "interviews"

    @property
    def conversations_dir(self) -> Path:
        return self.root / "conversations"

    @property
    def drafts_dir(self) -> Path:
        return self.root / "drafts"

    @property
    def self_dir(self) -> Path:
        return self.root / "self"

    @property
    def derived_dir(self) -> Path:
        return self.root / "derived"

    @property
    def evals_dir(self) -> Path:
        return self.root / "evals"

    # ------------------------------------------------------------ cache, rebuildable
    @property
    def cache_dir(self) -> Path:
        return cache_dir_for(self.root)

    @property
    def index_path(self) -> Path:
        return self.cache_dir / "index.db"

    @property
    def llm_cache_dir(self) -> Path:
        return self.cache_dir / "llm"

    # ------------------------------------------------------------------- helpers
    def exists(self) -> bool:
        return self.config_path.is_file()

    def load_config(self) -> HomeConfig:
        if not self.exists():
            raise HomeNotFound(f"No Repartee home at {self.root}. Run `repartee init {self.root}`.")
        return HomeConfig.load(self.config_path)

    def source_path(self, source: SourceConfig) -> Path:
        """Absolute path of a configured source (relative paths are relative to the home)."""
        path = Path(source.path).expanduser()
        return path if path.is_absolute() else (self.root / path)

    def ensure_dirs(self) -> None:
        for directory in (
            self.sources_dir,
            self.interviews_dir,
            self.conversations_dir,
            self.drafts_dir,
            self.self_dir,
            self.derived_dir,
            self.evals_dir,
            self.cache_dir,
            self.llm_cache_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def open(cls, explicit: str | Path | None = None) -> Home:
        """Resolve and return an existing home, or raise HomeNotFound."""
        home = cls(resolve_home(explicit))
        if not home.exists():
            raise HomeNotFound(f"No Repartee home at {home.root}. Run `repartee init {home.root}`.")
        return home
