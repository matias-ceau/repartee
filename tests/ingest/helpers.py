"""Small helpers shared by the ingest tests."""

from __future__ import annotations

from pathlib import Path

from repartee.config import OwnerConfig, SourceConfig
from repartee.ingest.base import ParseReport


def write(path: Path, text: str) -> Path:
    """Write text (UTF-8), creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def parse_with(adapter, path: Path, root: Path, source: SourceConfig, owner: OwnerConfig):
    """Run one adapter on one file; return (items, report)."""
    report = ParseReport(source=source.type, path=str(root))
    items = list(adapter.parse(path, root, source, owner, report))
    return items, report
