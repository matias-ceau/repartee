"""Adapter protocol, parse report and helpers shared by the ingest adapters."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, Field

from repartee.config import OwnerConfig, SourceConfig
from repartee.schema import Item

#: Most samples of unplaced lines kept per file in a report.
MAX_SAMPLES_PER_FILE = 5


class ParseReport(BaseModel):
    """What one source's parse produced, skipped and could not place."""

    source: str
    path: str
    files: int = 0
    items: int = 0
    #: Lines that could not be placed anywhere (dropped). Real exports show here
    #: where the format tables need extending.
    unparsed_lines: int = 0
    #: Lines recognised and deliberately dropped, by reason ("system", "media", "deleted").
    skipped: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def skip(self, reason: str, n: int = 1) -> None:
        """Count a recognised line that produces no item."""
        self.skipped[reason] = self.skipped.get(reason, 0) + n

    def warn(self, message: str) -> None:
        """Add a warning (exact duplicates are kept once)."""
        if message not in self.warnings:
            self.warnings.append(message)


class SourceAdapter(Protocol):
    """Turns the files of one source type into items."""

    name: str

    def discover(self, root: Path, source: SourceConfig) -> list[Path]: ...

    def parse(
        self,
        path: Path,
        root: Path,
        source: SourceConfig,
        owner: OwnerConfig,
        report: ParseReport,
    ) -> Iterator[Item]: ...


# ------------------------------------------------------------------ file walking


@lru_cache(maxsize=256)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def glob_match(rel: str, pattern: str) -> bool:
    """Match a posix relative path against a glob with `**`; slash-free patterns match names."""
    pattern = pattern.strip().lstrip("/")
    if pattern.endswith("/"):
        pattern += "**"
    if "/" not in pattern:
        return any(_glob_regex(pattern).match(part) for part in rel.split("/"))
    return bool(_glob_regex(pattern).match(rel))


def relative(path: Path, root: Path) -> str:
    """Posix path of `path` relative to `root` (the file name when root is the file)."""
    if path == root:
        return path.name
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def walk_files(root: Path, suffixes: set[str], exclude: list[str]) -> list[Path]:
    """Files under root with one of the suffixes, sorted; follows links, skips hidden and loops."""
    if root.is_file():
        return [root] if root.suffix.lower() in suffixes else []
    if not root.is_dir():
        return []
    found: list[Path] = []
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".") or Path(name).suffix.lower() not in suffixes:
                continue
            path = Path(dirpath) / name
            rel = relative(path, root)
            if any(glob_match(rel, pattern) for pattern in exclude):
                continue
            found.append(path)
    return found


def is_under(path: Path, folder: Path) -> bool:
    """True when path, after resolving links, is inside folder."""
    try:
        return path.resolve().is_relative_to(folder.resolve())
    except OSError:
        return False


# ------------------------------------------------------------------- text helpers


def read_text(path: Path, report: ParseReport | None = None) -> str:
    """Decode a text file: UTF-8 (with or without BOM), UTF-16 with BOM, else lossy UTF-8."""
    return decode_bytes(path.read_bytes(), str(path), report)


def decode_bytes(data: bytes, label: str, report: ParseReport | None = None) -> str:
    """Decode bytes as read_text does."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
        if report is not None:
            report.warn(f"{label}: not valid UTF-8; undecodable bytes were replaced")
    return text.replace("\r\n", "\n").replace("\r", "\n")


_FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?\n)?(?:---|\.\.\.)[ \t]*(?:\n|\Z)", re.DOTALL)


def split_frontmatter(
    text: str, label: str = "", report: ParseReport | None = None
) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from a markdown body. Bad YAML is a warning, not an error."""
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    body = text[match.end() :]
    try:
        data = yaml.safe_load(match.group(1) or "") or {}
    except yaml.YAMLError as exc:
        if report is not None:
            report.warn(f"{label}: frontmatter is not valid YAML ({str(exc).splitlines()[0]})")
        return {}, body
    if not isinstance(data, dict):
        return {}, body
    return {str(k): v for k, v in data.items()}, body


def to_datetime(value: Any) -> datetime | None:
    """Naive datetime from a YAML date/datetime or an ISO-like string; None if not a date."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text).replace(tzinfo=None)
        except ValueError:
            m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
            if m:
                try:
                    return datetime(int(m[1]), int(m[2]), int(m[3]))
                except ValueError:
                    return None
    return None
