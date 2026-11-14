"""Ingest: turn sources and the home's own files into normalized items. No LLM, no network.

See docs/superpowers/specs/2026-11-14-mirror-design.md section 8.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from repartee.config import HomeConfig, OwnerConfig, SourceConfig
from repartee.home import Home
from repartee.ingest.base import (
    ParseReport,
    SourceAdapter,
    is_under,
    read_text,
    relative,
    split_frontmatter,
    to_datetime,
    walk_files,
)
from repartee.ingest.chunking import chunk_items
from repartee.ingest.lang import detect_language
from repartee.ingest.markdown import MarkdownAdapter
from repartee.ingest.tastes import TastesAdapter
from repartee.ingest.whatsapp import WhatsAppAdapter
from repartee.schema import ID_PREFIX, Item, Kind, is_holdout, make_item_id

__all__ = [
    "ADAPTERS",
    "ParseReport",
    "SourceAdapter",
    "chunk_items",
    "detect_language",
    "ingest_home",
    "ingest_source",
    "parse_draft",
    "parse_transcript",
]

ADAPTERS: dict[str, SourceAdapter] = {
    "whatsapp": WhatsAppAdapter(),
    "vault": MarkdownAdapter("vault"),
    "writing": MarkdownAdapter("writing"),
    "tastes": TastesAdapter(),
}

#: Only the owner's items of these kinds can be held out for the blind voice test.
HOLDOUT_KINDS = {Kind.MESSAGE, Kind.ESSAY, Kind.JOURNAL, Kind.NOTE}

#: Neighbours (in the same container) a language-less item looks at to inherit one.
LANG_WINDOW = 20


# ----------------------------------------------------------------- post-processing


def _inherit_languages(items: list[Item]) -> None:
    detected = [item.lang for item in items]
    groups: dict[tuple[str, str], list[int]] = {}
    for idx, item in enumerate(items):
        groups.setdefault((item.source, item.container), []).append(idx)
    half = LANG_WINDOW // 2
    for indices in groups.values():
        for pos, idx in enumerate(indices):
            if detected[idx] is not None:
                continue
            # Closest neighbours first, so ties go to the nearest language.
            near = sorted(
                (abs(p - pos), detected[indices[p]])
                for p in range(max(0, pos - half), min(len(indices), pos + half + 1))
                if p != pos and detected[indices[p]] is not None
            )
            if near:
                lang = Counter(lang for _, lang in near).most_common(1)[0][0]
                items[idx].lang = lang
                items[idx].meta["lang_inherited"] = True


def finish_items(items: list[Item], report: ParseReport, holdout_percent: int) -> list[Item]:
    """Drop duplicate ids, detect and inherit languages, flag holdout; update the report."""
    seen: set[str] = set()
    unique: list[Item] = []
    for item in items:
        if item.id in seen:
            continue
        seen.add(item.id)
        unique.append(item)
    if len(unique) < len(items):
        report.warn(
            f"{len(items) - len(unique)} duplicate items (same id) dropped; the same export "
            f"or file is probably present twice"
        )
    for item in unique:
        if item.lang is None:
            item.lang = detect_language(item.text)
    _inherit_languages(unique)
    for item in unique:
        item.holdout = (
            item.is_self and item.kind in HOLDOUT_KINDS and is_holdout(item.id, holdout_percent)
        )
    report.items = len(unique)
    return unique


# ------------------------------------------------------------------------ sources


def ingest_source(
    source: SourceConfig,
    root: Path,
    owner: OwnerConfig,
    *,
    holdout_percent: int = 5,
    skip_under: Path | None = None,
) -> tuple[list[Item], ParseReport]:
    """Parse every file of one configured source. Never raises on bad input."""
    report = ParseReport(source=source.type, path=str(root))
    adapter = ADAPTERS.get(source.type)
    if adapter is None:
        report.warn(f"unknown source type {source.type!r}")
        return [], report
    if not root.exists():
        report.warn(f"{root}: source path does not exist")
        return [], report
    items: list[Item] = []
    for path in adapter.discover(root, source):
        if skip_under is not None and is_under(path, skip_under):
            continue
        report.files += 1
        try:
            for item in adapter.parse(path, root, source, owner, report):
                items.append(item)
        except Exception as exc:  # one unreadable file must not stop the others
            report.warn(f"{relative(path, root)}: could not be read ({type(exc).__name__}: {exc})")
    return finish_items(items, report, holdout_percent), report


# ---------------------------------------------------------- the home's own files

_TURN = re.compile(r"^\*\*(me|mirror)(?::\*\*|\*\*\s*:)[ \t]?(.*)$", re.IGNORECASE)
_FINAL = re.compile(r"^##[ \t]+final[ \t#]*$", re.IGNORECASE)
_TOP_HEADING = re.compile(r"^#{1,2}[ \t]+\S")
_TRANSCRIPT_SOURCE = {"interview": "interviews", "mirror": "conversations"}
_SOURCE_KIND = {"interviews": Kind.INTERVIEW, "conversations": Kind.MIRROR}


def parse_transcript(
    path: Path,
    owner: OwnerConfig,
    *,
    source: str | None = None,
    root: Path | None = None,
) -> list[Item]:
    """One item per `**me:**` / `**mirror:**` turn; only the owner's turns are `is_self`.

    `source` is "interviews" or "conversations" (default: from frontmatter `kind`,
    else the parent folder name). Ids are keyed by the path relative to `root`
    (default: the file name) and the turn index.
    """
    fm, body = split_frontmatter(read_text(path), path.name)
    fm_kind = str(fm.get("kind", "")).strip().lower()
    if source is None:
        source = _TRANSCRIPT_SOURCE.get(fm_kind) or (
            path.parent.name if path.parent.name in _SOURCE_KIND else "interviews"
        )
    kind = Kind(fm_kind) if fm_kind in _TRANSCRIPT_SOURCE else _SOURCE_KIND.get(
        source, Kind.INTERVIEW
    )
    rel = relative(path, root) if root is not None else path.name
    when = to_datetime(fm.get("date"))
    title = str(fm["title"]) if fm.get("title") else None
    mode = fm.get("mode")

    turns: list[tuple[str, list[str]]] = []
    for line in body.split("\n"):
        m = _TURN.match(line)
        if m:
            turns.append((m.group(1).lower(), [m.group(2)]))
        elif turns:
            turns[-1][1].append(line)

    items: list[Item] = []
    for index, (role, lines) in enumerate(turns):
        text = "\n".join(lines).strip()
        if not text:
            continue
        mine = role == "me"
        items.append(
            Item(
                id=make_item_id(ID_PREFIX[source], rel, index),
                source=source,
                kind=kind,
                container=rel,
                author=owner.name if mine else "mirror",
                is_self=mine,
                timestamp=when,
                text=text,
                title=title,
                seq=index,
                meta={"role": role, "mode": mode, "path": rel},
            )
        )
    return items


def parse_draft(path: Path, owner: OwnerConfig, *, root: Path | None = None) -> list[Item]:
    """Items from the `## Final` section of a draft file, one per blank-line-separated block.

    A draft whose frontmatter says `accepted: false` yields nothing.
    """
    fm, body = split_frontmatter(read_text(path), path.name)
    if fm.get("accepted") is False:
        return []
    final: list[str] | None = None
    for line in body.split("\n"):
        if final is None:
            if _FINAL.match(line):
                final = []
        elif _TOP_HEADING.match(line):
            break
        else:
            final.append(line)
    if not final:
        return []
    blocks = [b.strip() for b in re.split(r"\n[ \t]*\n", "\n".join(final))]
    rel = relative(path, root) if root is not None else path.name
    meta = {
        "path": rel,
        "register_name": fm.get("register_name"),
        "recipient": fm.get("recipient"),
        "intent": fm.get("intent"),
    }
    when = to_datetime(fm.get("date"))
    title = str(fm.get("title") or fm.get("intent") or "") or None
    items = []
    for block in blocks:
        if not block:
            continue
        index = len(items)
        items.append(
            Item(
                id=make_item_id(ID_PREFIX["drafts"], rel, "final", index),
                source="drafts",
                kind=Kind.DRAFT,
                container=rel,
                author=owner.name,
                is_self=True,
                timestamp=when,
                text=block,
                title=title,
                seq=index,
                meta=dict(meta),
            )
        )
    return items


def _ingest_internal(
    folder: Path,
    source: str,
    parse: Callable[[Path], list[Item]],
    holdout_percent: int,
) -> tuple[list[Item], ParseReport]:
    report = ParseReport(source=source, path=str(folder))
    items: list[Item] = []
    for path in walk_files(folder, {".md"}, []):
        report.files += 1
        try:
            items.extend(parse(path))
        except Exception as exc:
            rel = relative(path, folder)
            report.warn(f"{rel}: could not be read ({type(exc).__name__}: {exc})")
    return finish_items(items, report, holdout_percent), report


def ingest_home(home: Home, config: HomeConfig) -> tuple[list[Item], list[ParseReport]]:
    """Every configured source plus the home's interviews/, conversations/ and drafts/."""
    owner = config.owner
    percent = config.holdout_percent
    items: list[Item] = []
    reports: list[ParseReport] = []
    for source in config.sources:
        root = home.source_path(source)
        # Never read the self-model back: inside the home skip self/, outside skip the home.
        skip = home.self_dir if is_under(root, home.root) else home.root
        got, report = ingest_source(source, root, owner, holdout_percent=percent, skip_under=skip)
        items.extend(got)
        reports.append(report)
    internal = [
        (home.interviews_dir, "interviews",
         lambda p: parse_transcript(p, owner, source="interviews", root=home.interviews_dir)),
        (home.conversations_dir, "conversations",
         lambda p: parse_transcript(p, owner, source="conversations",
                                    root=home.conversations_dir)),
        (home.drafts_dir, "drafts", lambda p: parse_draft(p, owner, root=home.drafts_dir)),
    ]
    for folder, name, parse in internal:
        got, report = _ingest_internal(folder, name, parse, percent)
        items.extend(got)
        reports.append(report)
    return items, reports
