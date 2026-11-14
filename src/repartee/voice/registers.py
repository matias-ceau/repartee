"""Registers (a kind of text in a language, e.g. "whatsapp.fr") and chat bursts."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

from repartee.schema import Item, Kind

#: Channels written as documents rather than as chat messages.
LONGFORM_CHANNELS = frozenset({"essay", "journal", "note", "interview", "mirror", "draft"})

#: Kinds whose channel is the kind name itself.
_KIND_CHANNELS = {Kind.ESSAY, Kind.JOURNAL, Kind.NOTE, Kind.INTERVIEW, Kind.MIRROR, Kind.DRAFT}

#: Owner messages closer together than this, with nobody else in between, form one burst.
BURST_GAP = timedelta(minutes=2)


def register_of(item: Item) -> str | None:
    """'<channel>.<lang>' for the owner's writing; None if not the owner's or no language."""
    if not item.is_self:
        return None
    if item.kind == Kind.DRAFT:
        name = item.meta.get("register_name")
        if isinstance(name, str) and name:
            return name
    if item.kind == Kind.MESSAGE:
        channel = item.source
    elif item.kind in _KIND_CHANNELS:
        channel = item.kind.value
    else:  # medical observations, taste lists: not writing voice
        return None
    if not item.lang:
        return None
    return f"{channel}.{item.lang.lower()}"


def group_by_register(items: Iterable[Item]) -> dict[str, list[Item]]:
    """The owner's non-holdout items grouped by register, in input order."""
    groups: dict[str, list[Item]] = {}
    for item in items:
        if item.holdout:
            continue
        name = register_of(item)
        if name is not None:
            groups.setdefault(name, []).append(item)
    return groups


def is_chat_register(register_name: str) -> bool:
    """True for message registers (any chat source), false for documents."""
    channel = register_name.rsplit(".", 1)[0] if "." in register_name else register_name
    return channel not in LONGFORM_CHANNELS


def chat_order(items: Iterable[Item]) -> dict[str, list[Item]]:
    """Chat messages grouped by container, each in conversation order."""
    by_container: dict[str, list[Item]] = {}
    for item in items:
        if item.kind == Kind.MESSAGE:
            by_container.setdefault(item.container, []).append(item)
    for msgs in by_container.values():
        msgs.sort(key=lambda i: (i.seq, i.timestamp or datetime.min, i.id))
    return by_container


def bursts(items: Iterable[Item], *, gap: timedelta = BURST_GAP) -> list[list[Item]]:
    """Owner message bursts: consecutive owner messages in one chat, each under `gap` apart.

    Pass the other participants' messages too, so that their turns split bursts.
    Messages without a timestamp never join a burst.
    """
    out: list[list[Item]] = []
    for msgs in chat_order(items).values():
        current: list[Item] = []
        prev: Item | None = None
        for item in msgs:
            if not item.is_self:
                current = []
            elif current and prev is current[-1] and _within(prev, item, gap, strict=True):
                current.append(item)
            else:
                current = [item]
                out.append(current)
            prev = item
    return out


def _within(a: Item, b: Item, gap: timedelta, *, strict: bool) -> bool:
    if a.timestamp is None or b.timestamp is None:
        return not strict
    delta = b.timestamp - a.timestamp
    return delta < gap if strict else delta <= gap
