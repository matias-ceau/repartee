"""Real samples of the owner's writing to show the drafter: same register, best fit first.

Only the owner's non-holdout items are ever candidates.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from repartee.schema import Item, Kind, SearchFilters
from repartee.store.corpus import Corpus
from repartee.store.embed import fold
from repartee.store.search import name_key
from repartee.voice.registers import LONGFORM_CHANNELS, is_chat_register, register_of
from repartee.voice.stylometry import tokenize

#: Score weights. A text written to the same person outweighs one on the same subject,
#: which outweighs a recent one; chat samples far from the usual length lose a little.
RECIPIENT_WEIGHT = 3.0
TOPIC_WEIGHT = 2.0
RECENCY_WEIGHT = 0.5
TYPICAL_WEIGHT = 0.3
TOPIC_CHUNKS = 30  # chunks fetched from the corpus for topical similarity


def channel_filters(register_name: str) -> SearchFilters:
    """Filters selecting the owner's items of a register's channel (language not included)."""
    channel = register_name.rsplit(".", 1)[0]
    if channel in LONGFORM_CHANNELS:
        return SearchFilters(author="self", kinds=[Kind(channel)])
    return SearchFilters(author="self", sources=[channel], kinds=[Kind.MESSAGE])


def register_items(corpus: Corpus, register_name: str) -> list[Item]:
    """The owner's non-holdout, non-empty items of a register (accepted drafts included)."""
    filters = [channel_filters(register_name)]
    if filters[0].kinds != [Kind.DRAFT]:
        filters.append(SearchFilters(author="self", kinds=[Kind.DRAFT]))
    out = []
    for f in filters:
        for item in corpus.iter_items(f, holdout=False):
            if not item.holdout and item.text.strip() and register_of(item) == register_name:
                out.append(item)
    return out


def recipient_names(recipient: str | Sequence[str] | None) -> list[str]:
    """Names to look a recipient up by; a page key ("people/ana-ruiz") counts by its slug."""
    if recipient is None:
        return []
    raw = [recipient] if isinstance(recipient, str) else list(recipient)
    out: list[str] = []
    for name in raw:
        name = name.rsplit("/", 1)[-1].strip() if name else ""
        if name and name_key(name) and name not in out:
            out.append(name)
    return out


def _same_person(author: str, keys: set[tuple[str, ...]]) -> bool:
    words = tuple(name_key(author).split())
    return any(words[: len(k)] == k for k in keys)


def recipient_containers(corpus: Corpus, names: Sequence[str]) -> dict[str, float]:
    """Chats with this person: 1.0 for a one-to-one chat, 0.5 for a group they write in."""
    keys = {tuple(name_key(n).split()) for n in names if name_key(n)}
    if not keys:
        return {}
    found: set[str] = set()
    for name in names:
        for item in corpus.iter_items(SearchFilters(participant=name, author="other",
                                                    kinds=[Kind.MESSAGE])):
            if _same_person(item.author, keys):
                found.add(item.container)
    out: dict[str, float] = {}
    for container in sorted(found):
        others = {name_key(i.author) for i in corpus.iter_items(
            SearchFilters(containers=[container], author="other", kinds=[Kind.MESSAGE]))}
        out[container] = 1.0 if len(others) == 1 else 0.5
    for name in names:  # a chat named after them, whoever wrote in it
        chat = SearchFilters(containers=[name], kinds=[Kind.MESSAGE])
        if next(corpus.iter_items(chat), None) is not None:
            out[name] = 1.0
    return out


def _words(text: str) -> set[str]:
    return {fold(t) for t in tokenize(text) if len(t) > 1}


def _topical(corpus: Corpus, register_name: str, topic: str | None,
             candidates: dict[str, Item]) -> dict[str, float]:
    """0..1 per candidate id: rank of its chunk in a search, plus word overlap with the topic."""
    if not topic or not topic.strip():
        return {}
    hits = corpus.search(topic, channel_filters(register_name), k=TOPIC_CHUNKS)
    wanted = _words(topic)
    out: dict[str, float] = {}
    for rank, hit in enumerate(hits):
        base = 1.0 - rank / len(hits)
        for item_id in hit.chunk.item_ids:
            item = candidates.get(item_id)
            if item is None or item_id in out:
                continue
            overlap = len(wanted & _words(item.text)) / len(wanted) if wanted else 0.0
            out[item_id] = 0.6 * base + 0.4 * overlap
    return out


def pick_samples(corpus: Corpus, register_name: str, *,
                 recipient: str | Sequence[str] | None = None, topic: str | None = None,
                 k: int = 8) -> list[Item]:
    """Up to k of the owner's texts in the register, best first, never two from one minute.

    Score: written in a chat with the recipient, then topical similarity to `topic`, then
    a small recency bonus; chat messages far from the owner's usual length rank lower.
    """
    if k <= 0:
        return []
    candidates = register_items(corpus, register_name)
    if not candidates:
        return []
    by_id = {c.id: c for c in candidates}
    names = recipient_names(recipient)
    to_them = recipient_containers(corpus, names) if names else {}
    topical = _topical(corpus, register_name, topic, by_id)
    times = [c.timestamp for c in candidates if c.timestamp]
    first, last = (min(times), max(times)) if times else (None, None)
    chat = is_chat_register(register_name)
    lengths = {c.id: len(tokenize(c.text)) for c in candidates} if chat else {}
    usual = statistics.median(lengths.values()) if lengths else 0.0

    scored: list[tuple[float, str]] = []
    for item in candidates:
        score = RECIPIENT_WEIGHT * to_them.get(item.container, 0.0)
        score += TOPIC_WEIGHT * topical.get(item.id, 0.0)
        if item.timestamp and first and last and last > first:
            score += RECENCY_WEIGHT * (item.timestamp - first) / (last - first)
        if chat:
            score -= TYPICAL_WEIGHT * abs(math.log((lengths[item.id] + 1) / (usual + 1)))
        scored.append((-score, item.id))
    scored.sort()

    picked: list[Item] = []
    minutes: set[object] = set()
    texts: set[str] = set()
    for _, item_id in scored:
        item = by_id[item_id]
        minute = item.timestamp.replace(second=0, microsecond=0) if item.timestamp else None
        text = fold(" ".join(item.text.split()))
        if (minute is not None and minute in minutes) or text in texts:
            continue
        picked.append(item)
        texts.add(text)
        if minute is not None:
            minutes.add(minute)
        if len(picked) >= k:
            break
    return picked
