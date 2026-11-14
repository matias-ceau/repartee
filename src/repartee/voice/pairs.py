"""Reply pairs: what someone said in a chat and what the owner answered."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import timedelta

from repartee.schema import Item, ReplyPair, stable_hash
from repartee.voice.registers import _within, bursts, chat_order


def mine_reply_pairs(items: Iterable[Item], *, max_context_messages: int = 4,
                     max_gap_minutes: int = 240) -> list[ReplyPair]:
    """Each owner burst that answers someone else within the gap, with its preceding context.

    Context walks back from the burst over at most `max_context_messages` messages and stops
    at a gap over `max_gap_minutes` or at a holdout item. Bursts holding a holdout are skipped.
    """
    gap = timedelta(minutes=max_gap_minutes)
    pairs: list[ReplyPair] = []
    for container, msgs in chat_order(items).items():
        others = {m.author for m in msgs if not m.is_self}
        recipient = next(iter(others)) if len(others) == 1 else container
        position = {m.id: n for n, m in enumerate(msgs)}
        for burst in bursts(msgs):
            if any(m.holdout for m in burst):
                continue
            start = position[burst[0].id]
            if start == 0:
                continue
            prev = msgs[start - 1]
            if prev.is_self or prev.holdout or not _within(prev, burst[0], gap, strict=False):
                continue
            context = [prev]
            j = start - 2
            while j >= 0 and len(context) < max_context_messages:
                cand = msgs[j]
                if cand.holdout or not _within(cand, context[0], gap, strict=False):
                    break
                context.insert(0, cand)
                j -= 1
            reply_ids = [m.id for m in burst]
            langs = Counter(m.lang for m in burst if m.lang)
            pairs.append(ReplyPair(
                id="rp-" + stable_hash(*reply_ids),
                container=container,
                recipient=recipient,
                timestamp=burst[0].timestamp,
                context="\n".join(f"{m.author}: {m.text}" for m in context),
                reply=[m.text for m in burst],
                context_item_ids=[m.id for m in context],
                reply_item_ids=reply_ids,
                lang=langs.most_common(1)[0][0] if langs else None,
            ))
    return pairs
