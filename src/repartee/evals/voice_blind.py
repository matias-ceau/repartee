"""The blind voice test: can the owner tell their own real text from the mirror's?

For held-out texts (never seen by the voice code), a model writes a neutral brief of what
each says; the drafter writes it again from the brief alone, to the same person in the same
register and the same shape; the owner sees both in an order fixed by the seed and picks
the real one. The score is the share of pairs where they picked the mirror's: 50% means
they could not tell the two apart.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from pydantic import BaseModel, Field

from repartee.evals.common import (
    Saver,
    Unit,
    bounded,
    draft_unit,
    holdout_units,
    make_brief,
    reason,
    sample,
    settings,
)
from repartee.runtime import Runtime

log = logging.getLogger(__name__)

MEANING = ("Score is the share of judged pairs where the owner picked the mirror's text as "
           "their own. 50% means the owner could not tell the mirror from their real writing "
           "(indistinguishable); 0% means they always spotted it. Few judged pairs make the "
           "number noisy: see counts.")


class BlindPair(BaseModel):
    """One real text (or burst) of the owner's and the mirror's version of it."""

    item_id: str
    register_name: str
    brief: str
    real: list[str]
    generated: list[str]
    real_is_a: bool
    #: Every item on the real side (several for a burst), first one is item_id.
    item_ids: list[str] = Field(default_factory=list)
    recipient: str | None = None
    #: The conversation the real text answered, as given to the drafter.
    context: str | None = None

    @property
    def a(self) -> list[str]:
        return self.real if self.real_is_a else self.generated

    @property
    def b(self) -> list[str]:
        return self.generated if self.real_is_a else self.real


def real_goes_first(seed: int, item_id: str) -> bool:
    """The A/B side of the real text: fixed by the seed and the item, a fair coin otherwise."""
    return random.Random(f"{seed}:{item_id}").random() < 0.5


class BlindTest:
    """Prepare pairs, record the owner's picks, score them and save the result."""

    def __init__(self, rt: Runtime, *, register_name: str | None = None, n: int = 10,
                 seed: int = 0):
        self.rt = rt
        self.register_name = register_name
        self.n = n
        self.seed = seed
        self.pairs: list[BlindPair] = []
        #: Items that could not be made into a pair: {"item_id", "register_name", "reason"}.
        self.skipped: list[dict[str, str]] = []
        self._picks: dict[str, bool] = {}
        self._saver = Saver(rt.home, "voice")

    async def prepare(self) -> list[BlindPair]:
        """Brief and draft n held-out texts (two model calls each, plus a rare revision)."""
        llm = self.rt.require_llm()
        drafter = self.rt.drafter()
        owner = self.rt.config.owner.name
        chosen = sample(holdout_units(self.rt.corpus, self.register_name), self.n, self.seed)

        async def make(unit: Unit) -> BlindPair:
            brief = await make_brief(llm, owner, unit)
            generated = await draft_unit(drafter, unit, brief)
            return BlindPair(
                item_id=unit.key, register_name=unit.register_name, brief=brief,
                real=unit.texts(), generated=generated,
                real_is_a=real_goes_first(self.seed, unit.key), item_ids=unit.item_ids,
                recipient=unit.recipient, context=unit.context)

        results = await bounded([lambda u=unit: make(u) for unit in chosen])
        self.pairs, self.skipped, self._picks = [], [], {}
        for unit, result in zip(chosen, results, strict=True):
            if isinstance(result, BlindPair):
                self.pairs.append(result)
            else:
                log.warning("blind test: skipped %s: %s", unit.key, reason(result))
                self.skipped.append({"item_id": unit.key, "register_name": unit.register_name,
                                     "reason": reason(result)})
        return list(self.pairs)

    def record(self, pair: BlindPair, picked_a: bool) -> None:
        """The owner picked A (True) or B (False) as their real text; a new pick replaces one."""
        if not any(p.item_id == pair.item_id for p in self.pairs):
            raise ValueError(f"{pair.item_id} is not a pair of this test")
        self._picks[pair.item_id] = bool(picked_a)

    def _counts(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for pair in self.pairs:
            if pair.item_id not in self._picks:
                continue
            fooled = int(self._picks[pair.item_id] != pair.real_is_a)
            for key in (pair.register_name, "overall"):
                entry = counts.setdefault(key, {"judged": 0, "fooled": 0})
                entry["judged"] += 1
                entry["fooled"] += fooled
        return counts

    def score(self) -> dict[str, float]:
        """Share fooled per register and "overall", over judged pairs; {} before any pick."""
        return {key: c["fooled"] / c["judged"] for key, c in self._counts().items()}

    def save(self) -> Path:
        """Write evals/voice-<timestamp>.json in the home (real texts included: home only)."""
        pairs = []
        for pair in self.pairs:
            picked = self._picks.get(pair.item_id)
            pairs.append({
                **pair.model_dump(),
                "picked_a": picked,
                "fooled": None if picked is None else picked != pair.real_is_a,
            })
        return self._saver.save({
            "eval": "voice_blind",
            "settings": settings(self.rt, register_name=self.register_name, n=self.n,
                                 seed=self.seed),
            "meaning": MEANING,
            "score": self.score(),
            "counts": self._counts(),
            "pairs": pairs,
            "skipped": self.skipped,
        })
