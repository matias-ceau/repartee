"""Style distance: a cheap, automatic proxy for the blind test.

For a sample of the owner's non-held-out texts, a neutral brief is written and the mirror
drafts from it; each draft is measured against the register's style profile with the same
statistics the drafter checks itself with. Next to it stands the floor: how far the owner's
own held-out texts (which the profile never saw) are from the same profile. A mirror whose
number is near the floor writes, by these measures, like the owner.
"""

from __future__ import annotations

import logging
import random
from statistics import fmean

from repartee.evals.common import (
    Saver,
    Unit,
    bounded,
    draft_unit,
    holdout_units,
    make_brief,
    reason,
    settings,
    units,
)
from repartee.runtime import Runtime
from repartee.voice.stylometry import StyleProfile, distance

log = logging.getLogger(__name__)

SEED = 0
OWNER_TEXTS = 200  # held-out texts measured per register for the floor (no model calls)

MEANING = ("Mean normalized distance from each register's measured style profile: 0 is "
           "identical, 1 is a major deviation. ':mirror' is over the mirror's drafts, ':owner' "
           "over the owner's own held-out texts, which is the floor to compare against.")


def _interleave(pools: dict[str, list[Unit]], n: int) -> list[Unit]:
    """n units taking turns across registers, each pool shuffled by the seed."""
    queues = {}
    for name in sorted(pools):
        ordered = sorted(pools[name], key=lambda u: u.key)
        random.Random(f"{SEED}:{name}").shuffle(ordered)
        queues[name] = ordered
    out: list[Unit] = []
    while len(out) < n and any(queues.values()):
        for name in sorted(queues):
            if queues[name] and len(out) < n:
                out.append(queues[name].pop(0))
    return out


async def style_distance_report(rt: Runtime, *, n: int = 20) -> dict[str, float]:
    """'<register>:mirror' and '<register>:owner' distances, plus 'overall:…'; saved in evals/.

    Registers without a style profile (run `repartee voice`) are left out; a register with
    no successful draft has no ':mirror' key. About 2n model calls.
    """
    profiles: dict[str, StyleProfile] = rt.voice_builder().profiles()
    if not profiles:
        return {}

    owner_detail: dict[str, dict[str, float | int]] = {}
    owner_all: list[float] = []
    report: dict[str, float] = {}
    for name in sorted(profiles):
        held = sorted(holdout_units(rt.corpus, name), key=lambda u: u.key)
        random.Random(SEED).shuffle(held)
        scores = [distance(profiles[name], u.texts()) for u in held[:OWNER_TEXTS]]
        if scores:
            report[f"{name}:owner"] = fmean(scores)
            owner_detail[name] = {"texts": len(scores), "mean": fmean(scores)}
            owner_all += scores

    pools = {name: units(rt.corpus, holdout=False, register_name=name) for name in profiles}
    chosen = _interleave(pools, n)
    drafts: list[dict] = []
    failed: list[dict[str, str]] = []
    if chosen:
        llm = rt.require_llm()
        drafter = rt.drafter()
        owner = rt.config.owner.name

        async def one(unit: Unit) -> tuple[str, list[str]]:
            brief = await make_brief(llm, owner, unit)
            return brief, await draft_unit(drafter, unit, brief)

        results = await bounded([lambda u=unit: one(u) for unit in chosen])
        for unit, result in zip(chosen, results, strict=True):
            if isinstance(result, Exception):
                log.warning("style distance: skipped %s: %s", unit.key, reason(result))
                failed.append({"item_id": unit.key, "register_name": unit.register_name,
                               "reason": reason(result)})
                continue
            brief, generated = result
            drafts.append({
                "item_ids": unit.item_ids, "register_name": unit.register_name,
                "recipient": unit.recipient, "brief": brief, "source": unit.texts(),
                "generated": generated,
                "distance": distance(profiles[unit.register_name], generated),
            })

    mirror_all = []
    for name in sorted(profiles):
        scores = [d["distance"] for d in drafts if d["register_name"] == name]
        if scores:
            report[f"{name}:mirror"] = fmean(scores)
            mirror_all += scores
    if mirror_all:
        report["overall:mirror"] = fmean(mirror_all)
    if owner_all:
        report["overall:owner"] = fmean(owner_all)
    report = dict(sorted(report.items()))

    Saver(rt.home, "style").save({
        "eval": "style_distance",
        "settings": settings(rt, n=n, seed=SEED, registers=sorted(profiles)),
        "meaning": MEANING,
        "report": report,
        "owner": owner_detail,
        "drafts": drafts,
        "failed": failed,
    })
    return report
