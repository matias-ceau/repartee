"""Style distance: the mirror's drafts and the owner's held-out texts against the same profile."""

from __future__ import annotations

import json

from repartee.evals import style_distance_report
from repartee.llm import FakeLLM
from repartee.runtime import Runtime
from repartee.sync import run_voice


async def test_report_has_mirror_and_owner_numbers_per_register(rt: Runtime,
                                                                fake: FakeLLM) -> None:
    registers = await run_voice(rt, commit=False)
    assert {"whatsapp.fr", "whatsapp.en"} <= set(registers)
    report = await style_distance_report(rt, n=6)
    for register in registers:
        assert report[f"{register}:mirror"] >= 0.0
        assert report[f"{register}:owner"] >= 0.0
    assert {"overall:mirror", "overall:owner"} <= set(report)
    assert len(fake.calls("draft")) == 6
    assert len(fake.calls("brief")) == 6


async def test_prompts_are_not_held_out_and_results_are_saved(rt: Runtime,
                                                              fake: FakeLLM) -> None:
    await run_voice(rt, commit=False)
    report = await style_distance_report(rt, n=4)
    held_out = [i.text for i in rt.corpus.iter_items(holdout=True) if i.is_self]
    for request in fake.calls("brief") + fake.calls("draft"):
        seen = "\n".join(m.text for m in request.messages)
        assert not any(text in seen for text in held_out if len(text) > 25)
    [path] = sorted(rt.home.evals_dir.glob("style-*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["eval"] == "style_distance"
    assert data["settings"]["n"] == 4
    assert data["report"] == report
    assert len(data["drafts"]) == 4
    assert {"item_ids", "register_name", "brief", "generated", "distance"} <= set(
        data["drafts"][0])
    assert data["owner"]["whatsapp.fr"]["texts"] > 0


async def test_without_profiles_the_report_is_empty(rt: Runtime, fake: FakeLLM) -> None:
    assert await style_distance_report(rt, n=3) == {}
    assert not fake.calls("draft")
