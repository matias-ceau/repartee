"""The sample persona, ingested, with a scripted model. All data is fictional."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from repartee.llm import FakeLLM
from repartee.llm.base import LLMRequest
from repartee.runtime import Runtime, open_runtime
from repartee.store.embed import HashEmbedder
from repartee.sync import run_ingest
from repartee.voice.prompts import DraftOut, StyleCardOut

PERSONA = Path(__file__).resolve().parents[2] / "examples" / "persona"
TEXT_BLOCK = re.compile(r"<text>\n(.*?)\n</text>", re.DOTALL)


def neutral_brief(request: LLMRequest) -> dict:
    """A brief that names the text's length in characters and nothing of its wording."""
    text = TEXT_BLOCK.search(request.messages[-1].text).group(1)
    return {"brief": f"Pass on a piece of news of about {len(text)} characters."}


def scripted_model() -> FakeLLM:
    fake = FakeLLM()
    fake.on("brief", neutral_brief)
    fake.on("draft", DraftOut(variants=[["ouais ça me va", "je te dis ça demain"]]))
    fake.on("revise", DraftOut(variants=[["ouais ça me va", "je te dis ça demain"]]))
    fake.on("style_card", StyleCardOut(description="Short, lowercase, dry."))
    return fake


@pytest.fixture
def rt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Runtime:
    """The persona in tmp_path/persona, ingested, with a scripted fake model."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    home = tmp_path / "persona"
    shutil.copytree(PERSONA, home)
    runtime = open_runtime(home, llm=scripted_model(), embedder=HashEmbedder())
    run_ingest(runtime, commit=False)
    yield runtime
    runtime.close()


@pytest.fixture
def fake(rt: Runtime) -> FakeLLM:
    return rt.llm  # type: ignore[return-value]
