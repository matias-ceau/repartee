"""Style cards and stored profiles. Fictional fixtures, FakeLLM only."""

from __future__ import annotations

import json

from repartee.llm.base import LLMError
from repartee.llm.fake import FakeLLM
from repartee.voice.cards import VoiceBuilder
from repartee.voice.prompts import StyleCardOut
from repartee.voice.stylometry import StyleProfile

from .conftest import HOLDOUT_TEXT

DESCRIPTION = "J'écris par petites rafales, sans majuscules, et je ris en mdr."


def described() -> FakeLLM:
    return FakeLLM().on("style_card", StyleCardOut(description=DESCRIPTION))


async def test_build_writes_profiles_and_cards_and_skips_thin_registers(env):
    fake = described()
    written = await VoiceBuilder(env.home, env.corpus, env.model, fake).build()

    assert sorted(written) == ["whatsapp.en", "whatsapp.fr"]  # essay.en has only 5 texts
    voice = env.home.derived_dir / "voice"
    stored = json.loads((voice / "whatsapp.fr.json").read_text(encoding="utf-8"))
    assert stored["register_name"] == "whatsapp.fr"
    assert stored["n_texts"] == 249  # 210 + 40 owner messages, minus the held-out one
    assert (voice / "whatsapp.fr" / "ana-ruiz.json").is_file()  # 209 messages to Ana
    assert not (voice / "whatsapp.fr" / "léo.json").exists()  # only 40
    assert not (voice / "essay.en.json").exists()

    model = env.reload_model()  # build saved the cards
    card = model.voice_card("whatsapp.fr")
    assert card is not None and model.voice_card("essay.en") is None
    assert DESCRIPTION in card
    assert "Starts messages in lowercase" in card  # a describe() statement
    assert "mdr" in card  # laughter
    assert "## Notes" in card
    assert HOLDOUT_TEXT not in card

    calls = fake.calls("style_card")
    assert len(calls) == 2
    assert all(c.tier == "smart" and c.output_schema is StyleCardOut for c in calls)
    for call in calls:
        prompt = "\n".join(m.text for m in call.messages)
        assert HOLDOUT_TEXT not in prompt
        assert "zorglub" not in prompt


async def test_build_only_named_registers(env):
    written = await VoiceBuilder(env.home, env.corpus, env.model, None).build(["whatsapp.en"])
    assert written == ["whatsapp.en"]
    assert env.reload_model().voice_card("whatsapp.fr") is None


async def test_build_without_llm_is_statistics_only(env):
    written = await VoiceBuilder(env.home, env.corpus, env.model, None).build()
    assert "whatsapp.fr" in written
    card = env.reload_model().voice_card("whatsapp.fr")
    assert "Starts messages in lowercase" in card
    assert "How it reads" not in card


async def test_build_survives_a_failed_description(env):
    fake = FakeLLM().on("style_card", LLMError)
    written = await VoiceBuilder(env.home, env.corpus, env.model, fake).build()
    assert "whatsapp.fr" in written
    assert "Starts messages in lowercase" in env.reload_model().voice_card("whatsapp.fr")


async def test_build_preserves_owner_notes(env):
    await VoiceBuilder(env.home, env.corpus, env.model, None).build()
    path = env.home.self_dir / "voice" / "whatsapp.fr.md"
    path.write_text(path.read_text(encoding="utf-8") + "Never use semicolons.\n",
                    encoding="utf-8")

    model = env.reload_model()
    await VoiceBuilder(env.home, env.corpus, model, described()).build()
    card = env.reload_model().voice_card("whatsapp.fr")
    assert "Never use semicolons." in card
    assert DESCRIPTION in card
    assert card.index(DESCRIPTION) < card.index("## Notes") < card.index("Never use semicolons.")


async def test_profiles_read_back(env):
    builder = VoiceBuilder(env.home, env.corpus, env.model, None)
    await builder.build()
    profile = builder.profile("whatsapp.fr")
    assert isinstance(profile, StyleProfile) and profile.form == "chat"
    assert profile.metrics["lowercase_start_rate"] > 0.9
    assert builder.profile("essay.en") is None
    by_name = builder.recipient_profile("whatsapp.fr", "Ana Ruiz")
    by_key = builder.recipient_profile("whatsapp.fr", "people/ana-ruiz")
    assert by_name is not None and by_name == by_key
    assert by_name.n_texts == 209
    assert builder.recipient_profile("whatsapp.fr", "Léo") is None
