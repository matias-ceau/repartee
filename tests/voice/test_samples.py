"""Choosing real samples of the owner's writing. Fictional fixtures."""

from __future__ import annotations

from repartee.voice.registers import register_of
from repartee.voice.samples import pick_samples

from .conftest import HOLDOUT_TEXT, TOPIC_TEXT


def test_pick_samples_prefers_recipient_and_excludes_holdout(env):
    samples = pick_samples(env.corpus, "whatsapp.fr", recipient="Ana Ruiz", k=8)
    assert 0 < len(samples) <= 8
    assert all(s.is_self and not s.holdout for s in samples)
    assert all(register_of(s) == "whatsapp.fr" for s in samples)
    assert all(s.container == "Ana Ruiz" for s in samples)
    assert HOLDOUT_TEXT not in [s.text for s in samples]
    minutes = [s.timestamp.replace(second=0, microsecond=0) for s in samples]
    assert len(set(minutes)) == len(minutes)

    leo = pick_samples(env.corpus, "whatsapp.fr", recipient="Léo", k=5)
    assert [s.container for s in leo] == ["Léo"] * 5


def test_pick_samples_never_returns_holdout_even_when_it_matches(env):
    samples = pick_samples(env.corpus, "whatsapp.fr", topic="zorglub aveugle réservée", k=50)
    assert all(not s.holdout for s in samples)
    assert HOLDOUT_TEXT not in [s.text for s in samples]


def test_pick_samples_topic_and_register(env):
    samples = pick_samples(env.corpus, "whatsapp.fr", topic="vélo canal dimanche", k=4)
    assert samples[0].text == TOPIC_TEXT
    english = pick_samples(env.corpus, "whatsapp.en", k=6)
    assert english and all(s.container == "Sam Patel" for s in english)
    assert pick_samples(env.corpus, "whatsapp.de") == []


def test_pick_samples_is_deterministic(env):
    a = pick_samples(env.corpus, "whatsapp.fr", recipient="Ana Ruiz", topic="ce soir")
    b = pick_samples(env.corpus, "whatsapp.fr", recipient="Ana Ruiz", topic="ce soir")
    assert [s.id for s in a] == [s.id for s in b]


def test_pick_samples_leaves_out_excluded_ids(env):
    first = pick_samples(env.corpus, "whatsapp.fr", recipient="Ana Ruiz", k=4)
    excluded = {s.id for s in first[:2]}
    again = pick_samples(env.corpus, "whatsapp.fr", recipient="Ana Ruiz", k=4, exclude=excluded)
    assert again and not excluded & {s.id for s in again}
