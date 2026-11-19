"""Style statistics. Fixtures are invented; expected values are hand-computed in comments."""

import math
from datetime import datetime, timedelta

import pytest

from repartee.schema import Item, Kind
from repartee.voice.stylometry import (
    build_profile,
    compare,
    describe,
    distance,
    find_emoji,
    find_laughter,
    profile_items,
    tokenize,
)

T0 = datetime(2024, 3, 1, 12, 0)

# words per message (tokens):  3, 3, 2, 4, 1, 1, 3, 3, 1, 4   -> 25 words
CHAT = [
    "salut ça va",               # lowercase start
    "Oui, très bien.",           # capital, final full stop, 1 comma
    "trop cool!!",               # exclamation, doubled punctuation
    "tu viens ce soir?",         # question
    "nooon 😂😂",                 # stretched word, 2 emoji
    "hahaha 👍🏽",                 # laughter "haha", 1 emoji (skin tone)
    "je sais pas...",            # ellipsis, not a full stop
    "C'est GÉNIAL, vraiment",    # capital, caps word, 1 comma; "C'est" is one word
    "ok.",                       # final full stop
    "👨‍👩‍👧 on arrive à 8h",        # 1 emoji (ZWJ family); first letter "o"; "8h" one word
]


def test_tokenize_keeps_elisions_hyphens_and_splits_cjk():
    assert tokenize("C'est peut-être 8.30 à https://x.org/a ok") == [
        "C'est", "peut-être", "8.30", "à", "ok"]
    assert tokenize("今日は") == ["今", "日", "は"]
    assert tokenize("") == []


def test_chat_metrics_exact():
    p = build_profile("whatsapp.fr", CHAT, burst_sizes=[2, 1, 3, 1, 3])
    assert p.form == "chat"
    assert p.n_texts == 10
    assert p.n_words == 25
    expected = {
        "words_per_message_median": 3.0,        # sorted 1,1,1,2,3,3,3,3,4,4
        "words_per_message_p90": 4.0,           # nearest rank: 9th of 10
        "messages_per_burst_mean": 2.0,         # 10 / 5
        "lowercase_start_rate": 0.8,            # "Oui" and "C'est" are capitalised
        "final_period_rate": 0.2,               # "bien." and "ok."
        "ellipsis_rate": 0.1,
        "exclamation_rate": 0.1,
        "question_rate": 0.1,
        "repeated_punct_rate": 0.1,
        "emoji_per_message": 0.4,               # 2 + 1 + 1 over 10
        "elongation_rate": 0.1,                 # "nooon"
        "caps_word_rate": 1 / 23,               # GÉNIAL / cased words of 2+ letters
        "commas_per_100_words": 8.0,            # 2 / 25 * 100
    }
    assert set(p.metrics) == set(expected)
    for key, value in expected.items():
        assert p.metrics[key] == pytest.approx(value), key
    assert p.top_emoji == [("😂", 0.5), ("👍🏽", 0.25), ("👨‍👩‍👧", 0.25)]
    assert p.laughter == [("haha", 10.0)]


LONG = (
    "I wrote this last year. It was hard; I was tired.\n\n"
    "Why do we write? Maybe — as my teacher said (long ago) — to think.\n\n"
    "We keep going, slowly, and I like it."
)


def test_longform_metrics_exact():
    p = build_profile("essay.en", [LONG])
    # sentences: 5, 6, 4, 9, 8 words -> 32 words, mean 6.4
    sd = math.sqrt((1.4**2 + 0.4**2 + 2.4**2 + 2.6**2 + 1.6**2) / 5)
    expected = {
        "words_per_sentence_mean": 6.4,
        "words_per_sentence_sd": sd,
        "sentences_per_paragraph_mean": 5 / 3,
        "first_person_rate": 4 / 32,            # I, I, my, I
        "question_rate": 0.2,                   # 1 of 5 sentences
        "commas_per_100_words": 2 / 32 * 100,
        "semicolons_per_1000_words": 1 / 32 * 1000,
        "dashes_per_1000_words": 2 / 32 * 1000,
        "parentheses_per_1000_words": 1 / 32 * 1000,
    }
    assert p.form == "longform"
    assert (p.n_texts, p.n_words) == (1, 32)
    assert set(p.metrics) == set(expected)
    for key, value in expected.items():
        assert p.metrics[key] == pytest.approx(value), key


def _msg(i, text, minutes, *, is_self=True, container="Ana", holdout=False):
    return Item(id=f"m-{container}-{i}", source="whatsapp", kind=Kind.MESSAGE,
                container=container, author="Moi" if is_self else container, is_self=is_self,
                timestamp=T0 + timedelta(minutes=minutes), text=text, lang="fr", seq=i,
                holdout=holdout)


def test_burst_sizes_from_items():
    owner = [
        _msg(1, "a", 0), _msg(2, "b", 0.5),          # burst of 2
        _msg(4, "d", 2),                             # after Ana: burst of 1
        _msg(5, "e", 5), _msg(6, "f", 5.5),          # 3 min gap: new burst of 2
        _msg(1, "g", 0, container="Bob"),
        _msg(2, "h", 1, container="Bob", holdout=True),  # burst with a holdout: not counted
    ]
    others = [_msg(3, "c", 1, is_self=False)]
    p = profile_items("whatsapp.fr", owner, others)
    assert p.n_texts == 6                            # holdout never measured
    assert p.metrics["messages_per_burst_mean"] == pytest.approx(5 / 3)
    assert p.lang_mix == {"fr": 1.0}


def test_emoji_and_laughter_tables():
    assert find_emoji("👍🏽👨‍👩‍👧 et 🇫🇷 ok ❤️ :)") == ["👍🏽", "👨‍👩‍👧", "🇫🇷", "❤️"]
    assert find_emoji("plain text, 100% ©") == []
    assert find_laughter("hahaha HAHA mdrrr jajaja ahah lol xD kkkk ptdr") == [
        "haha", "haha", "mdr", "jaja", "ahah", "lol", "xd", "kkk", "ptdr"]
    assert find_laughter("aha Mahalo ha chat") == []
    # laughter is not counted as a stretched word, real stretching is
    p = build_profile("whatsapp.fr", ["kkkk", "mdrrr", "ouiii"])
    assert p.metrics["elongation_rate"] == pytest.approx(1 / 3)


def test_distinctive_phrases_against_background():
    owner = ["en vrai c'est bien", "on se voit demain", "en vrai je sais pas", "c'est bien ça",
             "en vrai on verra", "je sais pas trop", "en vrai oui", "on se voit ce soir",
             "c'est bien", "oui demain"]                                  # 36 words
    background = ["de toute façon c'est bien", "de toute façon on verra", "on se voit demain",
                  "de toute façon oui", "je sais pas", "c'est bien ça"]
    p = build_profile("whatsapp.fr", owner, background=background)
    phrases = dict(p.distinctive_phrases)
    assert p.distinctive_phrases[0][0] == "en vrai"
    assert phrases["en vrai"] == pytest.approx(4 / 36 * 1000)          # per 1000 words
    assert "de toute façon" not in phrases                            # theirs, not mine
    assert "c'est bien" not in phrases                                # shared
    assert "vrai" not in phrases and "en" not in phrases              # subsumed by "en vrai"


TERSE = ["ok", "jsuis là", "on y va", "trop bien", "grave", "t'es où", "ça marche", "à toute",
         "vas-y", "jsp"] * 3
FORMAL = ["Hello Ana, I wanted to let you know that I will arrive at the station tomorrow "
          "morning at eight. Please tell me whether that works for you."]


def test_compare_flags_long_formal_draft_against_terse_profile():
    p = build_profile("whatsapp.fr", TERSE)
    devs = {d.metric: d for d in compare(p, FORMAL)}
    for metric in ("words_per_message_median", "lowercase_start_rate", "final_period_rate"):
        assert devs[metric].severity == "major", metric
    assert "words_per_message_p90" not in devs        # same advice as the median
    assert devs["words_per_message_median"].expected == 2
    assert devs["words_per_message_median"].actual == 27
    msgs = " ".join(d.message for d in devs.values())
    assert "about 2 words" in msgs
    assert "Start messages in lowercase" in msgs
    assert "Drop the final full stop" in msgs
    for d in devs.values():
        assert "_" not in d.message


def test_compare_quiet_for_in_style_text():
    p = build_profile("whatsapp.fr", TERSE)
    assert compare(p, ["on y va", "trop bien"]) == []
    assert compare(p, []) == []


def test_compare_longform_semicolons_and_sentence_length():
    p = build_profile("essay.en", ["I like short lines. They help me. I write them often. "
                                   "It is my way. I keep to it."] * 3)
    draft = ("Writing, which is a practice that many people regard as solitary; is, in my "
             "considered opinion, a deeply social act; one that connects us to everyone who "
             "has ever read anything at all; and that is why it matters so very much to me.")
    devs = {d.metric: d for d in compare(p, [draft])}
    assert devs["words_per_sentence_mean"].severity == "major"
    assert devs["words_per_sentence_mean"].message.startswith("Much too long sentences")
    assert devs["semicolons_per_1000_words"].message.startswith("Drop the semicolons")


def test_distance_zero_for_same_texts_and_grows():
    p = build_profile("whatsapp.fr", TERSE)
    assert distance(p, TERSE) == pytest.approx(0.0)
    near = distance(p, ["on y va maintenant stp"])
    far = distance(p, FORMAL)
    assert 0 < near < far


def test_describe_is_plain_and_skips_thin_metrics():
    p = build_profile("whatsapp.fr", TERSE, burst_sizes=[1] * 30)
    lines = describe(p)
    assert "Messages are short: usually 2 words, rarely over 2." in lines
    assert "Starts messages in lowercase (100%)." in lines
    assert "Never ends a message with a full stop." in lines
    assert "Usually says it in a single message rather than several in a row." in lines
    for line in lines:
        assert "_" not in line and line.endswith(".")
    assert describe(build_profile("whatsapp.fr", TERSE[:10])) == []   # under 20 texts

    long = build_profile("essay.en", [LONG] * 7)            # 35 sentences, 21 paragraphs
    lines = describe(long)
    assert any(line.startswith("Sentences average 6.4 words") for line in lines)
    assert "Paragraphs average 1.7 sentences." in lines


def test_handles_empty_mixed_script_and_rtl_text():
    texts = ["", "   ", "مرحبا، كيف حالك؟", "今日は良い天気ですね。散歩に行きましょう！",
             "😂😂👍🏽", "ok", "Je suis à Tokyo 🗼, it's great", "ㅋㅋㅋ"]
    for form_register in ("whatsapp.xx", "essay.xx"):
        p = build_profile(form_register, texts, background=texts[2:])
        assert p.n_texts == 6
        assert p.n_words > 0
        describe(p)
        compare(p, texts)
        compare(p, ["", "😂"])
        assert distance(p, texts[2:]) >= 0
    chat = build_profile("whatsapp.xx", texts)
    assert chat.metrics["question_rate"] == pytest.approx(1 / 6)      # Arabic ؟
    assert chat.metrics["exclamation_rate"] == pytest.approx(1 / 6)   # full-width ！
    assert chat.metrics["lowercase_start_rate"] == pytest.approx(1 / 2)  # only cased starts
    assert chat.laughter == [("ㅋㅋ", 100 / 6)]
    empty = build_profile("whatsapp.xx", [])
    assert (empty.n_texts, empty.metrics) == (0, {})
    assert describe(empty) == [] and compare(empty, ["hi"]) == [] and distance(empty, []) == 0
