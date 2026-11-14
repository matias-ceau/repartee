"""Style statistics: measure how the owner writes in a register and how far a draft is from it.

No LLM. Script-neutral: words are runs of letters, marks and digits (elisions and hyphens
stay inside a word); Han and kana count one character per word, since those scripts do
not separate words with spaces.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Sequence
from typing import Literal

import regex
from pydantic import BaseModel, Field

from repartee.schema import Item
from repartee.voice.registers import bursts, is_chat_register

# ------------------------------------------------------------------ text pieces

_SEG = r"\p{Han}\p{Hiragana}\p{Katakana}"
_W = rf"[[\p{{L}}\p{{M}}\p{{N}}]--[{_SEG}]]"
_TOKEN = regex.compile(rf"(?V1)\p{{N}}+(?:[.,:/]\p{{N}}+)+|[{_SEG}]|{_W}+(?:['’\-]{_W}+)*")
_URL = regex.compile(r"https?://\S+|www\.\S+")
_PICT = regex.compile(r"\p{Extended_Pictographic}|\p{Regional_Indicator}|⃣")
_NOT_EMOJI = {"©", "®", "™"}  # pictographic, but typed as text symbols
_GRAPHEME = regex.compile(r"\X")
_LETTER = regex.compile(r"\p{L}")
_ELONGATED = regex.compile(r"(\p{L})\1{2,}")
_ELLIPSIS = regex.compile(r"\.{2,}|…|⋯|。{2,}")
_FINAL_STOP = regex.compile(r"(?:(?<!\.)\.|(?<!。)。|[۔।])$")
_EXCLAMATION = regex.compile(r"[!！¡]")
_QUESTION = regex.compile(r"[?？¿؟]")
_REPEATED = regex.compile(r"[!?！？¡¿؟]{2,}")
_COMMA = regex.compile(r"[,，、،]")
_SEMICOLON = regex.compile(r"[;；؛]")
_DASH = regex.compile(r"—|–|(?<=\S[ \t])-(?=[ \t])")
_PAREN = regex.compile(r"[(（]")
_HEADING = regex.compile(r"(?m)^[ \t]*#{1,6}[ \t].*$")
_PARAGRAPH = regex.compile(r"\n[ \t]*\n")
_LIST_BREAK = regex.compile(r"\n(?=[ \t]*(?:[-*+•][ \t]|\d+[.)][ \t]))")
_SENTENCE_END = regex.compile(r"[.!?…؟।]+[\"'”’»)\]]*(?=\s|$)|[。！？]+[」』”’)）]*")
_QUESTION_END = regex.compile(r"[?？؟][\"'”’»)\]」』）]*\s*$")

#: Laughter forms across languages: (canonical form, pattern matched on a whole casefolded word).
LAUGHTER: list[tuple[str, str]] = [
    ("haha", r"a?(?:h+a+){2,}h*"),
    ("ahah", r"a+h+(?:a+h+)+a*"),
    ("hehe", r"(?:h+[eé]+){2,}h*"),
    ("hihi", r"(?:h+i+){2,}h*"),
    ("lol", r"l+o+l+(?:o+l+)*"),
    ("mdr", r"m+d+r+"),
    ("ptdr", r"p+t+d+r+"),
    ("jaja", r"(?:j+a+){2,}j*"),
    ("jeje", r"(?:j+e+){2,}j*"),
    ("kkk", r"k{3,}"),
    ("rsrs", r"(?:r+s+){2,}"),
    ("xd", r"x+d+"),
    ("lmao", r"l+m+a+o+"),
    ("rofl", r"r+o+f+l+"),
    ("www", r"w{3,}"),
    ("ㅋㅋ", r"ㅋ{2,}"),
    ("ㅎㅎ", r"ㅎ{2,}"),
    ("ههه", r"ه{3,}"),
    ("хаха", r"(?:х+а+){2,}х*"),
    ("555", r"5{3,}"),
]
_LAUGHTER = [(canon, regex.compile(pattern)) for canon, pattern in LAUGHTER]

#: First-person singular words in common languages (casefolded; elided forms by their prefix).
#: Ambiguous words are left out (Italian "ma" means "but").
FIRST_PERSON = frozenset({
    "i", "me", "my", "mine", "myself", "im", "ive",
    "je", "j", "moi", "mon", "mes", "m", "moi-même",
    "yo", "mí", "mi", "mis", "conmigo", "mío", "mía",
    "io", "mio", "mia", "miei", "mie",
    "eu", "meu", "minha", "meus", "minhas", "comigo",
    "ich", "mich", "mir", "mein", "meine", "meinen", "meinem", "meiner", "meines",
    "ik", "mij", "mijn",
    "я", "меня", "мне", "мой", "моя", "моё", "мои",
    "أنا", "私", "僕", "俺", "我",
})


def tokenize(text: str) -> list[str]:
    """Words of a text, URLs removed; Han and kana one character per word."""
    return _TOKEN.findall(_URL.sub(" ", text))


def find_emoji(text: str) -> list[str]:
    """Emoji as written, one per grapheme cluster (skin tones, ZWJ sequences, flags count once)."""
    return [g for g in _GRAPHEME.findall(text) if g not in _NOT_EMOJI and _PICT.search(g)]


def _laugh(token: str) -> str | None:
    folded = token.casefold()
    for canon, pattern in _LAUGHTER:
        if pattern.fullmatch(folded):
            return canon
    return None


def find_laughter(text: str) -> list[str]:
    """Laughter words in the text, normalized to their canonical form ("HAHAHA" -> "haha")."""
    return [c for c in (_laugh(t) for t in tokenize(text)) if c]


def _is_elongated(token: str) -> bool:
    return _laugh(token) is None and bool(_ELONGATED.search(token.casefold()))


def _is_first_person(token: str) -> bool:
    folded = token.casefold()
    head = regex.split(r"['’]", folded, maxsplit=1)[0]
    return folded in FIRST_PERSON or head in FIRST_PERSON


def _cased(ch: str) -> bool:
    return ch.islower() or ch.isupper() or ch.istitle()


def _strip_trailing_emoji(text: str) -> str:
    parts = _GRAPHEME.findall(text.rstrip())
    while parts and (parts[-1].isspace() or parts[-1] in find_emoji(parts[-1])):
        parts.pop()
    return "".join(parts)


def _sentences(paragraph: str) -> list[str]:
    out: list[str] = []
    for block in _LIST_BREAK.split(paragraph):
        start = 0
        for m in _SENTENCE_END.finditer(block):
            rest = block[m.end():].lstrip()
            if rest and rest[0].islower() and not regex.search(r"[!?؟。！？]", m.group()):
                continue  # "e.g. this", "wait... no": not a sentence end
            out.append(block[start:m.end()])
            start = m.end()
        out.append(block[start:])
    return [s.strip() for s in out if tokenize(s)]


def _paragraphs(text: str) -> list[list[str]]:
    body = _HEADING.sub("", text.replace("\r\n", "\n"))
    paras = [_sentences(p) for p in _PARAGRAPH.split(body)]
    return [p for p in paras if p]


# ---------------------------------------------------------------------- metrics

#: Metric -> (comparison kind, smoothing for ratios, needs 20+ words in a draft to be judged).
_SPEC: dict[str, tuple[Literal["rate", "ratio"], float, bool]] = {
    "words_per_message_median": ("ratio", 1.0, False),
    "words_per_message_p90": ("ratio", 1.0, False),
    "messages_per_burst_mean": ("ratio", 0.25, False),
    "lowercase_start_rate": ("rate", 0.0, False),
    "final_period_rate": ("rate", 0.0, False),
    "ellipsis_rate": ("rate", 0.0, False),
    "exclamation_rate": ("rate", 0.0, False),
    "question_rate": ("rate", 0.0, False),
    "repeated_punct_rate": ("rate", 0.0, False),
    "emoji_per_message": ("ratio", 0.2, False),
    "elongation_rate": ("rate", 0.0, False),
    "caps_word_rate": ("ratio", 0.05, True),
    "commas_per_100_words": ("ratio", 2.0, True),
    "words_per_sentence_mean": ("ratio", 1.0, False),
    "words_per_sentence_sd": ("ratio", 2.0, False),
    "sentences_per_paragraph_mean": ("ratio", 0.5, False),
    "first_person_rate": ("ratio", 0.02, True),
    "semicolons_per_1000_words": ("ratio", 2.0, True),
    "dashes_per_1000_words": ("ratio", 2.0, True),
    "parentheses_per_1000_words": ("ratio", 2.0, True),
}
_MIN_DRAFT_WORDS = 20  # below this, per-word densities of a draft are noise
_MIN_SUPPORT = 20  # describe() states nothing backed by fewer texts


class _Measure(BaseModel):
    metrics: dict[str, float] = {}
    support: dict[str, int] = {}
    n_words: int = 0


def _chat_metrics(messages: Sequence[str], burst_sizes: Sequence[int] | None) -> _Measure:
    n = len(messages)
    out = _Measure()
    if burst_sizes:
        out.metrics["messages_per_burst_mean"] = sum(burst_sizes) / len(burst_sizes)
        out.support["messages_per_burst_mean"] = len(burst_sizes)
    if not n:
        return out
    counts: list[int] = []
    flags: Counter[str] = Counter()
    cased_starts = lower_starts = emoji = commas = cased_words = caps_words = 0
    msgs_with_words = msgs_with_cased = 0
    for m in messages:
        toks = tokenize(m)
        counts.append(len(toks))
        first = _LETTER.search(m)
        if first and _cased(first.group()):
            cased_starts += 1
            lower_starts += first.group().islower()
        flags["final_period_rate"] += bool(_FINAL_STOP.search(_strip_trailing_emoji(m)))
        flags["ellipsis_rate"] += bool(_ELLIPSIS.search(m))
        flags["exclamation_rate"] += bool(_EXCLAMATION.search(m))
        flags["question_rate"] += bool(_QUESTION.search(m))
        flags["repeated_punct_rate"] += bool(_REPEATED.search(m))
        flags["elongation_rate"] += any(_is_elongated(t) for t in toks)
        emoji += len(find_emoji(m))
        commas += len(_COMMA.findall(m))
        cased = [t for t in toks
                 if len(_LETTER.findall(t)) >= 2 and t.upper() != t.lower()]
        cased_words += len(cased)
        caps_words += sum(t == t.upper() for t in cased)
        msgs_with_words += bool(toks)
        msgs_with_cased += bool(cased)
    ranked = sorted(counts)
    words = sum(counts)
    out.n_words = words
    out.metrics["words_per_message_median"] = float(statistics.median(ranked))
    out.metrics["words_per_message_p90"] = float(ranked[math.ceil(0.9 * n) - 1])
    out.support["words_per_message_median"] = out.support["words_per_message_p90"] = n
    if cased_starts:
        out.metrics["lowercase_start_rate"] = lower_starts / cased_starts
        out.support["lowercase_start_rate"] = cased_starts
    for key in ("final_period_rate", "ellipsis_rate", "exclamation_rate", "question_rate",
                "repeated_punct_rate"):
        out.metrics[key] = flags[key] / n
        out.support[key] = n
    out.metrics["emoji_per_message"] = emoji / n
    out.support["emoji_per_message"] = n
    out.metrics["elongation_rate"] = flags["elongation_rate"] / n
    out.support["elongation_rate"] = n
    if cased_words:
        out.metrics["caps_word_rate"] = caps_words / cased_words
        out.support["caps_word_rate"] = msgs_with_cased
    if words:
        out.metrics["commas_per_100_words"] = commas / words * 100
        out.support["commas_per_100_words"] = msgs_with_words
    return out


def _longform_metrics(texts: Sequence[str]) -> _Measure:
    out = _Measure()
    paragraphs = [p for t in texts for p in _paragraphs(t)]
    sentences = [s for p in paragraphs for s in p]
    if not sentences:
        return out
    lengths = [len(tokenize(s)) for s in sentences]
    words = sum(lengths)
    joined = "\n\n".join(texts)
    tokens = [t for s in sentences for t in tokenize(s)]
    n_s = len(sentences)
    out.n_words = words
    out.metrics["words_per_sentence_mean"] = words / n_s
    out.metrics["words_per_sentence_sd"] = statistics.pstdev(lengths)
    out.metrics["sentences_per_paragraph_mean"] = n_s / len(paragraphs)
    out.metrics["first_person_rate"] = sum(map(_is_first_person, tokens)) / words
    out.metrics["question_rate"] = sum(bool(_QUESTION_END.search(s)) for s in sentences) / n_s
    out.metrics["commas_per_100_words"] = len(_COMMA.findall(joined)) / words * 100
    for key, pattern in (("semicolons_per_1000_words", _SEMICOLON),
                         ("dashes_per_1000_words", _DASH),
                         ("parentheses_per_1000_words", _PAREN)):
        out.metrics[key] = len(pattern.findall(joined)) / words * 1000
    out.support = {key: n_s for key in out.metrics}
    out.support["sentences_per_paragraph_mean"] = len(paragraphs)
    return out


# ---------------------------------------------------------------------- profile


class StyleProfile(BaseModel):
    """Measured style of one register (or of one recipient within it)."""

    register_name: str
    form: Literal["chat", "longform"]
    n_texts: int
    n_words: int
    metrics: dict[str, float]
    #: How many messages, sentences or paragraphs each metric rests on.
    support: dict[str, int] = Field(default_factory=dict)
    top_emoji: list[tuple[str, float]] = []  # (emoji, share of all emoji)
    laughter: list[tuple[str, float]] = []  # (normalized form, per 100 texts)
    openers: list[tuple[str, float]] = []  # (first word, share of messages)
    closers: list[tuple[str, float]] = []  # (last word, share of messages)
    distinctive_phrases: list[tuple[str, float]] = []  # (phrase, uses per 1000 words)
    lang_mix: dict[str, float] = {}  # language -> share of texts


class Deviation(BaseModel):
    """One way a draft departs from a profile, with an instruction to fix it."""

    metric: str
    expected: float
    actual: float
    severity: Literal["minor", "major"]
    message: str


def _top(counter: Counter[str], total: float, k: int, min_count: int = 1,
         min_share: float = 0.0) -> list[tuple[str, float]]:
    if not total:
        return []
    ranked = sorted(counter.items(), key=lambda kv: -kv[1])  # stable: first seen wins ties
    return [(key, n / total) for key, n in ranked
            if n >= min_count and n / total >= min_share][:k]


def _ngrams(texts: Sequence[str], orders: Sequence[int]) -> tuple[Counter[str], Counter[int]]:
    grams: Counter[str] = Counter()
    totals: Counter[int] = Counter()
    for text in texts:
        toks = [t.casefold() for t in tokenize(text)]
        for n in orders:
            for i in range(len(toks) - n + 1):
                grams[" ".join(toks[i:i + n])] += 1
                totals[n] += 1
    return grams, totals


def _drop_subsumed(cands: dict[str, float], counts: Counter[str]) -> list[str]:
    keep = []
    for c in cands:
        if not any(len(d) > len(c) and f" {c} " in f" {d} " and counts[d] >= 0.75 * counts[c]
                   for d in cands):
            keep.append(c)
    return sorted(keep, key=lambda g: -cands[g])


def _distinctive(texts: Sequence[str], background: Sequence[str], n_words: int,
                 *, min_count: int = 3, k: int = 20) -> list[tuple[str, float]]:
    if not n_words:
        return []
    mine, my_totals = _ngrams(texts, (1, 2, 3))
    scores: dict[str, float] = {}
    if background:
        theirs, their_totals = _ngrams(background, (1, 2, 3))
        a0 = 0.5
        for gram, a in mine.items():
            if a < min_count:
                continue
            n = gram.count(" ") + 1
            b, big_a, big_b = theirs[gram], my_totals[n], their_totals[n]
            delta = (math.log((a + a0) / (big_a - a + a0))
                     - math.log((b + a0) / (big_b - b + a0)))
            if delta >= math.log(2):  # at least twice as frequent, relatively
                scores[gram] = delta / math.sqrt(1 / (a + a0) + 1 / (b + a0))
    else:
        common = {g for g, _ in Counter({g: c for g, c in mine.items() if " " not in g})
                  .most_common(50)}
        for gram, a in mine.items():
            if a >= min_count and " " in gram and not set(gram.split(" ")) <= common:
                scores[gram] = float(a)
    return [(g, mine[g] / n_words * 1000) for g in _drop_subsumed(scores, mine)[:k]]


def _measure(form: str, texts: Sequence[str], burst_sizes: Sequence[int] | None) -> _Measure:
    if form == "chat":
        return _chat_metrics(texts, burst_sizes)
    return _longform_metrics(texts)


def build_profile(register_name: str, texts: Sequence[str], *,
                  burst_sizes: Sequence[int] | None = None,
                  background: Sequence[str] | None = None,
                  langs: Sequence[str | None] | None = None) -> StyleProfile:
    """Measure the owner's texts of one register; `background` is what others wrote to them."""
    form: Literal["chat", "longform"] = "chat" if is_chat_register(register_name) else "longform"
    lang_list = list(langs) if langs is not None else [None] * len(texts)
    kept = [(t, lang) for t, lang in zip(texts, lang_list, strict=False) if t and t.strip()]
    texts = [t for t, _ in kept]
    measure = _measure(form, texts, burst_sizes)
    n = len(texts)

    emoji = Counter(e for t in texts for e in find_emoji(t))
    laughs = Counter(x for t in texts for x in find_laughter(t))
    openers: Counter[str] = Counter()
    closers: Counter[str] = Counter()
    if form == "chat":
        for t in texts:
            toks = tokenize(t)
            if toks:
                openers[toks[0].casefold()] += 1
                closers[toks[-1].casefold()] += 1
    lang_counts = Counter(lang for _, lang in kept if lang)
    bg = [t for t in (background or []) if t and t.strip()]
    return StyleProfile(
        register_name=register_name,
        form=form,
        n_texts=n,
        n_words=measure.n_words if form == "longform" else sum(len(tokenize(t)) for t in texts),
        metrics=measure.metrics,
        support=measure.support,
        top_emoji=_top(emoji, emoji.total(), 10),
        laughter=[(k, laughs[k] * 100 / n) for k, _ in _top(laughs, n, 10)],
        openers=_top(openers, n, 5, min_count=3, min_share=0.05),
        closers=_top(closers, n, 5, min_count=3, min_share=0.05),
        distinctive_phrases=_distinctive(texts, bg, sum(len(tokenize(t)) for t in texts)),
        lang_mix={k: v for k, v in _top(lang_counts, lang_counts.total(), 20)},
    )


def profile_items(register_name: str, items: Sequence[Item],
                  background_items: Sequence[Item] = ()) -> StyleProfile:
    """Profile from items: owner's non-holdout texts, bursts from timing, others as background."""
    own = [i for i in items if i.is_self and not i.holdout]
    burst_sizes = None
    if is_chat_register(register_name):
        sizes = [len(b) for b in bursts([*items, *background_items])
                 if not any(i.holdout for i in b)]
        burst_sizes = sizes or None
    return build_profile(
        register_name,
        [i.text for i in own],
        burst_sizes=burst_sizes,
        background=[i.text for i in background_items if not i.is_self],
        langs=[i.lang for i in own],
    )


# ------------------------------------------------------------------- comparison


def _num(x: float) -> str:
    r = round(x, 1)
    return str(int(r)) if r == int(r) else f"{r:.1f}"


def _pct(x: float) -> str:
    v = x * 100
    return "under 1%" if 0 < v < 0.5 else f"{round(v)}%"


_RATE_NAMES = {
    "ellipsis_rate": "ellipses (...)",
    "exclamation_rate": "exclamation marks",
    "question_rate": "question marks",
    "repeated_punct_rate": "doubled punctuation (!!, ?!)",
    "elongation_rate": "stretched words (sooo, noooon)",
}
_PER_1000_NAMES = {
    "semicolons_per_1000_words": "semicolons",
    "dashes_per_1000_words": "dashes",
    "parentheses_per_1000_words": "parentheses",
}


def _message(metric: str, e: float, a: float, major: bool, profile: StyleProfile) -> str:
    more = a > e
    too = "Much too" if major else "Too"
    E, A = _pct(e), _pct(a)  # noqa: N806
    unit = "messages" if profile.form == "chat" else "sentences"
    if metric == "words_per_message_median":
        if more:
            return (f"{too} long: your messages are usually about {_num(e)} words, these have "
                    f"{_num(a)}. Cut each message to about {_num(e)} words, or split it into "
                    "several short ones.")
        return (f"{too} short: your messages are usually about {_num(e)} words, these have "
                f"{_num(a)}. Say a little more in each message.")
    if metric == "words_per_message_p90":
        if more:
            return (f"Your longest messages rarely go over {_num(e)} words; the longest here has "
                    f"{_num(a)}. Shorten or split it.")
        return (f"Let a message run longer now and then: yours sometimes reach {_num(e)} words, "
                f"the longest here has {_num(a)}.")
    if metric == "messages_per_burst_mean":
        if more:
            return (f"Send fewer separate messages: you usually send about {_num(e)} in a row, "
                    f"this is {_num(a)}. Merge some of them.")
        return (f"Split this into about {max(2, round(e))} separate messages: you usually send "
                f"about {_num(e)} in a row, this is {_num(a)}.")
    if metric == "lowercase_start_rate":
        if more:
            return (f"Start messages with a capital letter: you do in {_pct(1 - e)} of messages, "
                    f"this draft in {_pct(1 - a)}.")
        return f"Start messages in lowercase: you do in {E} of messages, this draft in {A}."
    if metric == "final_period_rate":
        if more and e < 0.02:
            return (f"Drop the final full stop: you almost never end a message with one; "
                    f"this draft does in {A} of messages.")
        if more:
            return (f"Drop the final full stop: you end {E} of messages with one, "
                    f"this draft {A}.")
        return f"End messages with a full stop: you do in {E} of messages, this draft in {A}."
    if metric in _RATE_NAMES:
        name = _RATE_NAMES[metric] if profile.form == "chat" else "questions"
        if more and e < 0.02:
            return f"Avoid {name}: you almost never use them; this draft does in {A} of {unit}."
        if more:
            return f"Use fewer {name}: you use them in {E} of {unit}, this draft in {A}."
        return f"Use {name} more: you use them in {E} of {unit}, this draft in {A}."
    if metric == "emoji_per_message":
        if more and e < 0.05:
            return (f"Leave out emoji: you almost never use them; this draft has {_num(a)} "
                    "per message.")
        if more:
            return f"Use fewer emoji: you average {_num(e)} per message, this draft {_num(a)}."
        usual = " ".join(x for x, _ in profile.top_emoji[:5])
        tail = f" Your usual ones: {usual}." if usual else ""
        return f"Add emoji: you average {_num(e)} per message, this draft {_num(a)}.{tail}"
    if metric == "caps_word_rate":
        if more and e < 0.01:
            return f"Avoid words in capitals: you almost never use them; {A} of words here are."
        if more:
            return f"Write fewer words in capitals: {E} of your words are, {A} here."
        return f"Put a word in capitals for emphasis now and then: {E} of your words are, {A} here."
    if metric == "commas_per_100_words":
        word = "fewer" if more else "more"
        return (f"Use {word} commas: you use about {_num(e)} per 100 words, "
                f"this draft {_num(a)}.")
    if metric == "words_per_sentence_mean":
        if more:
            return (f"{too} long sentences: yours average {_num(e)} words, these average "
                    f"{_num(a)}. Write shorter sentences.")
        return (f"{too} short sentences: yours average {_num(e)} words, these average "
                f"{_num(a)}. Join some sentences.")
    if metric == "words_per_sentence_sd":
        if more:
            return (f"Keep sentence length more even: yours vary by about {_num(e)} words, "
                    f"these by {_num(a)}.")
        return (f"Vary sentence length more, mixing short and long sentences: yours vary by "
                f"about {_num(e)} words, these by {_num(a)}.")
    if metric == "sentences_per_paragraph_mean":
        size = "shorter" if more else "longer"
        return (f"Use {size} paragraphs: yours average {_num(e)} sentences, "
                f"these {_num(a)}.")
    if metric == "first_person_rate":
        if more:
            return f"Use fewer first-person words (I, me, my): {E} of your words are, {A} here."
        return (f"Write more in the first person (I, me, my or the equivalent): {E} of your "
                f"words are, {A} here.")
    name = _PER_1000_NAMES[metric]
    if more and e < 0.5:
        return (f"Drop the {name}: you almost never use them; this draft has {_num(a)} "
                "per 1000 words.")
    if more:
        return f"Use fewer {name}: you use about {_num(e)} per 1000 words, this draft {_num(a)}."
    return f"Use {name} more: you use about {_num(e)} per 1000 words, this draft {_num(a)}."


def _draft_measure(profile: StyleProfile, texts: Sequence[str]) -> _Measure:
    parts = [t for t in texts if t and t.strip()]
    if profile.form == "chat":
        return _chat_metrics(parts, [len(parts)] if parts else None)
    return _longform_metrics(["\n\n".join(parts)] if parts else [])


def _scored(profile: StyleProfile, texts: Sequence[str]) -> list[tuple[str, float, float, float]]:
    """(metric, expected, actual, normalized deviation) for every comparable metric."""
    draft = _draft_measure(profile, texts)
    out = []
    for metric, expected in profile.metrics.items():
        if metric not in draft.metrics or metric not in _SPEC:
            continue
        kind, eps, per_word = _SPEC[metric]
        if per_word and draft.n_words < _MIN_DRAFT_WORDS:
            continue
        actual = draft.metrics[metric]
        if kind == "rate":
            norm = abs(actual - expected) / 0.5
        else:
            norm = abs(math.log((actual + eps) / (expected + eps))) / math.log(3)
        out.append((metric, expected, actual, min(norm, 2.0)))
    return out


def compare(profile: StyleProfile, texts: Sequence[str]) -> list[Deviation]:
    """Where a draft (one message or paragraph per text) departs from the profile, worst first."""
    devs = []
    flagged: dict[str, bool] = {}
    for metric, expected, actual, norm in _scored(profile, texts):
        if _SPEC[metric][0] == "rate":
            minor_at = 0.25 / 0.5  # absolute difference over 0.25
        else:
            minor_at = math.log(2) / math.log(3)  # ratio outside [0.5, 2]
        if norm <= minor_at + 1e-9:
            continue
        major = norm > 1.0 + 1e-9 and metric != "question_rate"  # questions follow content
        if metric == "words_per_message_p90" and flagged.get("words_per_message_median") == (
                actual > expected):
            continue  # same advice as the median
        flagged[metric] = actual > expected
        devs.append((norm, Deviation(
            metric=metric, expected=expected, actual=actual,
            severity="major" if major else "minor",
            message=_message(metric, expected, actual, major, profile),
        )))
    devs.sort(key=lambda nd: (nd[1].severity != "major", -nd[0]))
    return [d for _, d in devs]


def distance(profile: StyleProfile, texts: Sequence[str]) -> float:
    """Mean normalized deviation over the measured metrics: 0 = same, 1 = a major deviation."""
    scored = _scored(profile, texts)
    return sum(s[3] for s in scored) / len(scored) if scored else 0.0


# --------------------------------------------------------------------- describe


def _frequency(name: str, rate: float, unit: str) -> str:
    if rate < 0.02:
        return f"Almost never uses {name}."
    return f"Uses {name} in {_pct(rate)} of {unit}."


def describe(profile: StyleProfile) -> list[str]:
    """Plain statements for a style card; nothing resting on fewer than 20 texts."""
    m = profile.metrics

    def ok(key: str) -> bool:
        return key in m and profile.support.get(key, 0) >= _MIN_SUPPORT

    lines: list[str] = []
    if profile.form == "chat":
        if ok("words_per_message_median") and ok("words_per_message_p90"):
            med = m["words_per_message_median"]
            adj = "short" if med <= 8 else "of medium length" if med <= 20 else "long"
            lines.append(f"Messages are {adj}: usually {_num(med)} words, "
                         f"rarely over {_num(m['words_per_message_p90'])}.")
        if ok("messages_per_burst_mean"):
            b = m["messages_per_burst_mean"]
            lines.append("Usually says it in a single message rather than several in a row."
                         if b < 1.3 else
                         f"Often sends several messages in a row: about {_num(b)} at a time.")
        if ok("lowercase_start_rate"):
            r = m["lowercase_start_rate"]
            lines.append(f"Starts messages in lowercase ({_pct(r)})." if r >= 0.7 else
                         f"Starts messages with a capital letter ({_pct(1 - r)})." if r <= 0.3
                         else f"Starts {_pct(r)} of messages in lowercase.")
        if ok("final_period_rate"):
            r = m["final_period_rate"]
            lines.append("Never ends a message with a full stop." if r == 0 else
                         f"Rarely ends a message with a full stop ({_pct(r)})." if r <= 0.15
                         else f"Usually ends messages with a full stop ({_pct(r)})." if r >= 0.6
                         else f"Ends {_pct(r)} of messages with a full stop.")
        for key, name in _RATE_NAMES.items():
            if ok(key):
                lines.append(_frequency(name, m[key], "messages"))
        if ok("emoji_per_message"):
            e = m["emoji_per_message"]
            lines.append("Almost never uses emoji." if e < 0.05 else
                         f"Uses about {_num(e)} emoji per message.")
        if ok("caps_word_rate"):
            r = m["caps_word_rate"]
            lines.append("Almost never writes words in capitals." if r < 0.01 else
                         f"Writes {_pct(r)} of words in capitals.")
    else:
        if ok("words_per_sentence_mean"):
            mean = m["words_per_sentence_mean"]
            line = f"Sentences average {_num(mean)} words"
            if ok("words_per_sentence_sd"):
                sd = m["words_per_sentence_sd"]
                line += f" (most between {_num(max(1.0, mean - sd))} and {_num(mean + sd)})"
            lines.append(line + ".")
        if ok("sentences_per_paragraph_mean"):
            lines.append(f"Paragraphs average {_num(m['sentences_per_paragraph_mean'])} "
                         "sentences.")
        if ok("first_person_rate"):
            r = m["first_person_rate"]
            lines.append(f"Writes a lot in the first person ({_pct(r)} of words)." if r >= 0.04
                         else "Rarely writes in the first person." if r < 0.01
                         else f"First-person words make up {_pct(r)} of words.")
        if ok("question_rate"):
            r = m["question_rate"]
            lines.append("Rarely asks questions." if r < 0.02 else
                         f"Asks questions in {_pct(r)} of sentences.")
        for key, name in _PER_1000_NAMES.items():
            if ok(key):
                v = m[key]
                lines.append(f"Almost never uses {name}." if v < 0.5 else
                             f"Uses {name} about {_num(v)} times per 1000 words.")
    if ok("commas_per_100_words"):
        c = m["commas_per_100_words"]
        lines.append("Almost never uses commas." if c < 0.5 else
                     f"Uses about {_num(c)} commas per 100 words.")
    if profile.n_texts >= _MIN_SUPPORT:
        for label, pairs in (("opens", profile.openers), ("closes", profile.closers)):
            if pairs:
                listed = ", ".join(f'"{w}" ({_pct(s)})' for w, s in pairs)
                lines.append(f"Often {label} with {listed}.")
        mix = [(k, v) for k, v in profile.lang_mix.items() if v >= 0.05]
        if len(mix) > 1:
            lines.append("Mixes languages: " + ", ".join(f"{_pct(v)} {k}" for k, v in mix) + ".")
    return lines
