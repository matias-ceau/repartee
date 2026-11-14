"""Light offline language detection: scripts first, then stopword profiles."""

from __future__ import annotations

from collections import Counter

import regex

#: Languages a script alone identifies well enough.
SCRIPT_LANGS: list[tuple[str, str]] = [
    ("Hangul", "ko"),
    ("Hiragana", "ja"),
    ("Katakana", "ja"),
    ("Greek", "el"),
    ("Hebrew", "he"),
    ("Thai", "th"),
    ("Georgian", "ka"),
    ("Armenian", "hy"),
]

#: Frequent function words per language. Words shared by several languages count
#: for each of them with a smaller weight.
STOPWORDS: dict[str, str] = {
    "en": "the a an and or but of to in on at for with is are was were be been it it's i "
    "i'm you he she we they this that these those not no do does did don't have has had "
    "my your our their what when where why how if so just all there here about would "
    "will can could should think from by as me him her them us very really yes",
    "fr": "le la les un une des et ou mais de du au aux en dans sur pour avec est sont "
    "était être a ont avait je tu il elle on nous vous ils elles ce cette ces pas ne "
    "plus que qui quoi où quand comment si oui non c' j' l' d' qu' n' s' m' t' mon ma "
    "mes ton ta tes son sa ses moi toi lui leur très bien aussi alors donc peut fait "
    "c'est suis es sommes êtes ça là chez",
    "es": "el la los las un una unos unas y o pero de del al en con por para es son era "
    "fue ser está están estar yo tú él ella nosotros vosotros ellos ellas este esta "
    "estos estas no sí que qué quien cuando como cómo si mi mis tu tus su sus muy "
    "también pues porque hay hola tal creo deberíamos este ese eso ya",
    "de": "der die das ein eine einen dem den des und oder aber von zu im in an auf für "
    "mit ist sind war waren sein bin bist ich du er sie es wir ihr nicht kein keine "
    "was wer wann wo wie warum wenn dass ja nein mein dein sehr auch noch nur schon "
    "hat haben wird werden sollten glaube am",
    "it": "il lo la i gli le un uno una e o ma di del della dei a al alla in nel nella con "
    "per su è sono era essere io tu lui lei noi voi loro questo questa questi non sì "
    "che chi cosa quando come dove se mio mia tuo tua suo sua molto anche ancora "
    "penso dovremmo ci ho hai ha abbiamo",
    "pt": "o a os as um uma uns umas e ou mas de do da dos das em no na nos nas com por "
    "para é são era ser está estão eu tu ele ela nós vocês eles elas este esta isso "
    "isto não sim que quem quando como onde se meu minha teu tua seu sua muito também "
    "acho devíamos você ao à neste nesta",
    "nl": "de het een en of maar van naar in op aan voor met is zijn was waren ben ik jij "
    "je hij zij ze wij we jullie dit dat deze niet geen wat wie wanneer waar hoe "
    "waarom als ja nee mijn jouw zijn haar heel ook nog al denk moeten heb hebben "
    "er dan",
}

#: Letters that point to one language.
LETTER_HINTS: dict[str, tuple[str, float]] = {
    "ñ": ("es", 1.0),
    "¿": ("es", 1.0),
    "¡": ("es", 1.0),
    "ã": ("pt", 1.0),
    "õ": ("pt", 1.0),
    "ß": ("de", 1.0),
}

_WEIGHTS: dict[str, dict[str, float]] = {}
_owners: dict[str, list[str]] = {}
for _lang, _words in STOPWORDS.items():
    for _w in set(_words.split()):
        _owners.setdefault(_w, []).append(_lang)
for _w, _langs in _owners.items():
    _WEIGHTS[_w] = {lang: 1.0 / len(_langs) for lang in _langs}

_WORD = regex.compile(r"\p{L}+(?:['’]\p{L}+)*")
_LETTER = regex.compile(r"\p{L}")
_SCRIPT = {
    name: regex.compile(rf"\p{{Script={name}}}") for name, _ in SCRIPT_LANGS + [("Han", "zh")]
}
_LATIN = regex.compile(r"\p{Script=Latin}")

#: Minimum weighted score and lead over the runner-up for an answer.
MIN_SCORE = 1.5
MIN_LEAD = 1.5


def detect_language(text: str) -> str | None:
    """ISO 639-1 code of the text's language, or None when unsure."""
    sample = text[:4000]
    letters = len(_LETTER.findall(sample))
    if letters < 2:
        return None
    for name, code in SCRIPT_LANGS:
        if len(_SCRIPT[name].findall(sample)) >= max(2, letters * 0.15):
            return code
    if len(_SCRIPT["Han"].findall(sample)) >= letters * 0.5:
        return "zh"
    if len(_LATIN.findall(sample)) < letters * 0.5:
        return None
    scores: Counter[str] = Counter()
    for word in _WORD.findall(sample.lower().replace("’", "'")):
        tokens = [word]
        if "'" in word:
            head, _, tail = word.partition("'")
            tokens += [head + "'", tail]
        for token in tokens:
            for lang, weight in _WEIGHTS.get(token, {}).items():
                scores[lang] += weight
    for char, (lang, weight) in LETTER_HINTS.items():
        if char in sample:
            scores[lang] += weight
    if not scores:
        return None
    ranked = scores.most_common(2)
    best_lang, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if best < MIN_SCORE or best < MIN_LEAD * second:
        return None
    return best_lang
