"""What the evals share: units of the owner's writing, neutral briefs, drafting, saving results.

A *unit* is what one side of a comparison shows: a single chat message, a whole burst of
chat messages, or one longform text. Units are built from held-out items only (or from
non-held-out items only); a burst that mixes the two is never shown whole.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import regex
from pydantic import BaseModel, Field

from repartee.home import Home
from repartee.llm.base import LLMClient, LLMError, LLMRequest, Message, SystemBlock
from repartee.runtime import Runtime
from repartee.schema import DraftRequest, Item, Kind, SearchFilters
from repartee.store.corpus import Corpus
from repartee.voice.cards import recipient_of
from repartee.voice.drafter import Drafter
from repartee.voice.registers import bursts, is_chat_register, register_of, voice_item
from repartee.voice.stylometry import tokenize

#: Items (or brief-and-draft jobs) run at once; keeps a real run polite to the API.
CONCURRENCY = 4
MIN_WORDS = 4  # texts of three words or fewer say too little to judge
CONTEXT_MESSAGES = 6  # messages before a chat unit shown as the conversation it answers
CONTEXT_GAP = timedelta(hours=3)  # older messages belong to another conversation
QUOTE_WORDS = 5  # a brief sharing this many consecutive words with the text quotes it

_URL = regex.compile(r"https?://\S+|www\.\S+")
_HEADING_LINE = regex.compile(r"^[ \t]*#{1,6}[ \t].*$\n?", regex.MULTILINE)
_WIKILINK = regex.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_PARAGRAPH = regex.compile(r"\n[ \t]*\n")


# ------------------------------------------------------------------------ units


def plain(text: str) -> str:
    """Longform text without vault markup: no heading lines, wikilinks as their label."""
    text = _WIKILINK.sub(lambda m: m.group(2) or m.group(1), text)
    text = _HEADING_LINE.sub("", text)
    return regex.sub(r"\n{3,}", "\n\n", text).strip()


def eligible(items: Sequence[Item]) -> bool:
    """True when the texts are worth judging: over three words, not mostly a link, no media."""
    if any(item.meta.get("media") for item in items):
        return False  # a caption: the photo it captions is half the message
    text = "\n".join(item.text for item in items)
    without_links = _URL.sub(" ", text)
    if len(tokenize(without_links)) < MIN_WORDS:
        return False
    linked = sum(len(m) for m in _URL.findall(text))
    return linked * 2 <= len("".join(text.split()))


@dataclass
class Unit:
    """One text (or burst) of the owner's, with what a drafter needs to write it again."""

    register_name: str
    items: list[Item]
    recipient: str | None = None
    context: str | None = None
    chat: bool = True
    group: bool = False  # a group chat: the recipient is the chat's name

    @property
    def key(self) -> str:
        return self.items[0].id

    @property
    def item_ids(self) -> list[str]:
        return [item.id for item in self.items]

    def texts(self) -> list[str]:
        """The owner's real text as shown: one entry per message, or one longform text."""
        if self.chat:
            return [item.text.strip() for item in self.items]
        return [plain(item.text) for item in self.items]

    def shape_note(self) -> str:
        """The form the draft must take so that both sides of a pair look alike."""
        if self.chat:
            if len(self.items) == 1:
                return "(Send this as one single message.)"
            return f"(Send this as {len(self.items)} separate messages, one after the other.)"
        paragraphs = len([p for p in _PARAGRAPH.split(self.texts()[0]) if p.strip()])
        return ("(Write this as one paragraph.)" if paragraphs <= 1
                else f"(Write this as {paragraphs} paragraphs.)")

    def shape(self, parts: Sequence[str]) -> list[str]:
        """A draft in the same shape as the real side."""
        parts = [p.strip() for p in parts if p and p.strip()]
        if not self.chat:
            return ["\n\n".join(parts)] if parts else []
        if len(self.items) == 1 and len(parts) > 1:
            return ["\n".join(parts)]
        return parts


def _context(before: Sequence[Item], first: Item) -> str | None:
    """The conversation just before a chat unit, stopping at a held-out text of the owner's."""
    lines: list[str] = []
    newest = first
    for item in reversed(before):
        if item.is_self and item.holdout:
            break
        if (item.timestamp and newest.timestamp
                and newest.timestamp - item.timestamp > CONTEXT_GAP):
            break
        lines.append(f"{item.author}: {item.text.strip()}")
        newest = item
        if len(lines) >= CONTEXT_MESSAGES:
            break
    return "\n".join(reversed(lines)) or None


def units(corpus: Corpus, *, holdout: bool, register_name: str | None = None) -> list[Unit]:
    """Every eligible unit of the owner's writing that is entirely held out (or entirely not)."""
    candidates: dict[str, list[Item]] = {}
    longform: list[Unit] = []
    for item in corpus.iter_items(SearchFilters(author="self"), holdout=holdout):
        name = register_of(item)
        if name is None or (register_name is not None and name != register_name):
            continue
        if item.holdout != holdout or not item.text.strip():
            continue
        if is_chat_register(name) and item.kind == Kind.MESSAGE:
            candidates.setdefault(item.container, []).append(item)
        elif eligible([voice_item(item)]):
            longform.append(Unit(name, [voice_item(item)], chat=False))

    out: list[Unit] = []
    for container, mine in candidates.items():
        thread = sorted(corpus.iter_items(SearchFilters(containers=[container],
                                                        kinds=[Kind.MESSAGE])),
                        key=lambda i: (i.seq, i.timestamp or datetime.min, i.id))
        position = {item.id: n for n, item in enumerate(thread)}
        burst_of = {m.id: burst for burst in bursts(thread) for m in burst}
        recipient = recipient_of(thread, container)
        group = len({i.author for i in thread if not i.is_self}) > 1
        seen: set[str] = set()
        for item in mine:
            if item.id in seen:
                continue
            name = register_of(item)
            burst = burst_of.get(item.id, [item])
            whole = all(m.holdout == holdout and register_of(m) == name for m in burst)
            members = burst if whole else [item]
            seen.update(m.id for m in members)
            if not eligible(members):
                continue
            start = position.get(members[0].id, 0)
            out.append(Unit(name, list(members), recipient=recipient, group=group,
                            context=_context(thread[:start], members[0])))
    return out + longform


def holdout_units(corpus: Corpus, register_name: str | None = None) -> list[Unit]:
    """Units made only of held-out items: the real side of the blind test."""
    return units(corpus, holdout=True, register_name=register_name)


def sample(pool: Sequence[Unit], n: int, seed: int) -> list[Unit]:
    """n units drawn reproducibly from the seed."""
    ordered = sorted(pool, key=lambda u: u.key)
    random.Random(seed).shuffle(ordered)
    return ordered[: max(0, n)]


# ----------------------------------------------------------------------- briefs


class BriefOut(BaseModel):
    """What a `brief` request returns."""

    brief: str = Field(description="The brief only: what the text conveys, in plain words.")


class BriefRejected(Exception):
    """The brief is empty or carries the original's wording, so the pair would not be fair."""


BRIEF_SYSTEM = """\
You write briefs for a blind writing test. A ghostwriter who imitates one person will get \
only your brief and must write the text again from it; then the person tries to tell their \
own real text from the ghostwriter's. The test is fair only if the brief carries what the \
text says and nothing of how it is said.

Write the brief as a short note to the ghostwriter, in plain, neutral English, addressed to \
the person in the second person ("Tell Sam the train gets in at six and ask whether he can \
pick you up.").

- Cover everything the text conveys, in its order: news, facts, plans, questions, requests, \
answers, opinions, and the attitude behind it when that is part of the message (pleased, \
annoyed, teasing, reassuring). Keep names, places, dates, times and numbers: they are content.
- Do not quote the text, not even a few words. Do not reuse its distinctive words, \
expressions, slang, spelling, abbreviations, nicknames, greetings, emoji or laughter: say the \
same thing in your own plain words.
- Say nothing about its form: not its language, its length, how many messages, sentences or \
paragraphs it has, its punctuation, capitalisation or style.
- Add nothing the text does not convey, and do not comment on it.

Return only the brief."""


_LONGFORM_NAMES = {"essay": "an essay passage", "journal": "a journal entry", "note": "a note",
                   "interview": "an answer in an interview", "draft": "a text"}


def brief_prompt(owner: str, unit: Unit) -> str:
    """The user message of a `brief` request: the situation, then the real text."""
    if unit.chat:
        if unit.group:
            to = f' in the group chat "{unit.items[0].container}"'
        else:
            to = f" in a chat with {unit.recipient}" if unit.recipient else " in a chat"
        what = (f"{len(unit.items)} consecutive messages" if len(unit.items) > 1
                else "a message")
    else:
        channel = unit.register_name.rsplit(".", 1)[0]
        to, what = "", _LONGFORM_NAMES.get(channel, "a text")
    parts = [f"{owner} wrote {what}{to}."]
    if unit.context:
        parts.append("The messages just before it, for the situation only (brief what "
                     f"{owner} wrote, not what others said):\n<conversation>\n{unit.context}"
                     "\n</conversation>")
    parts.append(f"<text>\n{chr(10).join(unit.texts())}\n</text>")
    parts.append("Write the brief.")
    return "\n\n".join(parts)


def _words(text: str) -> list[str]:
    return [t.casefold() for t in tokenize(text)]


def quotes(brief: str, texts: Sequence[str], k: int = QUOTE_WORDS) -> bool:
    """True when the brief repeats a run of k words of the text (the whole text if shorter)."""
    said = _words(brief)
    for text in texts:
        words = _words(text)
        size = min(k, len(words))
        if size == 0:
            continue
        runs = {tuple(words[i:i + size]) for i in range(len(words) - size + 1)}
        if any(tuple(said[i:i + size]) in runs for i in range(len(said) - size + 1)):
            return True
    return False


async def make_brief(llm: LLMClient, owner: str, unit: Unit) -> str:
    """A neutral brief of what the unit says; BriefRejected if it would leak the wording."""
    response = await llm.complete(LLMRequest(
        tier="smart", purpose="brief", system=[SystemBlock(text=BRIEF_SYSTEM, cache=True)],
        messages=[Message(role="user", text=brief_prompt(owner, unit))],
        output_schema=BriefOut, max_tokens=2000))
    parsed = response.parsed
    brief = parsed.brief.strip() if isinstance(parsed, BriefOut) else ""
    if not brief:
        raise BriefRejected("the brief came back empty")
    if quotes(brief, unit.texts()):
        raise BriefRejected("the brief quoted the original text")
    return brief


async def draft_unit(drafter: Drafter, unit: Unit, brief: str) -> list[str]:
    """The mirror's version of the unit, from the brief alone, in the unit's shape."""
    result = await drafter.draft(DraftRequest(
        intent=f"{brief}\n\n{unit.shape_note()}", recipient=unit.recipient,
        register_name=unit.register_name, context=unit.context,
        exclude_ids=list(unit.item_ids)))
    shaped = unit.shape(result.variants[0] if result.variants else [])
    if not shaped:
        raise LLMError("the draft came back empty")
    return shaped


def reason(error: BaseException) -> str:
    """One line saying why an item failed."""
    if isinstance(error, BriefRejected):
        return f"brief rejected: {error}"
    return f"{type(error).__name__}: {error}"


async def bounded[T](jobs: Sequence[Callable[[], Awaitable[T]]],
                  limit: int = CONCURRENCY) -> list[T | Exception]:
    """Run the jobs, at most `limit` at once; results in order, failures returned in place."""
    gate = asyncio.Semaphore(max(1, limit))

    async def one(job: Callable[[], Awaitable[T]]) -> T | Exception:
        async with gate:
            try:
                return await job()
            except Exception as error:  # one failed item is an entry, never a crash
                return error

    return list(await asyncio.gather(*(one(job) for job in jobs)))


# ----------------------------------------------------------------------- saving


@dataclass
class Saver:
    """Writes one eval's results to <home>/evals/<prefix>-<timestamp>.json, never overwriting."""

    home: Home
    prefix: str
    created: datetime = field(default_factory=lambda: datetime.now().replace(microsecond=0))

    def save(self, payload: dict[str, Any]) -> Path:
        folder = self.home.evals_dir
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.prefix}-{self.created:%Y%m%d-%H%M%S}"
        path, n = folder / f"{stem}.json", 1
        while path.exists():
            n += 1
            path = folder / f"{stem}-{n}.json"
        body = {"created": self.created.isoformat(), **payload}
        path.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=str) + "\n",
                        encoding="utf-8")
        return path


def settings(rt: Runtime, **extra: Any) -> dict[str, Any]:
    """The settings a result depends on, saved next to it so runs can be compared."""
    return {
        **extra,
        "models": rt.config.models.model_dump(),
        "effort": rt.config.effort.model_dump(),
        "holdout_percent": rt.config.holdout_percent,
        "concurrency": CONCURRENCY,
    }
