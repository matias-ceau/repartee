"""Style cards: measured profiles per register (and per close recipient) plus a model's description.

Profiles go to ``derived/voice/<register>.json`` and ``derived/voice/<register>/<slug>.json``;
cards go to ``self/voice/<register>.md`` through the self-model, which keeps the owner's
``## Notes``. Held-out items never reach a profile, a card or the description prompt.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from repartee.home import Home, HomeNotFound
from repartee.llm.base import LLMClient, LLMError, LLMRequest, Message, SystemBlock
from repartee.schema import Item, Kind, slugify
from repartee.selfmodel.model import SelfModel
from repartee.store.corpus import Corpus
from repartee.voice.prompts import STYLE_CARD_SYSTEM, StyleCardOut, style_card_prompt
from repartee.voice.registers import bursts, is_chat_register, register_of
from repartee.voice.samples import recipient_names
from repartee.voice.stylometry import StyleProfile, describe, profile_items

log = logging.getLogger(__name__)

MIN_TEXTS = 30  # registers with fewer of the owner's texts get no profile and no card
MIN_RECIPIENT_MESSAGES = 200  # owner messages to one person before they get their own profile
DESCRIPTION_SAMPLES = 40  # texts (chat: bursts) shown to the model for the description
SAMPLE_CHARS = 700  # longer texts are cut for the description prompt
_MIN_LAUGHS = 0.5  # per 100 texts, below this a laughter form is not worth a line
_MIN_EMOJI = 0.05  # per chat message, below this "almost never uses emoji" says it all


def owner_name(home: Home) -> str:
    """The owner's name from repartee.yaml, or a neutral stand-in."""
    try:
        return home.load_config().owner.name
    except (HomeNotFound, OSError, ValueError):
        return "the owner"


def recipient_of(container_items: Sequence[Item], container: str) -> str:
    """The other person in a one-to-one chat, else the chat's name (as in reply pairs)."""
    others = {i.author for i in container_items if not i.is_self}
    return next(iter(others)) if len(others) == 1 else container


def _num(x: float) -> str:
    r = round(x, 1)
    return str(int(r)) if r == int(r) else f"{r:.1f}"


def render_card(profile: StyleProfile, description: str | None) -> str:
    """The generated block of a style card: measured habits, signature words, a description."""
    unit = "messages" if profile.form == "chat" else "texts"
    out = [f"_Measured over {profile.n_texts} {unit} ({profile.n_words} words), held-out "
           "texts excluded. Rewritten by `repartee voice`: put your own rules under Notes._"]
    if description and description.strip():
        out += ["", "### How it reads", "", description.strip()]
    habits = list(describe(profile))
    rich_emoji = profile.form != "chat" or profile.metrics.get("emoji_per_message", 0) >= _MIN_EMOJI
    if profile.top_emoji and rich_emoji:
        listed = ", ".join(f"{e} ({round(s * 100)}%)" for e, s in profile.top_emoji[:6])
        habits.append(f"Favourite emoji: {listed}.")
    laughs = [(form, rate) for form, rate in profile.laughter if rate >= _MIN_LAUGHS]
    if laughs:
        listed = ", ".join(f'"{form}" ({_num(rate)} per 100 {unit})' for form, rate in laughs[:4])
        habits.append(f"Laughs with {listed}.")
    if profile.distinctive_phrases:
        listed = ", ".join(f'"{p}"' for p, _ in profile.distinctive_phrases[:12])
        habits.append(f"Typical words and phrases: {listed}.")
    if habits:
        out += ["", "### Measured habits", "", *(f"- {line}" for line in habits)]
    return "\n".join(out)


def _spread(values: Sequence, k: int) -> list:
    """Up to k values evenly spaced over the sequence, first and last included."""
    n = len(values)
    if n <= k:
        return list(values)
    picks = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [values[i] for i in picks]


def _clip(text: str, limit: int = SAMPLE_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > limit // 2 else limit] + " …"


class VoiceBuilder:
    """Builds style profiles and cards from the corpus; reads stored profiles back."""

    def __init__(self, home: Home, corpus: Corpus, model: SelfModel, llm: LLMClient | None):
        self.home = home
        self.corpus = corpus
        self.model = model
        self.llm = llm

    @property
    def voice_dir(self) -> Path:
        return self.home.derived_dir / "voice"

    # ----------------------------------------------------------------- reading
    @staticmethod
    def _read(path: Path) -> StyleProfile | None:
        if not path.is_file():
            return None
        try:
            return StyleProfile.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError) as exc:
            log.warning("ignoring unreadable voice profile %s: %s", path, exc)
            return None

    def profile(self, register_name: str) -> StyleProfile | None:
        """The stored profile of a register, None when it was never built (or too thin)."""
        return self._read(self.voice_dir / f"{register_name}.json")

    def recipient_profile(self, register_name: str, recipient: str) -> StyleProfile | None:
        """The stored profile of the owner's messages to one person in a register, if any."""
        for name in recipient_names(recipient):
            found = self._read(self.voice_dir / register_name / f"{slugify(name)}.json")
            if found is not None:
                return found
        return None

    def profiles(self) -> dict[str, StyleProfile]:
        """Every stored register profile, by register name."""
        out = {}
        if self.voice_dir.is_dir():
            for path in sorted(self.voice_dir.glob("*.json")):
                found = self._read(path)
                if found is not None:
                    out[path.stem] = found
        return out

    # ---------------------------------------------------------------- building
    def _write(self, path: Path, profile: StyleProfile) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(profile.model_dump_json(indent=1) + "\n", encoding="utf-8")

    async def build(self, registers: Sequence[str] | None = None) -> list[str]:
        """Rebuild profiles and cards (all registers, or the named ones); returns those written."""
        owned: dict[str, list[Item]] = {}
        others: dict[str, list[Item]] = {}
        for item in self.corpus.iter_items():
            if item.is_self:
                name = register_of(item)
                if name is not None:
                    owned.setdefault(name, []).append(item)  # holdout included: splits bursts
            elif item.kind == Kind.MESSAGE:
                others.setdefault(item.container, []).append(item)

        wanted = set(registers) if registers is not None else None
        written: list[str] = []
        for name in sorted(owned):
            if wanted is not None and name not in wanted:
                continue
            items = owned[name]
            usable = [i for i in items if not i.holdout and i.text.strip()]
            if len(usable) < MIN_TEXTS:
                log.info("voice: %s has %d texts, fewer than %d; no card", name, len(usable),
                         MIN_TEXTS)
                continue
            chat = is_chat_register(name)
            containers = sorted({i.container for i in items})
            background = [o for c in containers for o in others.get(c, [])] if chat else []
            profile = profile_items(name, items, background)
            self._write(self.voice_dir / f"{name}.json", profile)
            if chat:
                self._build_recipients(name, items, others)
            description = await self._describe(name, chat, profile, usable, background)
            self.model.set_voice_card(name, render_card(profile, description))
            written.append(name)
        if written:
            self.model.save()
        return written

    def _build_recipients(self, name: str, items: list[Item],
                          others: dict[str, list[Item]]) -> None:
        folder = self.voice_dir / name
        if folder.is_dir():
            shutil.rmtree(folder)  # a stale per-person profile must not outlive its data
        by_recipient: dict[str, list[str]] = {}
        for container in sorted({i.container for i in items}):
            mine = [i for i in items if i.container == container]
            who = recipient_of([*mine, *others.get(container, [])], container)
            by_recipient.setdefault(who, []).append(container)
        for who, containers in sorted(by_recipient.items()):
            mine = [i for i in items if i.container in containers]
            if sum(1 for i in mine if not i.holdout) < MIN_RECIPIENT_MESSAGES:
                continue
            background = [o for c in containers for o in others.get(c, [])]
            self._write(folder / f"{slugify(who)}.json", profile_items(name, mine, background))

    async def _describe(self, name: str, chat: bool, profile: StyleProfile,
                        usable: list[Item], background: list[Item]) -> str | None:
        if self.llm is None:
            return None
        if chat:
            groups = bursts([*usable, *background])
            texts = ["\n".join(m.text for m in b) for b in _spread(groups, DESCRIPTION_SAMPLES)]
        else:
            texts = [_clip(i.text) for i in _spread(usable, DESCRIPTION_SAMPLES)]
        owner = owner_name(self.home)
        request = LLMRequest(
            tier="smart",
            purpose="style_card",
            system=[SystemBlock(text=STYLE_CARD_SYSTEM)],
            messages=[Message(role="user", text=style_card_prompt(
                owner, name, chat, describe(profile), texts))],
            output_schema=StyleCardOut,
        )
        try:
            response = await self.llm.complete(request)
        except LLMError as exc:
            log.warning("voice: no description for %s (%s); the card has statistics only",
                        name, exc)
            return None
        parsed = response.parsed
        return parsed.description if isinstance(parsed, StyleCardOut) else None

