"""The drafter: write a text in the owner's voice, measure it, revise once if it is clearly off.

The model gets what a ghostwriter would want (recipient and relationship, the style card,
real samples, past replies in similar situations, the thread, the intent) and phrases what
the owner wants to say. It does not add content of its own.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import yaml

from repartee.home import Home
from repartee.llm.base import (
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    SystemBlock,
)
from repartee.schema import (
    DraftRequest,
    DraftResult,
    PageKind,
    ReplyPair,
    SearchFilters,
    stable_hash,
)
from repartee.selfmodel.model import SelfModel
from repartee.store.corpus import Corpus
from repartee.voice.cards import VoiceBuilder, owner_name
from repartee.voice.prompts import (
    DraftOut,
    burst_hint,
    draft_prompt,
    draft_system,
    render_pair,
    render_sample,
    revise_prompt,
)
from repartee.voice.registers import group_by_register, is_chat_register, register_of
from repartee.voice.samples import pick_samples, recipient_containers, recipient_names
from repartee.voice.stylometry import Deviation, StyleProfile, compare, describe, tokenize

log = logging.getLogger(__name__)

SAMPLES = 8  # real texts shown with a draft
REPLY_PAIRS = 4  # similar past situations shown with a draft
CONTEXT_FOR_TOPIC = 600  # characters of the thread's end used to find topical samples
SAMPLE_CHARS = 1500  # longer longform samples are cut at a paragraph boundary

_CLAIM_ID = re.compile(r"[ \t]+\^c-[0-9a-f]{8}[ \t]*$", re.MULTILINE)
_MARKER = re.compile(r"^[ \t]*<!-- repartee:generated:(?:start|end) -->[ \t]*\n?", re.MULTILINE)
_BLANKS = re.compile(r"\n[ \t]*(?:\n[ \t]*)+")
_TOP_HEADING = re.compile(r"^(#{1,2}[ \t]+\S)", re.MULTILINE)
_OWNER_FACING = re.compile(r"^_Measured over .*_$")  # the card's note to the owner


def _clean_page(text: str) -> str:
    """A self-model page for a prompt: no claim ids, no generated markers, tight spacing."""
    text = _MARKER.sub("", _CLAIM_ID.sub("", text))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _differs(line: str, usual: Sequence[str]) -> bool:
    """True unless a usual line says the same thing with numbers within 25% of these.

    Opener and closer lists are never worth repeating: the samples to that person show them.
    """
    if line.startswith(("Often opens with", "Often closes with")):
        return False
    skeleton = _NUMBER.sub("#", line)
    mine = [float(x) for x in _NUMBER.findall(line)]
    for other in usual:
        if _NUMBER.sub("#", other) != skeleton:
            continue
        theirs = [float(x) for x in _NUMBER.findall(other)]
        if all(abs(a - b) <= max(1.0, 0.25 * max(a, b)) for a, b in zip(mine, theirs,
                                                                         strict=True)):
            return False
    return True


def card_text(model: SelfModel, register_name: str, owner: str) -> str | None:
    """A style card for a prompt: the generated content, then the owner's own notes."""
    page = model.page(f"voice/{register_name}")
    if page is None:
        return None
    generated = "\n".join(line for line in page.generated().split("\n")
                          if not _OWNER_FACING.match(line)).strip()
    notes = page.section_text("Notes").strip()
    if not generated and not notes:
        text = _clean_page(page.body())
        return text or None
    if notes:
        generated += (f"\n\n### Notes {owner} wrote about this register (these outrank "
                      f"everything above)\n\n{notes}")
    return generated.strip()


def _clip_longform(text: str, limit: int = SAMPLE_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text.rfind("\n\n", 0, limit)
    if cut < limit // 3:
        cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > 0 else limit].rstrip() + "\n[…]"


def _clean_variants(raw: Sequence[Sequence[str]]) -> list[list[str]]:
    out = []
    for variant in raw:
        parts = [p.strip() for p in variant if p and p.strip()]
        if parts:
            out.append(parts)
    return out


def _block(text: str) -> str:
    """One message or paragraph as one blank-line-free block that cannot open a section."""
    return _TOP_HEADING.sub(r"\\\1", _BLANKS.sub("\n", text.strip()))


class Drafter:
    """Drafts texts in the owner's voice and records what the owner kept."""

    def __init__(self, home: Home, corpus: Corpus, model: SelfModel, llm: LLMClient):
        self.home = home
        self.corpus = corpus
        self.model = model
        self.llm = llm
        self.voice = VoiceBuilder(home, corpus, model, None)

    # -------------------------------------------------------------- recipient
    def _recipient(self, recipient: str | None) -> tuple[str | None, list[str]]:
        """(person page key or None, names to look them up by in the corpus)."""
        if not recipient or not recipient.strip():
            return None, []
        recipient = recipient.strip()
        key = None
        if "/" in recipient and self.model.page(recipient) is not None:
            key = recipient
        else:
            key = self.model.resolve(PageKind.PERSON, recipient)
        names = [recipient]
        page = self.model.page(key) if key else None
        if page is not None:
            names = [page.title, *page.aliases, recipient]
        return key, recipient_names(names)

    # --------------------------------------------------------------- register
    def _register_sizes(self) -> dict[str, int]:
        """Words of owner text per register: from stored profiles, else counted."""
        stored = self.voice.profiles()
        if stored:
            return {name: p.n_words for name, p in stored.items()}
        groups = group_by_register(self.corpus.iter_items(SearchFilters(author="self"),
                                                          holdout=False))
        return {name: sum(len(tokenize(i.text)) for i in items) for name, items in groups.items()}

    def infer_register(self, request: DraftRequest) -> str:
        """Explicit register; else the recipient's usual chat; else the main chat; else largest."""
        if request.register_name:
            return request.register_name
        _, names = self._recipient(request.recipient)
        if names:
            containers = recipient_containers(self.corpus, names)
            if containers:
                counts: Counter[str] = Counter()
                filters = SearchFilters(author="self", containers=sorted(containers))
                for item in self.corpus.iter_items(filters, holdout=False):
                    name = register_of(item)
                    if name and is_chat_register(name):
                        counts[name] += 1
                if counts:
                    return min(counts, key=lambda n: (-counts[n], n))
        sizes = self._register_sizes()
        pool = {n: s for n, s in sizes.items() if is_chat_register(n)} or sizes
        if not pool:
            raise LookupError("none of your own writing is indexed yet: run `repartee ingest` "
                              "(and `repartee voice`) before drafting")
        return min(pool, key=lambda n: (-pool[n], n))

    # ------------------------------------------------------------------ draft
    def _pairs(self, register_name: str, context: str, names: list[str]) -> list[ReplyPair]:
        lang = register_name.rsplit(".", 1)[-1] if "." in register_name else None
        found: list[ReplyPair] = []
        for who in [*names, None]:
            for pair in self.corpus.search_reply_pairs(context, who, k=REPLY_PAIRS):
                if pair.id in {p.id for p in found}:
                    continue
                if lang and pair.lang and pair.lang != lang:
                    continue
                found.append(pair)
            if len(found) >= REPLY_PAIRS:
                break
        return found[:REPLY_PAIRS]

    async def draft(self, request: DraftRequest) -> DraftResult:
        """Draft, compare with the owner's measured style, revise once on a major deviation."""
        register = self.infer_register(request)
        chat = is_chat_register(register)
        owner = owner_name(self.home)
        page_key, names = self._recipient(request.recipient)
        page = self.model.page(page_key) if page_key else None
        who = page.title if page is not None else (request.recipient or "").strip() or None

        card = card_text(self.model, register, owner)
        mine = next((p for p in (self.voice.recipient_profile(register, n) for n in names)
                     if p is not None), None)
        general = self.voice.profile(register)
        profile: StyleProfile | None = mine or general
        usual = describe(general) if general is not None else []
        with_them = [line for line in describe(mine) if _differs(line, usual)] if mine else []

        topic = request.intent
        if request.context:
            topic += "\n" + request.context[-CONTEXT_FOR_TOPIC:]
        samples = pick_samples(self.corpus, register, recipient=names or None, topic=topic,
                               k=SAMPLES)
        pairs = self._pairs(register, request.context, names) if (
            chat and request.context and request.context.strip()) else []

        hint = None
        if chat and profile is not None:
            hint = burst_hint(owner, profile.metrics.get("words_per_message_median"),
                              profile.metrics.get("messages_per_burst_mean"))
        n = max(1, request.variants)
        prompt = draft_prompt(
            owner=owner, register_name=register, chat=chat, intent=request.intent,
            recipient=who, person_page=_clean_page(page.body()) if page is not None else None,
            card=card, recipient_habits=with_them,
            samples=[render_sample(s if chat else s.model_copy(
                update={"text": _clip_longform(s.text)}), chat) for s in samples],
            pairs=[render_pair(p) for p in pairs],
            context=request.context, burst_hint=hint, variants=n,
        )
        system = [SystemBlock(text=draft_system(owner), cache=True)]
        first = Message(role="user", text=prompt)
        response = await self.llm.complete(LLMRequest(
            tier="chat", purpose="draft", system=system, messages=[first],
            output_schema=DraftOut))
        variants = self._variants(response)[:n]
        if not variants:
            raise LLMError("the model returned an empty draft")

        checks = [compare(profile, v) if profile is not None else [] for v in variants]
        flagged = [i for i, devs in enumerate(checks) if any(d.severity == "major" for d in devs)]
        if flagged:
            await self._revise(owner, system, first, response, variants, checks, flagged,
                               profile)

        used = [k for k in (page_key, f"voice/{register}" if card else None) if k]
        used += [s.id for s in samples] + [p.id for p in pairs]
        return DraftResult(register_name=register, variants=variants, used=used,
                           deviations=[d.message for d in checks[0]])

    @staticmethod
    def _variants(response: LLMResponse) -> list[list[str]]:
        parsed = response.parsed
        if not isinstance(parsed, DraftOut):
            raise LLMError("the draft reply did not match the expected structure")
        return _clean_variants(parsed.variants)

    async def _revise(self, owner: str, system: list[SystemBlock], first: Message,
                      response: LLMResponse, variants: list[list[str]],
                      checks: list[list[Deviation]], flagged: list[int],
                      profile: StyleProfile | None) -> None:
        """Revise the flagged variants once, in place; on failure the first drafts stay."""
        follow_up = revise_prompt(owner, [(i + 1, [d.message for d in checks[i]])
                                          for i in flagged])
        assistant = response.message
        if not assistant.text and not assistant.raw:
            assistant = Message(role="assistant", text=response.text)
        try:
            revised = self._variants(await self.llm.complete(LLMRequest(
                tier="chat", purpose="revise", system=system,
                messages=[first, assistant, Message(role="user", text=follow_up)],
                output_schema=DraftOut)))
        except LLMError as exc:
            log.warning("draft revision failed, keeping the first draft: %s", exc)
            return
        for index, new in zip(flagged, revised, strict=False):
            variants[index] = new
            checks[index] = compare(profile, new) if profile is not None else []

    # ----------------------------------------------------------------- record
    def record(self, request: DraftRequest, result: DraftResult, *, variant: int = 0,
               final: Sequence[str] | None = None) -> Path:
        """Write drafts/<stamp>-<hash>.md: the mirror's variant and what the owner kept.

        `final=None` means the variant was accepted unchanged; an empty `final` means
        nothing was kept (the file says `accepted: false` and yields no items).
        """
        mirror = list(result.variants[variant]) if result.variants else []
        if final is None:
            kept = mirror
        else:
            kept = re.split(r"\n[ \t]*\n", final) if isinstance(final, str) else list(final)
        mirror_blocks = [b for b in (_block(t) for t in mirror) if b]
        final_blocks = [b for b in (_block(t) for t in kept) if b]
        now = datetime.now().replace(microsecond=0)
        front = {
            "kind": "draft",
            "date": now.isoformat(),
            "register_name": result.register_name,
            "recipient": request.recipient,
            "intent": request.intent,
            "accepted": bool(final_blocks),
            "edited": final is not None and [t.strip() for t in kept] != mirror,
        }
        body = ["---", yaml.safe_dump(front, sort_keys=False, allow_unicode=True,
                                      width=10_000).rstrip("\n"), "---", ""]
        if request.context and request.context.strip():
            body += ["## Context", "", _TOP_HEADING.sub(r"\\\1", request.context.strip()), ""]
        body += ["## Mirror", "", "\n\n".join(mirror_blocks), "", "## Final", ""]
        if final_blocks:
            body += ["\n\n".join(final_blocks), ""]
        text = "\n".join(body)

        self.home.drafts_dir.mkdir(parents=True, exist_ok=True)
        salt = 0
        while True:
            digest = stable_hash(now.isoformat(), request.intent, *final_blocks, salt, length=8)
            path = self.home.drafts_dir / f"{now:%Y%m%d-%H%M%S}-{digest}.md"
            if not path.exists():
                break
            salt += 1
        path.write_text(text, encoding="utf-8")
        return path
