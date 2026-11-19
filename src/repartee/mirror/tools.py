"""The mirror's tools over the corpus and the self-model (spec section 13).

Every tool returns a ToolOutcome: compact text the model reasons from (dated, attributed,
with ids it can follow up), the refs and one-line summary the owner sees beside the
answer, and an error flag. Tools never raise; a bad call is an outcome the model can read
and correct.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import regex
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from repartee.llm.base import ToolSpec
from repartee.mirror.context import claim_status, clip, fmt_span, fmt_when, indent
from repartee.schema import (
    ID_PREFIX,
    PAGE_DIRS,
    Claim,
    ClaimStatus,
    DraftRequest,
    DraftResult,
    Item,
    Kind,
    PageKind,
    SearchFilters,
)
from repartee.selfmodel import SelfModel, SelfModelError
from repartee.store import Corpus

log = logging.getLogger(__name__)

MAX_CHARS = 2000  # most tools
MAX_READ_CHARS = 3000  # read_page and read_memory return whole texts
SEARCH_HITS = 6
CLAIM_HITS = 8

_CLAIM_KINDS = ("person", "topic", "taste", "behaviour", "timeline", "health")
_SOURCES = tuple(ID_PREFIX)
_KINDS = tuple(k.value for k in Kind)
_WORD = regex.compile(r"[\p{L}\p{N}]{2,}")
_CLAIM_ID = regex.compile(r"[ \t]+\^(c-[0-9a-f]{8})[ \t]*$")
_CHAT_LINE = regex.compile(r"\s+(?=(?:…\s*)?\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] )")
_HEADING = regex.compile(r"^#{1,6}\s")
_MARKER = regex.compile(r"^\s*<!--\s*repartee:generated:(start|end)\s*-->\s*$")


class DrafterLike(Protocol):
    """What the mirror needs from voice.drafter.Drafter."""

    async def draft(self, request: DraftRequest) -> DraftResult: ...

    def record(self, request: DraftRequest, result: DraftResult, *, variant: int = 0,
               final: Sequence[str] | None = None) -> Path: ...


class VoiceLookup(Protocol):
    """What get_voice needs: style cards and real samples per register."""

    def registers(self) -> list[str]: ...

    def card(self, register_name: str) -> str | None: ...

    def samples(self, register_name: str, recipient: str | None, topic: str | None,
                k: int) -> list[Item]: ...


class ToolOutcome(BaseModel):
    """Result of one tool call."""

    content: str  # what the model reads
    refs: list[str] = []  # page keys, chunk ids, item ids shown to the owner
    summary: str = ""  # one line for the "looked up" panel
    is_error: bool = False


def error(name: str, message: str) -> ToolOutcome:
    """A failed call the model can read and correct."""
    return ToolOutcome(content=message, summary=f"{name}: {clip(message, 70)}", is_error=True)


# ------------------------------------------------------------------ arguments and specs


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _SearchSelf(_Args):
    query: str
    kind: Literal["person", "topic", "taste", "behaviour", "timeline", "health"] | None = None


class _ReadPage(_Args):
    path: str


class _SearchMemory(_Args):
    query: str
    person: str | None = None
    source: str | None = None
    after: str | None = None
    before: str | None = None
    author: Literal["self", "other"] | None = None


class _ReadMemory(_Args):
    chunk_id: str
    around: int | None = None


class _GetVoice(_Args):
    register_name: str | None = Field(default=None, alias="register")
    recipient: str | None = None
    topic: str | None = None


class _SimilarReplies(_Args):
    context: str
    recipient: str | None = None


class _NoteGap(_Args):
    question: str


def _schema(**properties: dict[str, Any]) -> dict[str, Any]:
    # Strict tool schemas: every property required, optional ones nullable.
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def _text(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _nullable(description: str, **schema: Any) -> dict[str, Any]:
    return {"anyOf": [{"type": "string", **schema}, {"type": "null"}],
            "description": description}


SPECS: dict[str, ToolSpec] = {spec.name: spec for spec in [
    ToolSpec(
        name="search_self",
        description=(
            "Search the self-model: distilled claims about the owner, one line each, with "
            "status (confirmed by the owner, or inferred with a confidence), validity dates and "
            "the ids of the material behind them. Also lists pages whose name matches. Start "
            "here to orient; check specifics in the raw material."),
        input_schema=_schema(
            query=_text("What to look for: a subject, a person's name, a few words."),
            kind=_nullable("Only this kind of page; null for all.", enum=list(_CLAIM_KINDS)),
        )),
    ToolSpec(
        name="read_page",
        description=(
            "Read one self-model page in full: every claim with its status, plus the owner's "
            "own notes. Pages are keyed like people/ana-ruiz or topics/moving-abroad; a "
            "person's or subject's name also works."),
        input_schema=_schema(path=_text("Page key, e.g. people/ana-ruiz, or a name.")),
    ),
    ToolSpec(
        name="search_memory",
        description=(
            "Search the raw record: chat messages, notes, journal entries, essays, interviews. "
            "Returns dated, attributed excerpts with chunk ids for read_memory. Keyword and "
            "meaning search combined; try other words or another language if it misses."),
        input_schema=_schema(
            query=_text("Words or a description of what to find."),
            person=_nullable("Only material involving this person (a name)."),
            source=_nullable(
                "Only this source (whatsapp, vault, writing, tastes, assistant, interviews, "
                "conversations, drafts) or this kind (message, note, journal, essay, "
                "interview, medical, taste, mirror, draft, assistant)."),
            after=_nullable("Only material from this date on: YYYY, YYYY-MM or YYYY-MM-DD."),
            before=_nullable("Only material before this date: YYYY, YYYY-MM or YYYY-MM-DD."),
            author=_nullable("'self' for what the owner wrote, 'other' for what others wrote.",
                             enum=["self", "other"]),
        )),
    ToolSpec(
        name="read_memory",
        description=(
            "Read one excerpt of the raw record in full, by chunk id (from search_memory) or "
            "by item id (from a claim's evidence), optionally with the excerpts just before "
            "and after it in the same chat or note."),
        input_schema=_schema(
            chunk_id=_text("A chunk id (ch-…) or an item id (wa-…, nt-…, …)."),
            around={"anyOf": [{"type": "integer"}, {"type": "null"}],
                    "description": "How many neighbouring excerpts on each side, 0 to 2."},
        )),
    ToolSpec(
        name="get_voice",
        description=(
            "How the owner writes: the style card of a register (a kind of text in a language, "
            "e.g. whatsapp.fr, essay.en) and real samples, preferring a recipient and topic."),
        input_schema=_schema(
            register=_nullable("Register name; null for the owner's main one."),
            recipient=_nullable("A person the text is for, to prefer samples sent to them."),
            topic=_nullable("A subject, to prefer samples about it."),
        )),
    ToolSpec(
        name="similar_replies",
        description=(
            "What the owner actually answered in chats when someone wrote something similar. "
            "Good for how they react, and for phrasing."),
        input_schema=_schema(
            context=_text("The incoming message or situation."),
            recipient=_nullable("Only replies to this person."),
        )),
    ToolSpec(
        name="note_gap",
        description=(
            "Record a question the record could not answer, so a later interview asks the "
            "owner. Use it when you had to say you don't know something they would expect "
            "you to know."),
        input_schema=_schema(question=_text("The question to ask the owner, in plain words.")),
    ),
]}

_ARGS: dict[str, type[_Args]] = {
    "search_self": _SearchSelf, "read_page": _ReadPage, "search_memory": _SearchMemory,
    "read_memory": _ReadMemory, "get_voice": _GetVoice, "similar_replies": _SimilarReplies,
    "note_gap": _NoteGap,
}

_MODE_TOOLS: dict[str, tuple[str, ...]] = {
    "chat": ("search_self", "read_page", "search_memory", "read_memory", "get_voice",
             "similar_replies", "note_gap"),
    "interview": ("search_self", "read_page", "search_memory", "read_memory", "note_gap"),
}


# ------------------------------------------------------------------------------ toolbox


class Toolbox:
    """The tools for one mirror: lookups over `corpus` and `model`, voice when given."""

    def __init__(self, corpus: Corpus, model: SelfModel, voice: VoiceLookup | None = None, *,
                 owner_name: str = "the owner"):
        self.corpus = corpus
        self.model = model
        self.voice = voice
        self.owner = owner_name

    def specs(self, mode: str) -> list[ToolSpec]:
        """Tool definitions offered in a mode; stable order, so the cached prefix holds."""
        names = _MODE_TOOLS.get(mode, ())
        return [SPECS[n] for n in names if n != "get_voice" or self.voice is not None]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        """Run a tool; failures come back as outcomes, never as exceptions."""
        if name not in _ARGS:
            return error(name, f"There is no tool {name!r}. Available: {', '.join(SPECS)}.")
        try:
            args = _ARGS[name].model_validate(arguments or {})
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
                for e in exc.errors())
            return error(name, f"Bad arguments for {name}: {problems}.")
        try:
            return getattr(self, f"_{name}")(args)
        except _Refusal as exc:
            return error(name, str(exc))
        except Exception as exc:  # a tool must never take the conversation down
            log.exception("mirror tool %s failed", name)
            return error(name, f"{name} failed: {type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- self-model

    def _search_self(self, args: _SearchSelf) -> ToolOutcome:
        if not args.query:
            raise _Refusal("search_self needs a query.")
        kind = PageKind(args.kind) if args.kind else None
        claims = self._find_claims(args.query, kind)
        pages = self._pages_named(args.query, kind)
        where = f' in {PAGE_DIRS[kind]}' if kind else ""
        if not claims and not pages:
            return ToolOutcome(
                content=(f'No claims in the self-model match "{args.query}"{where}. It may not '
                         "cover this yet; search_memory looks through the raw material."),
                summary=f'self-model "{clip(args.query, 40)}" → nothing')

        dates = self._evidence_dates(claims)
        blocks = [f'Self-model, "{args.query}"{where}: {len(claims)} claims, best match first.']
        for claim in claims:
            blocks.append(self._claim_block(claim, dates))
        page_keys = list(dict.fromkeys([c.page for c in claims] + pages))
        listed = []
        for key in page_keys:
            page = self.model.page(key)
            if page is not None:
                count = len(self.model.claims(page=key))
                listed.append(f"{key} ({page.title}, {count} claims)")
        if listed:
            blocks.append("Pages: " + "; ".join(listed) + ". read_page shows a whole page; "
                          "read_memory(<evidence id>) shows the material behind a claim.")
        return ToolOutcome(content=_bound(blocks, MAX_CHARS), refs=page_keys,
                           summary=f'self-model "{clip(args.query, 40)}" → {len(claims)} claims')

    def _find_claims(self, query: str, kind: PageKind | None) -> list[Claim]:
        found: list[Claim] = []
        for indexed, _score in self.corpus.search_claims(query, kind, k=CLAIM_HITS):
            current = self.model.claim(indexed.id)  # the index can lag behind the model
            if current is not None and current.status != ClaimStatus.REJECTED:
                found.append(current)
        if found:
            return found
        # The claims index may not be synced yet: fall back to word overlap over the model.
        words = {w.casefold() for w in _WORD.findall(query)}
        scored = []
        for claim in self.model.claims():
            if kind is not None and self._kind_of(claim.page) != kind:
                continue
            text = f"{claim.text} {claim.page}".casefold()
            score = sum(1 for w in words if w in text)
            if score:
                scored.append((-score, -claim.confidence, claim.id, claim))
        return [entry[-1] for entry in sorted(scored)[:CLAIM_HITS]]

    def _kind_of(self, key: str) -> PageKind | None:
        page = self.model.page(key)
        return page.kind if page else None

    def _pages_named(self, query: str, kind: PageKind | None) -> list[str]:
        wanted = query.casefold().strip()
        words = {w.casefold() for w in _WORD.findall(query)}
        keys = []
        for page in self.model.pages(kind):
            if page.kind not in PAGE_DIRS or page.kind == PageKind.VOICE:
                continue
            for name in [page.title, *page.aliases]:
                name = name.casefold().strip()
                if name and (name == wanted or name in words
                             or (len(name) >= 4 and name in wanted)):
                    keys.append(page.key)
                    break
        return keys

    def _evidence_dates(self, claims: Sequence[Claim]) -> dict[str, datetime | None]:
        ids = [e for c in claims for e in c.evidence[:5]]
        return {i.id: i.timestamp for i in self.corpus.get_items(ids)}

    def _claim_block(self, claim: Claim, dates: dict[str, datetime | None]) -> str:
        page = self.model.page(claim.page)
        title = page.title if page else claim.page
        head = f"[{claim.id}] {claim.page} ({title}) › {claim.section} · {claim_status(claim)}"
        lines = [head, f"  {claim.text}"]
        if claim.evidence:
            shown = ", ".join(claim.evidence[:3])
            more = f" (+{len(claim.evidence) - 3})" if len(claim.evidence) > 3 else ""
            known = sorted(d for e in claim.evidence[:5] if (d := dates.get(e)) is not None)
            span = f", written {fmt_span(known[0], known[-1])}" if known else ""
            lines.append(f"  evidence: {shown}{more}{span}")
        return "\n".join(lines)

    def _read_page(self, args: _ReadPage) -> ToolOutcome:
        key = self._page_key(args.path)
        if key is None:
            raise _Refusal(f'No page "{args.path}". Pages are keyed like people/ana-ruiz or '
                           "topics/moving-abroad; search_self finds them.")
        page = self.model.page(key)
        assert page is not None
        lines, in_front = [], False
        for n, line in enumerate(page.render().replace("\r\n", "\n").split("\n")):
            if n == 0 and line.strip() == "---":
                in_front = True
                continue
            if in_front:
                in_front = line.strip() != "---"
                continue
            if _MARKER.match(line):
                continue
            lines.append(_CLAIM_ID.sub(self._annotate, line))
        aliases = f" (also: {', '.join(page.aliases)})" if page.aliases else ""
        header = (f"Page {key}{aliases}. Claim lines end with [id · status]; other text is "
                  "the owner's own. Empty sections are left out.")
        facts = self.model.facts(page=key)
        if facts:
            header += "\nKey facts: " + "; ".join(
                f"{f.slot}: {f.shown()} ({f.status})" for f in facts)
        body = "\n".join(_drop_empty_sections(lines)).strip()
        return ToolOutcome(content=_bound([header, body], MAX_READ_CHARS), refs=[key],
                           summary=f"page {key}")

    def _annotate(self, match: regex.Match) -> str:
        claim = self.model.claim(match.group(1))
        if claim is None:
            return ""
        return f"  [{claim.id} · {claim_status(claim)}]"

    def _page_key(self, path: str) -> str | None:
        key = path.strip().strip("/").replace("\\", "/")
        key = regex.sub(r"^self/", "", key)
        key = regex.sub(r"\.md$", "", key)
        if self.model.page(key) is not None:
            return key
        label = key.rsplit("/", 1)[-1]
        found = {k for kind in PAGE_DIRS if kind != PageKind.VOICE
                 if (k := self.model.resolve(kind, label)) is not None}
        return found.pop() if len(found) == 1 else None

    # ---------------------------------------------------------------- raw record

    def _search_memory(self, args: _SearchMemory) -> ToolOutcome:
        if not args.query:
            raise _Refusal("search_memory needs a query.")
        filters, described = self._filters(args)
        hits = self.corpus.search(args.query, filters, k=SEARCH_HITS)
        label = f'"{clip(args.query, 40)}"' + (f" ({described})" if described else "")
        if not hits:
            return ToolOutcome(
                content=(f"Nothing in the raw material matches {label}. Try other words, a "
                         "name, another language, or fewer filters."),
                summary=f"memory {label} → nothing")
        blocks = [f"Memory, {label}: {len(hits)} excerpts, best first."]
        for hit in hits:
            snippet = _CHAT_LINE.sub("\n", hit.snippet.strip())  # one message per line
            blocks.append(self._chunk_header(hit.chunk) + "\n" + indent(snippet))
        blocks.append("read_memory(chunk_id) shows a whole excerpt and what came around it.")
        refs = [hit.chunk.id for hit in hits]
        return ToolOutcome(content=_bound(blocks, MAX_CHARS), refs=refs,
                           summary=f"memory {label} → {len(hits)} excerpts")

    def _filters(self, args: _SearchMemory) -> tuple[SearchFilters, str]:
        filters, said = SearchFilters(), []
        if args.person:
            filters.participant = args.person
            said.append(f"with {args.person}")
        if args.source:
            source = args.source.casefold()
            if source in _SOURCES:
                filters.sources = [source]
            elif source in _KINDS:
                filters.kinds = [Kind(source)]
            else:
                raise _Refusal(f"Unknown source {args.source!r}. Sources: {', '.join(_SOURCES)};"
                               f" kinds: {', '.join(_KINDS)}.")
            said.append(f"from {source}")
        if args.after:
            filters.after = _parse_when(args.after, "after")
            said.append(f"from {args.after}")
        if args.before:
            filters.before = _parse_when(args.before, "before")
            said.append(f"before {args.before}")
        if args.author:
            filters.author = args.author
            said.append("by me" if args.author == "self" else "by others")
        return filters, ", ".join(said)

    def _chunk_header(self, chunk) -> str:
        where = f'chat "{chunk.container}"' if chunk.kind == Kind.MESSAGE else chunk.container
        title = f' "{chunk.title}"' if chunk.title and chunk.title not in chunk.container else ""
        kind = "" if chunk.kind == Kind.MESSAGE else f" · {chunk.kind}"
        people = ""
        if chunk.kind == Kind.MESSAGE and chunk.participants:
            people = " · " + ", ".join(chunk.participants[:6])
        head = (f"[{chunk.id}] {chunk.source}{kind} · {where}{title} · "
                f"{fmt_span(chunk.start, chunk.end)}{people}")
        if chunk.kind in (Kind.MIRROR, Kind.INTERVIEW):
            head += f" (a session with the mirror: only {self.owner}'s lines are memories)"
        elif chunk.kind == Kind.ASSISTANT:
            head += (f" (a conversation with an AI assistant: only {self.owner}'s lines are "
                     f"memories; the assistant's are clipped)")
        return head

    def _read_memory(self, args: _ReadMemory) -> ToolOutcome:
        wanted = args.chunk_id
        chunk = self.corpus.get_chunk(wanted) or self.corpus.chunk_for_item(wanted)
        if chunk is None:
            raise _Refusal(f"No excerpt or item {wanted!r}. Use a chunk id from search_memory "
                           "or an evidence id from a claim.")
        around = max(0, min(2, args.around or 0))
        chunks = self.corpus.neighbors(chunk.id, around, around) if around else [chunk]
        others = max(1, len(chunks) - 1)
        main_budget = MAX_READ_CHARS if len(chunks) == 1 else MAX_READ_CHARS * 3 // 5
        other_budget = (MAX_READ_CHARS - main_budget) // others
        blocks = []
        for c in chunks:
            budget = main_budget if c.id == chunk.id else other_budget
            mark = ""
            if c.id == chunk.id and wanted != chunk.id:
                mark = f" (contains item {wanted})"
            elif c.id != chunk.id:
                mark = " (neighbouring excerpt)"
            text = clip(c.text.strip(), max(80, budget - 120), flat=False)
            blocks.append(self._chunk_header(c) + mark + "\n" + indent(text))
        return ToolOutcome(content=_bound(blocks, MAX_READ_CHARS + 600),
                           refs=[c.id for c in chunks],
                           summary=f"read {chunk.source} · {clip(chunk.container, 30)} · "
                                   f"{fmt_when(chunk.start)}")

    # --------------------------------------------------------------------- voice

    def _get_voice(self, args: _GetVoice) -> ToolOutcome:
        if self.voice is None:
            raise _Refusal("Voice lookup is not available in this session.")
        registers = list(self.voice.registers())
        if not registers:
            raise _Refusal("No style cards yet: the voice statistics have not been built.")
        register = args.register_name or registers[0]
        if register not in registers:
            raise _Refusal(f"No register {register!r}. Registers: {', '.join(registers)}.")
        others = [r for r in registers if r != register]
        card = self.voice.card(register) or "(no style card yet)"
        samples = self.voice.samples(register, args.recipient, args.topic, 5)
        head = f"Voice: {register}" + (f" (other registers: {', '.join(others)})"
                                         if others else "")
        blocks = [head, "Style card:\n" + indent(clip(card.strip(), 900, flat=False))]
        if samples:
            to = f" (preferring ones to {args.recipient})" if args.recipient else ""
            lines = [f"Real samples by {self.owner}{to}:"]
            for item in samples:
                lines.append(f"[{item.id}] {fmt_when(item.timestamp)} · {item.container}\n"
                             + indent(clip(item.text, 220, flat=False)))
            blocks.append("\n".join(lines))
        bits = [f"voice {register}"] + ([f"to {args.recipient}"] if args.recipient else [])
        return ToolOutcome(content=_bound(blocks, MAX_CHARS),
                           refs=[f"voice/{register}", *(i.id for i in samples)],
                           summary=" ".join(bits) + f": card + {len(samples)} samples")

    def _similar_replies(self, args: _SimilarReplies) -> ToolOutcome:
        if not args.context:
            raise _Refusal("similar_replies needs the incoming message as context.")
        pairs = self.corpus.search_reply_pairs(args.context, args.recipient, k=4)
        to = f" to {args.recipient}" if args.recipient else ""
        if not pairs:
            return ToolOutcome(content=f"No similar situations found in the chats{to}.",
                               summary=f"similar replies{to} → none")
        blocks = [f"How {self.owner} answered in similar situations{to}:"]
        for pair in pairs:
            reply = "\n".join(pair.reply)
            blocks.append(
                f"[{pair.id}] {pair.recipient} · {fmt_when(pair.timestamp)}\n"
                f"  they wrote:\n{indent(clip(pair.context, 300, flat=False), '    ')}\n"
                f"  {self.owner} answered:\n{indent(clip(reply, 300, flat=False), '    ')}")
        return ToolOutcome(content=_bound(blocks, MAX_CHARS), refs=[p.id for p in pairs],
                           summary=f"similar replies{to} → {len(pairs)}")

    # ---------------------------------------------------------------------- gaps

    def _note_gap(self, args: _NoteGap) -> ToolOutcome:
        if not args.question:
            raise _Refusal("note_gap needs a question.")
        question = self.model.add_question(args.question)
        try:
            self.model.save()
        except SelfModelError as exc:  # kept in memory; the next save writes it
            log.warning("could not save the open question yet: %s", exc)
        return ToolOutcome(
            content=(f'Noted as an open question ({question.id}): "{question.text}". A later '
                     "interview will ask. In your answer, say you don't know rather than guess."),
            refs=[question.id], summary=f"noted a gap: {clip(question.text, 60)}")


class _Refusal(Exception):
    """A call the tool cannot serve; its message goes back to the model."""


def _parse_when(text: str, which: str) -> datetime:
    m = regex.fullmatch(r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?", text.strip())
    try:
        if m:
            return datetime(int(m[1]), int(m[2] or 1), int(m[3] or 1))
        return datetime.fromisoformat(text.strip()).replace(tzinfo=None)
    except ValueError:
        pass
    raise _Refusal(f"{which}={text!r} is not a date. Use YYYY, YYYY-MM or YYYY-MM-DD.")


def _drop_empty_sections(lines: list[str]) -> list[str]:
    """Leave out level-2+ headings with nothing under them before the next heading."""
    out: list[str] = []
    for n, line in enumerate(lines):
        if _HEADING.match(line) and not line.startswith("# "):
            rest = next((x for x in lines[n + 1:] if x.strip()), None)
            if rest is None or (_HEADING.match(rest) and rest.count("#", 0, 7)
                                <= line.count("#", 0, 7)):
                continue
        out.append(line)
    # Collapse the runs of blank lines the dropped headings leave behind.
    return [x for n, x in enumerate(out) if x.strip() or (n and out[n - 1].strip())]


def _bound(blocks: Sequence[str], limit: int) -> str:
    """Join blocks, dropping whole trailing ones past `limit` and saying how many."""
    out: list[str] = []
    size, budget = 0, limit - 70  # room for the "more not shown" line
    for n, block in enumerate(blocks):
        if out and size + len(block) + 2 > budget:
            out.append(f"[{len(blocks) - n} more not shown; narrow the request to see them.]")
            break
        out.append(block if len(block) <= budget else clip(block, budget, flat=False))
        size += len(block) + 2
    return "\n\n".join(out)
