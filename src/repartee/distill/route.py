"""Stage 2, route: decide which page each observation belongs on.

Order of rules, per observation:

1. Timeline observations go to ``timeline/<year>`` (year from ``when``, or a subject that
   is a year). Without a year they go to the person or topic page their subject names,
   else they are dropped.
2. A subject the self-model already knows (slug, title, alias, close match) goes there.
3. Unknown subjects are grouped by label; one ``route`` request may merge labels that mean
   the same subject, or map them to an existing page.
4. A group gets its own new page when it has at least ``min_observations`` observations
   (counting earlier ones on the same label), or when it is exempt: health subjects, and
   people who take part in at least one chat. Otherwise it goes to the kind's catch-all.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import regex
from pydantic import BaseModel, Field

from repartee.config import OwnerConfig
from repartee.distill.cost import Spend
from repartee.distill.prompts import ROUTE_SYSTEM, fill
from repartee.llm.base import LLMClient, LLMRequest, Message, SystemBlock
from repartee.schema import (
    CATCH_ALL_SLUG,
    CreatePage,
    Observation,
    PageKind,
    page_path,
    slugify,
)
from repartee.selfmodel import SelfModel
from repartee.selfmodel.pages import bare_name
from repartee.store.embed import Embedder, normalize_rows

log = logging.getLogger(__name__)

#: Titles of the per-kind pages for subjects with too little material of their own.
CATCH_ALL_TITLES: dict[PageKind, str] = {
    PageKind.PERSON: "People in passing",
    PageKind.TOPIC: "Other topics",
    PageKind.TASTE: "Other tastes",
    PageKind.BEHAVIOUR: "Other patterns",
    PageKind.HEALTH: "Other health notes",
}
MAX_ROUTE_LABELS = 1200  # labels asked about per run; the rest are filed as they are
ROUTE_BATCH = 150  # labels per route request
MAX_PAGES_LISTED = 300  # existing pages per kind shown to the route request

_YEAR = regex.compile(r"^(\d{4})")
_WORD = regex.compile(r"[\p{L}\p{N}]+")


def norm_label(label: str) -> str:
    """Comparison form of a subject label."""
    return " ".join(_WORD.findall(label.casefold()))


class RouteDecision(BaseModel):
    """Where one label should be filed: an existing page, or a canonical label."""

    kind: str
    label: str
    existing_page: str | None = None
    canonical: str = ""


class RouteResult(BaseModel):
    """One decision per label given."""

    decisions: list[RouteDecision] = Field(default_factory=list)


@dataclass
class Routing:
    """Where pending observations go."""

    pages: dict[str, list[Observation]] = field(default_factory=dict)
    created: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # observation ids with no possible page
    errors: list[str] = field(default_factory=list)
    merged: list[tuple[str, str]] = field(default_factory=list)  # (topic page, person page)

    def send(self, key: str, obs: Observation) -> None:
        """File one observation under a page key."""
        self.pages.setdefault(key, []).append(obs)


@dataclass
class _Group:
    kind: PageKind
    display: str
    labels: dict[str, str] = field(default_factory=dict)  # norm -> display
    observations: list[Observation] = field(default_factory=list)
    target: str | None = None


def _ensure_page(model: SelfModel, routing: Routing, kind: PageKind, slug: str, title: str,
                 aliases: Sequence[str] = ()) -> str | None:
    key = page_path(kind, slugify(slug))
    if model.page(key) is not None:
        return key
    result = model.apply([CreatePage(kind=kind, slug=slug, title=title, aliases=list(aliases))])
    if not result[0].ok:
        routing.errors.append(f"route: could not create {key} ({result[0].reason})")
        return None
    routing.created.append(key)
    return key


def chat_person(label: str, people: Iterable[str]) -> str | None:
    """The chat participant a label names; None if no match.

    A label matches a participant when one name is the start of the other: "Léo" names
    "Léo Garnier", and "Elliot Price" names a contact saved as "Elliot". When the label
    is the start of several names, the label itself is returned: it is a participant,
    but which one is unclear.
    """
    want = norm_label(label).split()
    if not want:
        return None
    named = [(person, norm_label(person).split()) for person in people]
    found = {person for person, words in named if words and words[: len(want)] == want}
    if len(found) == 1:
        return found.pop()
    if found:
        return label
    # The label is longer than the saved name: take the longest name it starts with.
    longer = [(len(words), person) for person, words in named
              if words and want[: len(words)] == words]
    if not longer:
        return None
    longest = max(n for n, _ in longer)
    best = [person for n, person in longer if n == longest]
    return best[0] if len(best) == 1 else None


def _route_request(model: SelfModel, groups: list[_Group], owner: OwnerConfig) -> LLMRequest:
    kinds = sorted({g.kind for g in groups})
    lines = ["Existing pages, by kind:"]
    for kind in kinds:
        pages = model.pages(kind)[:MAX_PAGES_LISTED]
        lines.append(f"{kind}:")
        if not pages:
            lines.append("  (none)")
        for page in pages:
            aliases = f" (also: {', '.join(page.aliases)})" if page.aliases else ""
            lines.append(f"  - {page.key}: {page.title}{aliases}")
    lines.append("")
    lines.append("New labels, with how many observations each and an example:")
    for group in groups:
        example = group.observations[0].text
        lines.append(f"- kind={group.kind} label={group.display!r} "
                     f"({len(group.observations)}): {example}")
    return LLMRequest(
        tier="smart",  # merging labels well matters more than its small cost
        purpose="route",
        system=[SystemBlock(text=fill(ROUTE_SYSTEM, owner.name), cache=True)],
        messages=[Message(role="user", text="\n".join(lines))],
        output_schema=RouteResult,
        max_tokens=16000,
    )


async def route(
    model: SelfModel,
    pending: Sequence[Observation],
    *,
    label_counts: dict[tuple[str, str], int],
    chat_people: Iterable[str],
    owner: OwnerConfig,
    min_observations: int,
    llm: LLMClient | None,
    spend: Spend | None = None,
    embedder: Embedder | None = None,
) -> Routing:
    """Assign pending observations to pages, creating pages as the rules allow.

    ``label_counts`` maps (subject kind, normalized label) to the number of current
    observations with that label, consumed or not, for the new-page threshold.
    """
    routing = Routing()
    people = [p for p in chat_people if not owner.is_self(p)]
    fold_named_topics(model, routing)
    groups: dict[tuple[PageKind, str], _Group] = {}

    for obs in pending:
        kind = PageKind(obs.subject_kind)
        if kind == PageKind.TIMELINE:
            year = _YEAR.match(obs.when or "") or regex.match(r"^(\d{4})$", obs.subject.strip())
            if year:
                key = _ensure_page(model, routing, kind, year.group(1), year.group(1))
            else:
                key = (model.resolve(PageKind.PERSON, obs.subject)
                       or model.resolve(PageKind.TOPIC, obs.subject))
            if key:
                routing.send(key, obs)
            else:
                routing.dropped.append(obs.id)
            continue
        if kind == PageKind.PERSON and owner.is_self(obs.subject):
            kind = PageKind.TOPIC  # never a page about the owner as "a person they know"
            key = _ensure_page(model, routing, kind, CATCH_ALL_SLUG, CATCH_ALL_TITLES[kind])
            if key:
                routing.send(key, obs)
            continue
        key = (named_entity(model, obs.subject) if kind == PageKind.TOPIC else None
               ) or model.resolve(kind, obs.subject)
        if key:
            routing.send(key, obs)
            continue
        norm = norm_label(obs.subject) or obs.subject.casefold()
        group = groups.setdefault((kind, norm), _Group(kind, obs.subject, {norm: obs.subject}))
        group.observations.append(obs)

    clustered = _cluster(model, list(groups.values()), embedder)
    merged = await _merge_labels(model, clustered, owner, llm, routing, spend)

    for group in merged:
        if group.target:
            key = group.target
        else:
            count = sum(label_counts.get((str(group.kind), n), 0) for n in group.labels)
            count = max(count, len(group.observations))
            in_chat = None
            if group.kind == PageKind.PERSON:
                in_chat = next(filter(None, (chat_person(label, people)
                                             for label in group.labels.values())), None)
            needed = min_observations
            if group.kind == PageKind.HEALTH:  # worth a page sooner, but not one per mention
                needed = min(2, min_observations)
            if in_chat or count >= needed:
                title = in_chat or group.display
                key = model.resolve(group.kind, title) if in_chat else None
                aliases = sorted({d for d in group.labels.values() if d != title})
                key = key or _ensure_page(model, routing, group.kind, title, title, aliases)
            else:
                key = _ensure_page(model, routing, group.kind, CATCH_ALL_SLUG,
                                   CATCH_ALL_TITLES[group.kind])
        for obs in group.observations:
            if key:
                routing.send(key, obs)
    return routing


def named_entity(model: SelfModel, label: str) -> str | None:
    """The person page (someone, or a pet) a topic label names exactly: "Molly (dog)" is
    the page "Molly". A topic about a named someone belongs on their page."""
    name = bare_name(label)
    return model.resolve(PageKind.PERSON, name, fuzzy=False) if name else None


def fold_named_topics(model: SelfModel, routing: Routing) -> None:
    """Merge topic pages that name a person page into it (made before this rule existed)."""
    for page in model.pages(PageKind.TOPIC):
        target = named_entity(model, page.title)
        if target is None:
            continue
        try:
            model.merge_page(page.key, target)
        except ValueError as err:
            log.info("route: %s", err)
            continue
        routing.merged.append((page.key, target))


#: Words left out when comparing labels ("my car" is "car", "sleep and lifestyle" starts
#: with "sleep").
_STOPWORDS = {"the", "a", "an", "and", "of", "my", "for", "with", "in", "on", "to", "at",
              "de", "la", "le", "les", "et", "du", "des", "d", "l", "mon", "ma", "mes",
              "à", "au", "aux", "en", "pour", "avec"}
#: Cosine similarity above which two labels of one kind name the same subject. Measured on
#: real labels with the multilingual MiniLM: same subject across languages scores 0.88 to
#: 0.99 ("energy"/"énergie"), neighbouring subjects stay under 0.85 ("covid"/"courbatures").
SAME_SUBJECT = 0.86


def _words(label: str) -> tuple[str, ...]:
    """Content words of a label, crude singulars, in order."""
    words = [w for w in norm_label(label).split() if w not in _STOPWORDS]
    return tuple(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words)


class _Sets:
    """Union-find over 0..n-1."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        self.parent[self.find(i)] = self.find(j)


def _cluster(model: SelfModel, groups: list[_Group], embedder: Embedder | None) -> list[_Group]:
    """Merge groups that are plainly one subject, without asking the model.

    Two rules, within one kind and never for people (names that look alike can be
    different people): a label that starts with a shorter label's words joins it ("car
    repair", "my car" join "car"), and labels the embedding model finds near-identical
    join each other ("sommeil", "sleep"). A group near-identical to an existing page's
    title goes to that page.
    """
    out: list[_Group] = []
    by_kind: dict[PageKind, list[_Group]] = {}
    for group in groups:
        if group.kind == PageKind.PERSON:
            out.append(group)
        else:
            by_kind.setdefault(group.kind, []).append(group)

    for kind, members in by_kind.items():
        sets = _Sets(len(members))
        union, find = sets.union, sets.find
        words = [_words(g.display) for g in members]
        index: dict[tuple[str, ...], int] = {}
        for i, w in enumerate(words):
            if w and w in index:  # same words ("my car", "car")
                union(i, index[w])
            elif w:
                index[w] = i
        for i, w in enumerate(words):
            for n in range(len(w) - 1, 0, -1):  # the longest shorter label it starts with
                j = index.get(w[:n])
                if j is not None:
                    union(i, j)
                    break

        pages = [p for p in model.pages(kind) if p.key.rsplit("/", 1)[-1] != CATCH_ALL_SLUG]
        if embedder is not None:
            labels = [g.display for g in members] + [p.title for p in pages]
            try:
                vectors = normalize_rows(np.asarray(embedder.embed(labels)))
            except Exception as error:  # noqa: BLE001 - grouping works without it
                log.warning("route: label embedding failed: %s", error)
                vectors = None
            if vectors is not None:
                sims = vectors[: len(members)] @ vectors.T
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        if sims[i, j] >= SAME_SUBJECT:
                            union(i, j)
                    if pages:
                        best = int(np.argmax(sims[i, len(members):]))
                        if sims[i, len(members) + best] >= SAME_SUBJECT:
                            members[i].target = members[i].target or pages[best].key

        components: dict[int, list[_Group]] = {}
        for i, group in enumerate(members):
            components.setdefault(find(i), []).append(group)
        for parts in components.values():
            head = max(parts, key=lambda g: (len(g.observations), -len(g.display)))
            joined = _join(None, head)
            joined.target = next((g.target for g in parts if g.target), None)
            for other in parts:
                if other is not head:
                    _join(joined, other)
            out.append(joined)
    return out


def _worth_asking(model: SelfModel, groups: list[_Group]) -> bool:
    """A kind with two new labels, or a new label and pages it might belong to."""
    per_kind = Counter(g.kind for g in groups)
    return any(n > 1 or any(p.key.rsplit("/", 1)[-1] != CATCH_ALL_SLUG
                            for p in model.pages(kind))
               for kind, n in per_kind.items())


def _batches(model: SelfModel, groups: list[_Group]) -> list[list[_Group]]:
    """Groups worth asking about, one kind per batch, at most ROUTE_BATCH labels each.

    Labels can only merge within a kind, so splitting by kind loses nothing; a long
    reply is what makes a weaker model stop short.
    """
    by_kind: dict[PageKind, list[_Group]] = {}
    for group in [g for g in groups if not g.target][:MAX_ROUTE_LABELS]:
        by_kind.setdefault(group.kind, []).append(group)
    out: list[list[_Group]] = []
    for kind_groups in by_kind.values():
        if not _worth_asking(model, kind_groups):
            continue
        for start in range(0, len(kind_groups), ROUTE_BATCH):
            out.append(kind_groups[start:start + ROUTE_BATCH])
    return out


async def _ask(model: SelfModel, batch: list[_Group], owner: OwnerConfig, llm: LLMClient,
               routing: Routing, spend: Spend | None) -> list[RouteDecision]:
    try:
        response = await llm.complete(_route_request(model, batch, owner))
    except Exception as error:  # noqa: BLE001 - routing works without the merge
        routing.errors.append(f"route: {error}")
        return []
    if spend is not None:
        spend.add("smart", response)
    if not isinstance(response.parsed, RouteResult):
        routing.errors.append("route: the reply had no decisions")
        return []
    return response.parsed.decisions


async def _merge_labels(model: SelfModel, groups: list[_Group], owner: OwnerConfig,
                        llm: LLMClient | None, routing: Routing,
                        spend: Spend | None) -> list[_Group]:
    """Ask the model which labels are the same subject; on any failure, keep them apart."""
    if llm is None:
        return groups
    batches = _batches(model, groups)
    if not batches:
        return groups
    replies = await asyncio.gather(*(_ask(model, batch, owner, llm, routing, spend)
                                     for batch in batches))
    decided = [d for reply in replies for d in reply]

    decisions = {(d.kind.strip().lower(), norm_label(d.label)): d for d in decided}
    out: dict[tuple[PageKind, str], _Group] = {}
    for group in groups:
        if group.target:  # already matched to a page by its embedding
            key = (group.kind, "page:" + group.target)
            out[key] = _join(out.get(key), group)
            continue
        decision = decisions.get((str(group.kind), norm_label(group.display)))
        if decision and decision.existing_page:
            page = model.page(decision.existing_page.strip().removesuffix(".md"))
            if page is not None and page.kind == group.kind:
                group.target = page.key
                out[(group.kind, "page:" + page.key)] = _join(
                    out.get((group.kind, "page:" + page.key)), group)
                continue
        canonical = " ".join((decision.canonical if decision else "").split()) or group.display
        key = (group.kind, norm_label(canonical) or canonical.casefold())
        joined = _join(out.get(key), group)
        joined.display = canonical
        joined.labels.setdefault(key[1], canonical)
        out[key] = joined
    return list(out.values())


def _join(into: _Group | None, group: _Group) -> _Group:
    if into is None:
        return _Group(group.kind, group.display, dict(group.labels), list(group.observations),
                      group.target)
    into.labels.update(group.labels)
    into.observations.extend(group.observations)
    return into
