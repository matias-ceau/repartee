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

import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

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

log = logging.getLogger(__name__)

#: Titles of the per-kind pages for subjects with too little material of their own.
CATCH_ALL_TITLES: dict[PageKind, str] = {
    PageKind.PERSON: "People in passing",
    PageKind.TOPIC: "Other topics",
    PageKind.TASTE: "Other tastes",
    PageKind.BEHAVIOUR: "Other patterns",
    PageKind.HEALTH: "Other health notes",
}
MAX_ROUTE_LABELS = 400  # labels per route request; the rest are filed as they are
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
    """The chat participant a label names (full name, or the start of it); None if no match.

    When the label is the start of several names, the label itself is returned: it is a
    participant, but which one is unclear.
    """
    want = norm_label(label).split()
    if not want:
        return None
    found = {person for person in people
             if norm_label(person).split()[: len(want)] == want}
    if len(found) == 1:
        return found.pop()
    return label if found else None


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
        tier="bulk",
        purpose="route",
        system=[SystemBlock(text=fill(ROUTE_SYSTEM, owner.name), cache=True)],
        messages=[Message(role="user", text="\n".join(lines))],
        output_schema=RouteResult,
        max_tokens=8000,
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
) -> Routing:
    """Assign pending observations to pages, creating pages as the rules allow.

    ``label_counts`` maps (subject kind, normalized label) to the number of current
    observations with that label, consumed or not, for the new-page threshold.
    """
    routing = Routing()
    people = [p for p in chat_people if not owner.is_self(p)]
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
        key = model.resolve(kind, obs.subject)
        if key:
            routing.send(key, obs)
            continue
        norm = norm_label(obs.subject) or obs.subject.casefold()
        group = groups.setdefault((kind, norm), _Group(kind, obs.subject, {norm: obs.subject}))
        group.observations.append(obs)

    merged = await _merge_labels(model, list(groups.values()), owner, llm, routing, spend)

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
            if in_chat or group.kind == PageKind.HEALTH or count >= min_observations:
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


def _worth_asking(model: SelfModel, groups: list[_Group]) -> bool:
    """A kind with two new labels, or a new label and pages it might belong to."""
    per_kind = Counter(g.kind for g in groups)
    return any(n > 1 or any(p.key.rsplit("/", 1)[-1] != CATCH_ALL_SLUG
                            for p in model.pages(kind))
               for kind, n in per_kind.items())


async def _merge_labels(model: SelfModel, groups: list[_Group], owner: OwnerConfig,
                        llm: LLMClient | None, routing: Routing,
                        spend: Spend | None) -> list[_Group]:
    """Ask the model which labels are the same subject; on any failure, keep them apart."""
    if llm is None or not _worth_asking(model, groups):
        return groups
    asked = groups[:MAX_ROUTE_LABELS]
    try:
        response = await llm.complete(_route_request(model, asked, owner))
    except Exception as error:  # noqa: BLE001 - routing works without the merge
        routing.errors.append(f"route: {error}")
        return groups
    if spend is not None:
        spend.add("bulk", response)
    result = response.parsed
    if not isinstance(result, RouteResult):
        routing.errors.append("route: the reply had no decisions")
        return groups

    decisions = {(d.kind.strip().lower(), norm_label(d.label)): d for d in result.decisions}
    out: dict[tuple[PageKind, str], _Group] = {}
    for group in groups:
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
