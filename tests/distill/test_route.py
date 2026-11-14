"""Routing observations to pages: known pages, new pages over a threshold, catch-alls."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from repartee.distill.route import RouteDecision, RouteResult, norm_label, route
from repartee.schema import CreatePage, Observation, PageKind
from tests.distill.helpers import OWNER, msg, one_per_chunk


def ob(n: int, kind: str, subject: str, *, when: str | None = None) -> Observation:
    return Observation(id=f"ob-{n}", chunk_id="ch-x", subject_kind=kind, subject=subject,
                       text=f"Observation {n} about {subject}.", when=when, evidence=["nt-1"])


def counts(observations):
    return dict(Counter((o.subject_kind, norm_label(o.subject)) for o in observations))


async def _route(env, pending, *, current=None, people=(), llm=None):
    return await route(env.model, pending, label_counts=counts(current or pending),
                       chat_people=people, owner=env.config.owner, min_observations=3,
                       llm=llm)


async def test_route_existing_alias_new_page_threshold_and_catch_all(env):
    env.model.apply([CreatePage(kind=PageKind.PERSON, slug="ana-ruiz", title="Ana Ruiz",
                                aliases=["Anita"])])
    pending = [
        ob(1, "person", "Anita"),
        ob(2, "topic", "cycling"), ob(3, "topic", "urban cycling"), ob(4, "topic", "Cycling"),
        ob(5, "topic", "knitting"),
        ob(6, "health", "migraines"),
        ob(7, "timeline", "moving", when="2021-05"),
        ob(8, "timeline", "Ana Ruiz"),
        ob(9, "timeline", "something"),
        ob(10, "person", "Cam"),
    ]
    env.fake.on("route", RouteResult(decisions=[
        RouteDecision(kind="topic", label="cycling", canonical="urban cycling"),
        RouteDecision(kind="topic", label="urban cycling", canonical="urban cycling"),
        RouteDecision(kind="topic", label="knitting", canonical="knitting"),
    ]))

    routing = await _route(env, pending, llm=env.fake)

    def ids(key):
        return sorted(o.id for o in routing.pages.get(key, []))

    assert ids("people/ana-ruiz") == ["ob-1", "ob-8"]
    assert ids("topics/urban-cycling") == ["ob-2", "ob-3", "ob-4"]
    assert ids("topics/misc") == ["ob-10", "ob-5"]
    assert ids("health/migraines") == ["ob-6"]
    assert ids("timeline/2021") == ["ob-7"]
    assert routing.dropped == ["ob-9"]
    assert set(routing.created) == {"topics/urban-cycling", "topics/misc", "health/migraines",
                                    "timeline/2021"}
    page = env.model.page("topics/urban-cycling")
    assert page.title == "urban cycling" and "cycling" in page.aliases
    assert env.model.page("topics/misc").title == "Other topics"
    (request,) = env.fake.calls("route")
    assert "people/ana-ruiz" not in request.messages[-1].text  # only kinds being routed
    assert "label='knitting'" in request.messages[-1].text


async def test_route_counts_earlier_observations_toward_threshold(env):
    earlier = [ob(1, "topic", "jazz"), ob(2, "topic", "jazz")]
    pending = [ob(3, "topic", "jazz")]
    routing = await _route(env, pending, current=earlier + pending)
    assert list(routing.pages) == ["topics/jazz"]


async def test_route_without_llm_or_on_failure_keeps_labels_apart(env):
    from repartee.llm import LLMError

    env.fake.on("route", LLMError("overloaded"))
    pending = [ob(1, "topic", "cycling"), ob(2, "topic", "urban cycling")]
    routing = await _route(env, pending, llm=env.fake)
    assert list(routing.pages) == ["topics/misc"]
    assert routing.errors and "overloaded" in routing.errors[0]


async def test_route_ignores_a_wrong_kind_existing_page(env):
    env.model.apply([CreatePage(kind=PageKind.PERSON, slug="jazz", title="Jazz")])
    env.fake.on("route", RouteResult(decisions=[
        RouteDecision(kind="taste", label="bebop", existing_page="people/jazz")]))
    routing = await _route(env, [ob(1, "taste", "bebop")], llm=env.fake)
    assert list(routing.pages) == ["tastes/misc"]


async def test_route_person_from_chat_exempt_from_threshold(env):
    env.index([
        msg("wa-1", "Léo Garnier", "On se voit demain ?", datetime(2022, 1, 5, 9, 0),
            container="Léo Garnier"),
        msg("wa-2", OWNER, "oui", datetime(2022, 1, 5, 9, 1), container="Léo Garnier", seq=1),
    ])
    env.fake.on("extract", one_per_chunk())
    distiller = env.distiller()
    people = distiller._chat_people()
    assert people == {"Léo Garnier", OWNER}

    routing = await route(distiller.model, [ob(1, "person", "Léo"), ob(2, "person", "Marc")],
                          label_counts={}, chat_people=people, owner=env.config.owner,
                          min_observations=3, llm=None)

    assert sorted(routing.pages) == ["people/léo-garnier", "people/misc"]
    page = distiller.model.page("people/léo-garnier")
    assert page.title == "Léo Garnier" and page.aliases == ["Léo"]
    assert distiller.model.page("people/misc").title == "People in passing"
