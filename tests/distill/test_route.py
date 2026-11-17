"""Routing observations to pages: known pages, new pages over a threshold, catch-alls."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from repartee.distill.route import (
    RouteDecision,
    RouteResult,
    chat_person,
    norm_label,
    route,
)
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
    assert ids("health/misc") == ["ob-6"]  # one health mention: the catch-all
    assert ids("timeline/2021") == ["ob-7"]
    assert routing.dropped == ["ob-9"]
    assert set(routing.created) == {"topics/urban-cycling", "topics/misc", "health/misc",
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


def test_chat_person_matches_either_way_round():
    people = ["Léo Garnier", "Elliot", "Ana", "Ana Ruiz", "Sam", "Samuel"]
    assert chat_person("Léo", people) == "Léo Garnier"
    assert chat_person("Elliot Price", people) == "Elliot"  # saved under a first name only
    assert chat_person("Ana Ruiz Vidal", people) == "Ana Ruiz"  # the longest name wins
    assert chat_person("Ana", people) == "Ana"  # start of two names: unclear, keep the label
    assert chat_person("Marc", people) is None
    assert chat_person("Elliotte", people) is None  # whole words only


async def test_route_full_name_lands_on_the_chat_contact_page(env):
    pending = [ob(1, "person", "Elliot"), ob(2, "person", "Elliot Price")]
    routing = await _route(env, pending, people=["Elliot"])
    assert list(routing.pages) == ["people/elliot"]
    assert len(routing.pages["people/elliot"]) == 2


async def test_route_asks_per_kind_in_batches_and_merges_across_them(env, monkeypatch):
    import repartee.distill.route as route_module

    monkeypatch.setattr(route_module, "ROUTE_BATCH", 2)
    pending = [ob(1, "health", "knee"), ob(2, "health", "right knee"),
               ob(3, "health", "patella"), ob(4, "topic", "jazz"), ob(5, "topic", "bebop")]
    env.fake.on("route", RouteResult(decisions=[
        RouteDecision(kind="health", label="right knee", canonical="knee"),
        RouteDecision(kind="health", label="patella", canonical="knee"),
    ]))
    routing = await _route(env, pending, llm=env.fake)

    requests = env.fake.calls("route")
    assert len(requests) == 3  # health in two batches, topic in one
    assert all("kind=topic" not in r.messages[-1].text for r in requests
               if "kind=health" in r.messages[-1].text)
    assert sorted(o.id for o in routing.pages["health/knee"]) == ["ob-1", "ob-2", "ob-3"]


async def test_route_merges_prefixes_and_near_identical_labels_without_the_model(env):
    import numpy as np

    class Fixed:
        """'sommeil' and 'sleep' share a vector; everything else is far apart."""

        name, dim = "fixed", 3

        def embed(self, texts):
            table = {"sleep": [1, 0, 0], "sommeil": [1, 0, 0], "Sleep": [1, 0, 0]}
            out = []
            for n, text in enumerate(texts):
                out.append(table.get(text, [0, np.cos(n), np.sin(n)]))
            return np.asarray(out, dtype=np.float32)

    env.model.apply([CreatePage(kind=PageKind.TOPIC, slug="cars", title="Cars")])
    pending = [ob(1, "health", "sleep"), ob(2, "health", "sommeil"),
               ob(3, "health", "sleep and lifestyle"), ob(4, "behaviour", "my car"),
               ob(5, "behaviour", "car"), ob(6, "behaviour", "car repair"),
               ob(7, "behaviour", "energy at work"), ob(8, "behaviour", "work"),
               ob(9, "person", "Alex"), ob(10, "person", "Alexandre")]
    routing2 = await route(env.model, pending, label_counts={}, chat_people=[],
                           owner=env.config.owner, min_observations=3, llm=None,
                           embedder=Fixed())

    def ids(key):
        return sorted(o.id for o in routing2.pages.get(key, []))

    assert ids("health/sleep") == ["ob-1", "ob-2", "ob-3"]
    assert ids("behaviour/car") == ["ob-4", "ob-5", "ob-6"]
    # "energy at work" does not start with "work": kept apart, too small for a page
    assert "ob-7" in ids("behaviour/misc") and "ob-8" in ids("behaviour/misc")
    # people are never merged this way
    assert ids("people/misc") == ["ob-10", "ob-9"]
