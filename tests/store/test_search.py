from __future__ import annotations

from datetime import datetime

import pytest

from repartee.schema import ClaimStatus, Kind, SearchFilters

from .conftest import make_chunk, make_claim, make_pair

CHUNKS = [
    make_chunk("c-bike", "Ana: I bought a new bicycle yesterday, a red bicycle with a basket.",
               start=datetime(2021, 3, 12, 14, 0)),
    make_chunk("c-bike2", "Ana: my bicycle got stolen.", start=datetime(2022, 5, 1, 9, 0)),
    make_chunk("c-tax", "Ana: the tax office called about the form.",
               start=datetime(2023, 1, 1, 10, 0)),
]


def test_keyword_search_finds_and_ranks(corpus):
    corpus.sync_chunks(CHUNKS)
    hits = corpus.search("red bicycle basket", mode="keyword")
    assert [h.chunk.id for h in hits][:2] == ["c-bike", "c-bike2"]
    assert hits[0].score > hits[1].score
    assert "bicycle" in hits[0].snippet
    assert corpus.search("submarine", mode="keyword") == []


@pytest.mark.parametrize("query", [
    'what "about" (x) AND -y: *', '"', "NEAR(a b)", "^", "col:val", "''", "", "   ",
    "a" * 5000, "😀 ?!", "OR AND NOT", "\x00", "bicycle*", "{title text}: red",
])
def test_search_handles_punctuation_and_operators(corpus, query):
    corpus.sync_chunks(CHUNKS)
    for mode in ("hybrid", "keyword", "semantic"):
        assert isinstance(corpus.search(query, mode=mode), list)
    assert isinstance(corpus.search_claims(query), list)
    assert isinstance(corpus.search_reply_pairs(query), list)


def test_search_diacritics_folded(corpus):
    corpus.sync_chunks([make_chunk("c-fr", "Cet été on part à la plage."),
                        make_chunk("c-other", "Rien à voir.")])
    assert [h.chunk.id for h in corpus.search("ete", mode="keyword")] == ["c-fr"]
    assert [h.chunk.id for h in corpus.search("ÉTÉ", mode="keyword")] == ["c-fr"]
    assert "été" in corpus.search("ete", mode="keyword")[0].snippet


def test_hybrid_uses_embeddings(corpus):
    corpus.sync_chunks([
        make_chunk("c-target", "photosynthesizing chloroplasts everywhere"),
        make_chunk("c-noise1", "we went to the cinema on friday"),
        make_chunk("c-noise2", "dinner at my parents was long"),
    ])
    q = "photosynthesis chloroplast"  # no whole word in common with any chunk
    assert corpus.search(q, mode="keyword") == []
    assert corpus.search(q, mode="semantic")[0].chunk.id == "c-target"
    assert corpus.search(q, mode="hybrid")[0].chunk.id == "c-target"


def test_hybrid_fuses_both_rankings(corpus):
    corpus.sync_chunks(CHUNKS)
    hits = corpus.search("bicycle", k=3)
    assert {h.chunk.id for h in hits[:2]} == {"c-bike", "c-bike2"}
    assert all(0 < h.score <= 2 / 61 for h in hits)


def test_search_without_embedder_falls_back_to_keyword(keyword_corpus):
    keyword_corpus.sync_chunks(CHUNKS)
    for mode in ("hybrid", "semantic", "keyword"):
        assert keyword_corpus.search("bicycle", mode=mode)[0].chunk.id in {"c-bike", "c-bike2"}
    s = keyword_corpus.stats()
    assert s["embedder"] is None
    assert s["search"] == "keyword"


def test_search_k_limits(corpus):
    corpus.sync_chunks([make_chunk(f"c{n}", f"apple number {n}") for n in range(30)])
    assert len(corpus.search("apple", k=5)) == 5
    assert corpus.search("apple", k=0) == []


def test_snippet_is_a_window_around_the_match(corpus):
    long = ("filler words here. " * 60) + "the hidden treasure lies here. " + ("more." * 100)
    corpus.sync_chunks([make_chunk("c-long", long)])
    hit = corpus.search("treasure", mode="keyword")[0]
    assert "treasure" in hit.snippet
    assert len(hit.snippet) <= 260


def test_filters_participant_author_dates_kinds(corpus):
    corpus.sync_chunks([
        make_chunk("c1", "talk about bread", participants=["Ana Ruiz", "Camille"],
                   self_text="bread is good", start=datetime(2021, 1, 1)),
        make_chunk("c2", "talk about bread", participants=["Léo Petit"],
                   start=datetime(2023, 6, 1)),
        make_chunk("c3", "bread notes", source="vault", kind=Kind.NOTE, container="bread.md",
                   participants=["Camille"], self_text="bread notes"),
        make_chunk("c4", "bread in the group", participants=["~ Ana", "Bo"],
                   container="Group", start=datetime(2022, 1, 1)),
    ])

    def ids(**kw):
        return sorted(h.chunk.id for h in corpus.search("bread", SearchFilters(**kw), k=10))

    assert ids(participant="ana") == ["c1", "c4"]
    assert ids(participant="ANA RUIZ") == ["c1"]
    assert ids(participant="leo") == ["c2"]
    assert ids(participant="Ruiz") == ["c1"]
    assert ids(author="self") == ["c1", "c3"]
    assert ids(author="other") == ["c1", "c2", "c4"]
    assert ids(after=datetime(2022, 1, 1)) == ["c2", "c4"]
    assert ids(before=datetime(2022, 1, 1)) == ["c1", "c4"]
    assert ids(kinds=[Kind.NOTE]) == ["c3"]
    assert ids(sources=["whatsapp"], containers=["Group"]) == ["c4"]
    sem = corpus.search("bread", SearchFilters(kinds=[Kind.NOTE]), mode="semantic")
    assert [h.chunk.id for h in sem] == ["c3"]


def test_neighbors_same_container_ordered(corpus):
    corpus.sync_chunks([
        make_chunk("c3", "third", start=datetime(2021, 1, 3)),
        make_chunk("c1", "first", start=datetime(2021, 1, 1)),
        make_chunk("x", "elsewhere", container="Other", start=datetime(2021, 1, 2)),
        make_chunk("c2", "second", start=datetime(2021, 1, 2)),
        make_chunk("c4", "fourth", start=datetime(2021, 1, 4)),
    ])
    assert [c.id for c in corpus.neighbors("c2")] == ["c1", "c2", "c3"]
    assert [c.id for c in corpus.neighbors("c1", before=2, after=2)] == ["c1", "c2", "c3"]
    assert [c.id for c in corpus.neighbors("c4", before=1, after=0)] == ["c3", "c4"]
    assert corpus.neighbors("missing") == []


def test_neighbors_undated_use_input_order(corpus):
    corpus.sync_chunks([
        make_chunk(f"n{n}", f"section {n}", source="vault", kind=Kind.NOTE, container="a.md")
        for n in (0, 1, 2)
    ])
    assert [c.id for c in corpus.neighbors("n1")] == ["n0", "n1", "n2"]


def test_claims_index_skips_rejected(corpus):
    corpus.sync_claims([
        make_claim("c-1", "Ana and I cycle together on Sundays."),
        make_claim("c-2", "I cycle to work every day.", page="behaviour/commuting",
                   section="Pattern"),
        make_claim("c-3", "I hate cycling.", status=ClaimStatus.REJECTED),
        make_claim("c-4", "Cycling is my main hobby.", page="core", section="Who"),
    ])
    found = corpus.search_claims("cycle cycling")
    assert "c-3" not in [c.id for c, _ in found]
    assert {c.id for c, _ in found} >= {"c-1", "c-2"}
    from repartee.schema import PageKind

    assert [c.id for c, _ in corpus.search_claims("cycle", kind=PageKind.BEHAVIOUR)] == ["c-2"]
    assert [c.id for c, _ in corpus.search_claims("cycling hobby", kind=PageKind.CORE)] == ["c-4"]
    corpus.sync_claims([make_claim("c-1", "Ana and I cycle together on Sundays.",
                                   status=ClaimStatus.REJECTED)])
    assert corpus.search_claims("Sundays") == []
    assert corpus.stats()["claims"] == 1


def test_reply_pair_search_by_recipient(corpus):
    corpus.sync_reply_pairs([
        make_pair("p1", "Ana Ruiz: are you coming to dinner tonight?", ["yes!", "8pm?"]),
        make_pair("p2", "Léo Petit: are you coming to dinner tonight?", ["can't, sorry"],
                  recipient="Léo Petit", container="Léo Petit"),
        make_pair("p3", "Ana Ruiz: did you send the tax form?", ["not yet"]),
    ])
    got = corpus.search_reply_pairs("coming for dinner?", recipient="Ana Ruiz")
    assert got[0].id == "p1" and "p2" not in [p.id for p in got]
    assert corpus.search_reply_pairs("dinner tonight", recipient="leo")[0].id == "p2"
    assert corpus.stats()["reply_pairs"] == 3


def test_reply_pair_search_without_embedder(keyword_corpus):
    keyword_corpus.sync_reply_pairs([
        make_pair("p1", "Ana Ruiz: dinner tonight?", ["yes"]),
        make_pair("p2", "Ana Ruiz: tax form?", ["no"]),
    ])
    assert [p.id for p in keyword_corpus.search_reply_pairs("dinner")] == ["p1"]
    assert keyword_corpus.search_claims("anything") == []


def test_stats_keys(corpus):
    from .conftest import make_item

    corpus.sync_items([
        make_item("a", "x", ts=datetime(2020, 1, 1), is_self=True, author="Camille"),
        make_item("b", "y", ts=datetime(2024, 1, 1), source="vault", kind=Kind.NOTE),
    ])
    corpus.sync_chunks(CHUNKS)
    s = corpus.stats()
    for key in ("items", "chunks", "by_source", "by_kind", "self_items", "embedded_chunks",
                "embedder", "first", "last", "claims", "reply_pairs"):
        assert key in s
    assert s["items"] == 2 and s["chunks"] == 3 and s["self_items"] == 1
    assert s["by_source"] == {"vault": 1, "whatsapp": 1}
    assert s["by_kind"] == {"message": 1, "note": 1}
    assert s["embedded_chunks"] == 3 and s["embedder"] == "hash-256"
    assert s["first"] == datetime(2020, 1, 1) and s["last"] == datetime(2024, 1, 1)
    assert s["search"] == "hybrid"
    assert corpus.embedder_name == "hash-256"
