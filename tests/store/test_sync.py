from __future__ import annotations

from datetime import datetime

from repartee.schema import Kind
from repartee.store.corpus import Corpus, SyncStats
from repartee.store.embed import HashEmbedder

from .conftest import RecordingEmbedder, make_chunk, make_item


def test_sync_items_add_update_remove_unchanged(corpus):
    first = [make_item("a", "one"), make_item("b", "two"), make_item("c", "three")]
    assert corpus.sync_items(first) == SyncStats(added=3)
    second = [make_item("a", "one"), make_item("b", "two, edited"), make_item("d", "four")]
    stats = corpus.sync_items(second)
    assert (stats.added, stats.updated, stats.removed, stats.unchanged) == (1, 1, 1, 1)
    assert stats.remapped == {}
    assert corpus.get_item("b").text == "two, edited"
    assert corpus.get_item("c") is None
    assert corpus.sync_items(second) == SyncStats(unchanged=3)


def test_sync_items_metadata_change_counts_as_update(corpus):
    corpus.sync_items([make_item("a", "one")])
    stats = corpus.sync_items([make_item("a", "one", holdout=True)])
    assert stats.updated == 1
    assert corpus.get_item("a").holdout is True


def test_sync_items_reports_remap(corpus):
    corpus.sync_items([
        make_item("nt-old", "A paragraph about gardens.", source="vault", container="old.md"),
        make_item("nt-keep", "Unrelated.", source="vault", container="other.md"),
    ])
    stats = corpus.sync_items([
        make_item("nt-new", "A paragraph about gardens.", source="vault", container="new.md"),
        make_item("nt-keep", "Unrelated.", source="vault", container="other.md"),
    ])
    assert stats.remapped == {"nt-old": "nt-new"}
    assert (stats.added, stats.removed, stats.unchanged) == (1, 1, 1)


def test_remap_is_one_to_one_in_order_and_same_source_only(corpus):
    corpus.sync_items([
        make_item("x1", "ok", source="vault", container="a.md", seq=0),
        make_item("x2", "ok", source="vault", container="a.md", seq=1),
        make_item("x3", "ok", source="vault", container="a.md", seq=2),
        make_item("w1", "same words", source="whatsapp"),
        make_item("blank", "   ", source="vault"),
    ])
    stats = corpus.sync_items([
        make_item("y1", "ok", source="vault", container="b.md", seq=0),
        make_item("y2", "ok", source="vault", container="b.md", seq=1),
        make_item("v1", "same words", source="vault"),  # other source: not a remap
        make_item("blank2", "   ", source="vault"),  # empty text is never remapped
    ])
    assert stats.remapped == {"x1": "y1", "x2": "y2"}
    assert stats.removed == 5 and stats.added == 4


def test_remap_ignores_text_that_survives_unchanged(corpus):
    corpus.sync_items([make_item("a", "hello", source="vault"),
                       make_item("b", "hello", source="vault")])
    stats = corpus.sync_items([make_item("a", "hello", source="vault"),
                               make_item("c", "hello", source="vault")])
    assert stats.remapped == {"b": "c"}


def test_duplicate_ids_in_input_last_wins(corpus):
    stats = corpus.sync_items([make_item("a", "first"), make_item("a", "second")])
    assert stats.added == 1
    assert corpus.get_item("a").text == "second"


def test_sync_chunks_and_lookups(corpus):
    corpus.sync_items([make_item("i1", "salut"), make_item("i2", "ça va ?")])
    ch = make_chunk("ch1", "Ana: salut\nAna: ça va ?", item_ids=["i1", "i2"])
    assert corpus.sync_chunks([ch]) == SyncStats(added=1)
    assert corpus.get_chunk("ch1") == ch
    assert corpus.chunk_for_item("i2").id == "ch1"
    assert corpus.chunk_for_item("nope") is None
    assert corpus.chunk_hash("ch1") is not None
    assert corpus.chunk_hash("missing") is None
    before = corpus.chunk_hash("ch1")
    corpus.sync_chunks([make_chunk("ch1", "Ana: salut!", item_ids=["i1", "i2"])])
    assert corpus.chunk_hash("ch1") != before
    corpus.sync_chunks([])
    assert corpus.get_chunk("ch1") is None
    assert corpus.chunk_for_item("i1") is None


def test_get_items_keeps_order_and_existing_ids(corpus):
    corpus.sync_items([make_item(f"i{n}", f"text {n}") for n in range(1200)])
    got = corpus.get_items(["i5", "missing", "i1100", "i2"])
    assert [i.id for i in got] == ["i5", "i1100", "i2"]
    wanted = [f"i{n}" for n in range(0, 1300, 7)]
    assert corpus.existing_item_ids(wanted) == {w for w in wanted if int(w[1:]) < 1200}
    assert corpus.existing_item_ids([]) == set()


def test_reembeds_only_changed_chunks(tmp_path):
    rec = RecordingEmbedder()
    c = Corpus(tmp_path / "c.db", embedder=rec)
    chunks = [make_chunk(f"c{n}", f"chunk number {n} about topic {n}") for n in range(5)]
    c.sync_chunks(chunks)
    assert len(rec.embedded) == 5
    rec.calls.clear()
    c.sync_chunks(chunks)
    assert rec.embedded == []
    chunks[2] = make_chunk("c2", "rewritten text")
    stats = c.sync_chunks(chunks + [make_chunk("c9", "brand new")])
    assert (stats.added, stats.updated, stats.unchanged) == (1, 1, 4)
    assert len(rec.embedded) == 2 and any("rewritten text" in t for t in rec.embedded)
    rec.calls.clear()
    # Changing only metadata does not change the embedded text: no re-embedding.
    chunks[0] = make_chunk("c0", chunks[0].text, participants=["Ana Ruiz", "Camille"])
    assert c.sync_chunks(chunks + [make_chunk("c9", "brand new")]).updated == 1
    assert rec.embedded == []
    c.close()


def test_changing_embedder_reembeds(tmp_path):
    chunks = [make_chunk(f"c{n}", f"text {n}") for n in range(3)]
    c = Corpus(tmp_path / "c.db", embedder=RecordingEmbedder(name="model-a"))
    c.sync_chunks(chunks)
    c.close()
    rec = RecordingEmbedder(name="model-b")
    c = Corpus(tmp_path / "c.db", embedder=rec)
    c.sync_chunks(chunks)
    assert len(rec.embedded) == 3
    assert c.stats()["embedded_chunks"] == 3
    c.close()


def test_embeddings_catch_up_after_keyword_only_sync(tmp_path):
    chunks = [make_chunk(f"c{n}", f"text {n}") for n in range(3)]
    c = Corpus(tmp_path / "c.db")
    c.sync_chunks(chunks)
    assert c.stats()["embedded_chunks"] == 0
    c.close()
    c = Corpus(tmp_path / "c.db", embedder=HashEmbedder())
    assert c.sync_chunks(chunks).unchanged == 3
    assert c.stats()["embedded_chunks"] == 3
    c.close()


def test_embedder_failing_during_sync_degrades_to_keyword(tmp_path):
    from repartee.store.embed import EmbedderUnavailable

    class Broken:
        name = "broken"
        dim = 8

        def embed(self, texts):
            raise EmbedderUnavailable("model download failed")

    c = Corpus(tmp_path / "c.db", embedder=Broken())
    stats = c.sync_chunks([make_chunk("c1", "the bicycle is red")])
    assert stats.added == 1
    s = c.stats()
    assert s["embedder"] is None and "download failed" in s["embedder_error"]
    assert [h.chunk.id for h in c.search("bicycle")] == ["c1"]
    c.close()


def test_iter_items_holdout_modes(corpus):
    corpus.sync_items([
        make_item("a", "one", is_self=True, author="Camille", holdout=True, seq=0),
        make_item("b", "two", is_self=True, author="Camille", seq=1),
        make_item("c", "three", seq=2),
    ])
    assert [i.id for i in corpus.iter_items()] == ["a", "b", "c"]
    assert [i.id for i in corpus.iter_items(holdout=True)] == ["a"]
    assert [i.id for i in corpus.iter_items(holdout=False)] == ["b", "c"]
    from repartee.schema import SearchFilters

    own = corpus.iter_items(SearchFilters(author="self"), holdout=False)
    assert [i.id for i in own] == ["b"]


def test_iter_items_and_chunks_filters(corpus):
    corpus.sync_items([
        make_item("m1", "hi", ts=datetime(2021, 1, 1), seq=0),
        make_item("m2", "hey", ts=datetime(2023, 1, 1), seq=1, author="Léo Petit",
                  container="Léo Petit"),
        make_item("n1", "note", source="vault", kind=Kind.NOTE, container="a.md",
                  author="Camille", is_self=True),
    ])
    from repartee.schema import SearchFilters

    f = SearchFilters(after=datetime(2022, 1, 1))
    assert [i.id for i in corpus.iter_items(f)] == ["m2"]
    assert [i.id for i in corpus.iter_items(SearchFilters(participant="leo"))] == ["m2"]
    assert [i.id for i in corpus.iter_items(SearchFilters(kinds=[Kind.NOTE]))] == ["n1"]
    assert [i.id for i in corpus.iter_items(SearchFilters(author="other"))] == ["m1", "m2"]
    corpus.sync_chunks([make_chunk("c1", "x", source="vault", kind=Kind.NOTE, container="a.md"),
                        make_chunk("c2", "y")])
    assert [c.id for c in corpus.iter_chunks(SearchFilters(sources=["vault"]))] == ["c1"]
    assert len(list(corpus.iter_chunks())) == 2


def test_reopen_persists(tmp_path):
    path = tmp_path / "sub" / "c.db"
    c = Corpus(path, embedder=HashEmbedder())
    c.sync_items([make_item("a", "persisted text")])
    c.sync_chunks([make_chunk("c1", "persisted chunk about lighthouses", item_ids=["a"])])
    c.close()
    rec = RecordingEmbedder()
    c = Corpus(path, embedder=rec)
    assert c.get_item("a").text == "persisted text"
    assert c.sync_chunks([make_chunk("c1", "persisted chunk about lighthouses",
                                     item_ids=["a"])]).unchanged == 1
    assert rec.embedded == []
    assert [h.chunk.id for h in c.search("lighthouses", mode="semantic")] == ["c1"]
    c.close()
