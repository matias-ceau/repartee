"""Whole-source and whole-home ingest."""

from __future__ import annotations

from repartee.config import HomeConfig, OwnerConfig, SourceConfig
from repartee.home import Home
from repartee.ingest import ADAPTERS, ingest_home, ingest_source
from repartee.schema import is_holdout

from .helpers import write
from .test_internal import DRAFT, TRANSCRIPT


def test_adapters_registered():
    assert set(ADAPTERS) == {"whatsapp", "vault", "writing", "tastes", "assistant"}


def test_holdout_only_on_owner_text_kinds(tmp_path, owner):
    lines = [
        f"12/03/2021, {n // 60:02d}:{n % 60:02d} - {'Cam' if n % 2 else 'Ana'}: message {n}"
        for n in range(600)
    ]
    write(tmp_path / "chat" / "c.txt", "\n".join(lines) + "\n")
    items, _ = ingest_source(SourceConfig(type="whatsapp", path="."), tmp_path / "chat", owner,
                             holdout_percent=20)
    assert any(i.holdout for i in items)
    for item in items:
        assert item.holdout == (item.is_self and is_holdout(item.id, 20))

    write(tmp_path / "tastes" / "books.md", "\n".join(f"- book {n}" for n in range(500)))
    tastes, _ = ingest_source(SourceConfig(type="tastes", path="."), tmp_path / "tastes", owner,
                              holdout_percent=100)
    assert tastes and not any(i.holdout for i in tastes)

    write(tmp_path / "vault" / "n.md", "a note\n")
    notes, _ = ingest_source(SourceConfig(type="vault", path="."), tmp_path / "vault", owner,
                             holdout_percent=100)
    assert all(i.holdout for i in notes)


def test_ingest_source_missing_root_is_a_warning(tmp_path, owner):
    items, report = ingest_source(SourceConfig(type="vault", path="x"), tmp_path / "nope", owner)
    assert items == [] and report.warnings


def test_ingest_source_dedupes_duplicate_exports(tmp_path, owner):
    chat = "12/03/2021, 14:05 - Ana: hi\n12/03/2021, 14:06 - Cam: yo\n"
    write(tmp_path / "Ana.txt", chat)
    write(tmp_path / "old" / "Ana.txt", chat)
    items, report = ingest_source(SourceConfig(type="whatsapp", path="."), tmp_path, owner)
    assert len(items) == 2
    assert any("duplicate" in w for w in report.warnings)


def test_ingest_home_includes_internal_folders_and_tolerates_missing(tmp_path):
    home = Home(tmp_path / "home")
    vault = tmp_path / "vault"
    write(vault / "note.md", "outside vault note\n")
    write(home.root / "sources" / "whatsapp" / "Ana.txt", "12/03/2021, 14:05 - Ana: hi\n")
    write(home.interviews_dir / "2026-11-14-childhood.md", TRANSCRIPT)
    write(home.drafts_dir / "20261114-180211-ab12.md", DRAFT)
    # conversations/ deliberately missing; one configured source missing too.
    config = HomeConfig(
        owner=OwnerConfig(name="Camille Martin", aliases=["Cam"]),
        sources=[
            SourceConfig(type="whatsapp", path="sources/whatsapp"),
            SourceConfig(type="vault", path=str(vault)),
            SourceConfig(type="tastes", path="sources/tastes"),
        ],
    )
    items, reports = ingest_home(home, config)
    sources = {i.source for i in items}
    assert sources == {"whatsapp", "vault", "interviews", "drafts"}
    assert {r.source for r in reports} >= {"whatsapp", "vault", "tastes", "interviews",
                                           "conversations", "drafts"}
    assert len({i.id for i in items}) == len(items)


def test_ingest_home_never_reads_self_model(tmp_path):
    home = Home(tmp_path / "home")
    write(home.self_dir / "people" / "ana.md", "self-model\n")
    write(home.root / "sources" / "vault" / "n.md", "note\n")
    (home.root / "sources" / "vault" / "self").symlink_to(home.self_dir, target_is_directory=True)
    config = HomeConfig(sources=[SourceConfig(type="vault", path="sources/vault")])
    items, _ = ingest_home(home, config)
    assert [i.container for i in items] == ["n.md"]
