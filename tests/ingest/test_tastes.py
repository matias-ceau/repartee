"""Taste exports: CSV, JSON listening history, markdown lists."""

from __future__ import annotations

import json

from repartee.config import SourceConfig
from repartee.ingest import ingest_source

from .helpers import write

SRC = SourceConfig(type="tastes", path="tastes")


def test_tastes_csv_json_markdown(tmp_path, owner):
    write(
        tmp_path / "music.csv",
        "Artist,Track,Plays,Year\nThe Owls,Night Bus,42,2021\nLa Rue,Été,7,2022\n",
    )
    history = [
        {"endTime": "2021-03-01 10:00", "artistName": "The Owls", "trackName": "Night Bus"},
        {"endTime": "2021-03-02 10:00", "artistName": "The Owls", "trackName": "Dawn"},
        {"endTime": "2022-01-01 10:00", "artistName": "La Rue", "trackName": "Été"},
        {"endTime": "2021-04-01 10:00", "artistName": "La Rue", "trackName": "Été"},
    ]
    write(tmp_path / "history.json", json.dumps(history))
    write(tmp_path / "books.md", "# Books\n\n- *The Long Walk* — loved it\n- Dune: too long\n")
    items, report = ingest_source(SRC, tmp_path, owner)
    by = {i.container: i for i in items}
    assert set(by) == {"music.csv", "history.json", "books.md"}
    assert all(i.kind == "taste" and i.is_self and i.id.startswith("ts-") for i in items)
    music = by["music.csv"].text.splitlines()
    assert any("The Owls" in line and "Night Bus" in line and "42" in line for line in music)
    hist = by["history.json"].text.splitlines()
    assert "2021 · The Owls: 2 plays" in hist
    assert "2021 · La Rue: 1 plays" in hist
    assert "2022 · La Rue: 1 plays" in hist
    assert by["books.md"].text.splitlines() == ["*The Long Walk* — loved it", "Dune: too long"]
    assert report.unparsed_lines == 0


def test_tastes_split_at_fifty_lines(tmp_path, owner):
    rows = "\n".join(f"Band {n},Song {n},{n},2020" for n in range(120))
    write(tmp_path / "music.csv", "artist,track,plays,year\n" + rows + "\n")
    items, _ = ingest_source(SRC, tmp_path, owner)
    assert [len(i.text.splitlines()) for i in items] == [50, 50, 20]
    assert len({i.id for i in items}) == 3


def test_tastes_bad_json_is_a_warning(tmp_path, owner):
    write(tmp_path / "broken.json", "{not json")
    items, report = ingest_source(SRC, tmp_path, owner)
    assert items == []
    assert report.warnings
