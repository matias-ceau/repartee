"""WhatsApp export parsing."""

from __future__ import annotations

import time
import zipfile
from datetime import datetime, timedelta

from repartee.config import SourceConfig
from repartee.ingest.whatsapp import WhatsAppAdapter, chat_name_for

from .helpers import parse_with, write

SRC = SourceConfig(type="whatsapp", path="sources/whatsapp")


def _parse(tmp_path, owner, text, name="WhatsApp Chat with Ana Ruiz.txt", source=SRC):
    path = write(tmp_path / name, text)
    return parse_with(WhatsAppAdapter(), path, tmp_path, source, owner)


def test_whatsapp_ios_brackets(tmp_path, owner):
    text = (
        "[12/03/2021, 14:05:33] Ana Ruiz: salut\n"
        "[12/03/2021, 14:06:01] Cam: salut toi\n"
        "[12/03/2021, 14:07:12] Ana Ruiz: ça va ?\n"
        "[13/03/2021, 09:00:00] Camille: oui et toi\n"
        "[13/03/2021, 09:01:00] Ana Ruiz: super\n"
    )
    items, report = _parse(tmp_path, owner, text)
    assert [i.author for i in items] == ["Ana Ruiz", "Cam", "Ana Ruiz", "Camille", "Ana Ruiz"]
    assert items[0].timestamp == datetime(2021, 3, 12, 14, 5, 33)
    assert items[3].timestamp == datetime(2021, 3, 13, 9, 0, 0)
    assert [i.is_self for i in items] == [False, True, False, True, False]
    assert {i.container for i in items} == {"Ana Ruiz"}
    assert [i.seq for i in items] == [0, 1, 2, 3, 4]
    assert all(i.kind == "message" and i.source == "whatsapp" for i in items)
    assert all(i.id.startswith("wa-") for i in items)
    assert report.unparsed_lines == 0
    assert report.items == 5


def test_whatsapp_android_dash(tmp_path, owner):
    text = (
        "12/03/2021, 14:05 - Ana Ruiz: salut\n"
        "12/03/2021, 14:06 - Camille: coucou\n"
        "25/03/2021, 08:00 - Ana Ruiz: debout ?\n"
    )
    items, report = _parse(tmp_path, owner, text)
    assert [i.text for i in items] == ["salut", "coucou", "debout ?"]
    assert items[1].timestamp == datetime(2021, 3, 12, 14, 6)
    assert items[2].timestamp == datetime(2021, 3, 25, 8, 0)
    assert items[1].is_self
    assert report.unparsed_lines == 0


def test_whatsapp_us_12h(tmp_path, owner):
    text = (
        "3/12/21, 2:05 PM - Ana: hi\n"
        "3/12/21, 2:06 PM - Cam: hey\n"
        "3/25/21, 12:15 AM - Ana: late one\n"
        "3/25/21, 12:30 PM - Ana: lunch?\n"
    )
    items, report = _parse(tmp_path, owner, text, name="WhatsApp Chat - Ana.txt")
    assert items[0].timestamp == datetime(2021, 3, 12, 14, 5)
    assert items[1].timestamp == datetime(2021, 3, 12, 14, 6)
    assert items[2].timestamp == datetime(2021, 3, 25, 0, 15)
    assert items[3].timestamp == datetime(2021, 3, 25, 12, 30)
    assert items[0].container == "Ana"
    assert report.unparsed_lines == 0


def test_whatsapp_date_variants(tmp_path, owner):
    text = (
        "[12.03.21, 14:05:33] Ana: hallo\n"
        "2021-03-13 09:00 - Ana: iso\n"
        "12/03/2021 à 15:00 - Ana: ancienne version\n"
        "‎[14/03/2021, 10:00:00] Ana: marque\n"
    )
    items, report = _parse(tmp_path, owner, text)
    assert [i.timestamp for i in items] == [
        datetime(2021, 3, 12, 14, 5, 33),
        datetime(2021, 3, 13, 9, 0),
        datetime(2021, 3, 12, 15, 0),
        datetime(2021, 3, 14, 10, 0),
    ]
    assert report.unparsed_lines == 0


def test_whatsapp_date_order_forced(tmp_path, owner):
    text = "03/04/21, 10:00 - Ana: a\n03/05/21, 10:00 - Ana: b\n"
    source = SourceConfig(type="whatsapp", path=".", options={"date_order": "mdy"})
    items, _ = _parse(tmp_path, owner, text, source=source)
    assert items[0].timestamp == datetime(2021, 3, 4, 10, 0)
    source = SourceConfig(type="whatsapp", path=".", options={"date_order": "dmy"})
    items, _ = _parse(tmp_path, owner, text, source=source)
    assert items[0].timestamp == datetime(2021, 4, 3, 10, 0)


def test_whatsapp_ambiguous_date_order_is_reported(tmp_path, owner):
    text = "03/04/21, 10:00 - Ana: a\n05/04/21, 10:00 - Ana: b\n"
    items, report = _parse(tmp_path, owner, text)
    assert items[0].timestamp == datetime(2021, 4, 3, 10, 0)  # day-first by default
    assert any("date_order" in w for w in report.warnings)


def test_whatsapp_ambiguous_order_uses_chronology(tmp_path, owner):
    # Month-first reading is chronological; day-first would jump back and forth.
    text = "01/02/21, 10:00 - Ana: a\n01/03/21, 10:00 - Ana: b\n02/01/21, 10:00 - Ana: c\n"
    items, report = _parse(tmp_path, owner, text)
    assert [i.timestamp.date().isoformat() for i in items] == [
        "2021-01-02",
        "2021-01-03",
        "2021-02-01",
    ]
    assert report.warnings


def test_whatsapp_conflicting_dates_are_not_guessed(tmp_path, owner):
    # Day-first and month-first evidence in one file: lines that do not fit are not placed.
    text = "25/03/21, 10:00 - Ana: a\n26/03/21, 10:00 - Ana: b\n03/27/21, 10:00 - Ana: c\n"
    items, report = _parse(tmp_path, owner, text)
    assert [i.text for i in items] == ["a", "b"]
    assert report.unparsed_lines == 1
    assert report.warnings


def test_whatsapp_multiline_message(tmp_path, owner):
    text = (
        "[12/03/2021, 14:05:33] Ana Ruiz: première ligne\n"
        "deuxième ligne\n"
        "\n"
        "troisième\n"
        "[12/03/2021, 14:06:00] Cam: ok\n"
    )
    items, _ = _parse(tmp_path, owner, text)
    assert items[0].text == "première ligne\ndeuxième ligne\n\ntroisième"
    assert items[1].text == "ok"


def test_whatsapp_skips_system_media_deleted_english(tmp_path, owner):
    text = (
        "12/03/2021, 14:00 - Messages and calls are end-to-end encrypted. No one outside of "
        "this chat, not even WhatsApp, can read or listen to them. Tap to learn more.\n"
        "12/03/2021, 14:01 - Ana created group \"Trip: Lisbon\"\n"
        "12/03/2021, 14:01 - Ana added Cam\n"
        "12/03/2021, 14:02 - Ana changed the subject from \"Trip\" to \"Trip: Lisbon\"\n"
        "12/03/2021, 14:03 - Ana: <Media omitted>\n"
        "12/03/2021, 14:04 - Ana: This message was deleted\n"
        "12/03/2021, 14:05 - Cam: You deleted this message\n"
        "12/03/2021, 14:06 - Ana: IMG-20210312-WA0001.jpg (file attached)\n"
        "12/03/2021, 14:07 - Ana: real text\n"
        "12/03/2021, 14:08 - Bob left\n"
        "12/03/2021, 14:09 - Ana Ruiz is a contact\n"
    )
    items, report = _parse(tmp_path, owner, text, name="WhatsApp Chat with Trip.txt")
    assert [i.text for i in items] == ["real text"]
    assert report.unparsed_lines == 0
    assert report.skipped["system"] == 6
    assert report.skipped["media"] == 2
    assert report.skipped["deleted"] == 2


def test_whatsapp_skips_system_media_deleted_ios_english(tmp_path, owner):
    text = (
        "[12/03/2021, 14:00:00] Ana Ruiz: ‎Messages and calls are end-to-end encrypted. "
        "No one outside of this chat, not even WhatsApp, can read or listen to them.\n"
        "[12/03/2021, 14:01:00] Ana Ruiz: ‎image omitted\n"
        "[12/03/2021, 14:02:00] Ana Ruiz: ‎<attached: 00000012-PHOTO-2021-03-12.jpg>\n"
        "[12/03/2021, 14:03:00] Cam: ‎This message was deleted.\n"
        "[12/03/2021, 14:04:00] Ana Ruiz: ‎Missed voice call, ‎Tap to call back\n"
        "[12/03/2021, 14:05:00] Ana Ruiz: kept\n"
    )
    items, report = _parse(tmp_path, owner, text)
    assert [i.text for i in items] == ["kept"]
    assert report.unparsed_lines == 0


def test_whatsapp_skips_system_media_deleted_french(tmp_path, owner):
    text = (
        "12/03/2021 14:00 - Les messages et les appels sont chiffrés de bout en bout. "
        "Aucun tiers, pas même WhatsApp, ne peut les lire ou les écouter.\n"
        "12/03/2021 14:01 - Ana a créé le groupe « Les copains »\n"
        "12/03/2021 14:01 - Ana vous a ajouté\n"
        "12/03/2021 14:02 - Léo a quitté\n"
        "12/03/2021 14:03 - Ana: <Médias omis>\n"
        "12/03/2021 14:04 - Ana: Ce message a été supprimé\n"
        "12/03/2021 14:05 - Camille: Vous avez supprimé ce message\n"
        "12/03/2021 14:06 - Ana: vrai texte\n"
    )
    items, report = _parse(tmp_path, owner, text, name="Discussion WhatsApp avec Les copains.txt")
    assert [i.text for i in items] == ["vrai texte"]
    assert items[0].container == "Les copains"
    assert report.unparsed_lines == 0


def test_whatsapp_media_with_caption_keeps_caption(tmp_path, owner):
    text = "12/03/2021, 14:06 - Ana: IMG-20210312-WA0001.jpg (file attached)\nlook at this\n"
    items, _ = _parse(tmp_path, owner, text)
    assert [i.text for i in items] == ["look at this"]
    assert items[0].meta.get("media") is True


def test_whatsapp_edited_marker_stripped(tmp_path, owner):
    text = (
        "[12/03/2021, 14:05:33] Ana Ruiz: on dit 20h ‎<This message was edited>\n"
        "12/03/2021, 14:06 - Cam: plutôt 21h <Ce message a été modifié>\n"
        "12/03/2021, 14:07 - Cam: pas modifié\n"
    )
    items, _ = _parse(tmp_path, owner, text)
    assert [i.text for i in items] == ["on dit 20h", "plutôt 21h", "pas modifié"]
    assert items[0].meta["edited"] is True
    assert items[1].meta["edited"] is True
    assert "edited" not in items[2].meta


def test_whatsapp_ordinary_text_resembling_system_is_kept(tmp_path, owner):
    text = "12/03/2021, 14:00 - Ana: Bob left\n12/03/2021, 14:01 - Ana: I left early\n"
    items, _ = _parse(tmp_path, owner, text)
    assert [i.text for i in items] == ["Bob left", "I left early"]


def test_whatsapp_unrecognised_system_line_is_counted(tmp_path, owner):
    text = "12/03/2021, 14:00 - Ana zrobiła coś dziwnego\n12/03/2021, 14:01 - Ana: ok\n"
    items, report = _parse(tmp_path, owner, text)
    assert [i.text for i in items] == ["ok"]
    assert report.unparsed_lines == 1
    assert any("zrobiła" in w for w in report.warnings)


def test_whatsapp_ids_stable_across_longer_export(tmp_path, owner):
    base = (
        "12/03/2021, 14:05 - Ana: hi\n"
        "12/03/2021, 14:05 - Ana: hi\n"
        "12/03/2021, 14:06 - Cam: hello\n"
    )
    more = (
        "13/03/2021, 10:00 - Ana: again\n"
        "13/03/2021, 10:01 - Cam: yes\n"
        "13/03/2021, 10:02 - Ana: hi\n"
    )
    short, _ = _parse(tmp_path / "a", owner, base)
    long, _ = _parse(tmp_path / "b", owner, base + more)
    assert [i.id for i in long[:3]] == [i.id for i in short]
    assert short[0].id != short[1].id
    assert len({i.id for i in long}) == 6


def test_whatsapp_unknown_format_counts_unparsed(tmp_path, owner):
    lines = [
        "2021年3月12日 14:05 Ana: こんにちは",
        "Ana Ruiz said hello at noon",
        "<<garbage>> 12-xx-2021",
        "‏‎",
        "[not a date] Ana: hi",
        "99/99/9999, 99:99 - Ana: impossible",
    ]
    items, report = _parse(tmp_path, owner, "\n".join(lines) + "\n\n")
    assert items == []
    assert report.unparsed_lines == 5  # the line of only direction marks is blank
    assert report.warnings


def test_whatsapp_binary_garbage_does_not_crash(tmp_path, owner):
    path = tmp_path / "weird.txt"
    path.write_bytes(b"\xff\xfe\x00\x01\x02garbage\x80\x81\n12/03/2021, 14:05 - Ana: hi\n")
    items, report = parse_with(WhatsAppAdapter(), path, tmp_path, SRC, owner)
    assert isinstance(items, list)


def test_whatsapp_zip_and_folder_exports(tmp_path, owner):
    root = tmp_path / "whatsapp"
    write(root / "Ana Ruiz" / "_chat.txt", "12/03/2021, 14:05 - Ana Ruiz: dossier\n")
    write(root / "Ana Ruiz" / "notes.md", "not a chat")
    zpath = root / "WhatsApp Chat - Léo.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("_chat.txt", "[12/03/2021, 14:05:33] Léo: zip\n")
        zf.writestr("00000001-PHOTO.jpg", b"\xff\xd8")
    adapter = WhatsAppAdapter()
    files = adapter.discover(root, SRC)
    assert sorted(p.name for p in files) == ["WhatsApp Chat - Léo.zip", "_chat.txt"]
    by_name = {}
    for f in files:
        items, _ = parse_with(adapter, f, root, SRC, owner)
        by_name[items[0].container] = items[0].text
    assert by_name == {"Ana Ruiz": "dossier", "Léo": "zip"}


def test_chat_name_prefixes():
    assert chat_name_for("WhatsApp Chat with Ana Ruiz.txt") == "Ana Ruiz"
    assert chat_name_for("WhatsApp Chat - Ana.txt") == "Ana"
    assert chat_name_for("Discussion WhatsApp avec Léo.txt") == "Léo"
    assert chat_name_for("family.txt") == "family"


def test_whatsapp_large_file_is_fast(tmp_path, owner):
    start = datetime(2021, 1, 1, 8, 0)
    lines = []
    for n in range(50_000):
        ts = start + timedelta(minutes=7 * n)
        who = "Cam" if n % 3 else "Ana Ruiz"
        lines.append(f"{ts:%d/%m/%Y}, {ts:%H:%M} - {who}: message number {n} with some words")
    path = write(tmp_path / "big.txt", "\n".join(lines) + "\n")
    t0 = time.perf_counter()
    items, report = parse_with(WhatsAppAdapter(), path, tmp_path, SRC, owner)
    elapsed = time.perf_counter() - t0
    assert len(items) == 50_000
    assert report.unparsed_lines == 0
    assert elapsed < 10


def test_whatsapp_year_first_two_digit_dates(tmp_path, owner):
    items, report = _parse(tmp_path, owner, "21. 3. 12. 오후 2:05 - Ana: 안녕\n")
    assert items[0].timestamp == datetime(2021, 3, 12, 14, 5)
    assert any("year-month-day" in w for w in report.warnings)
    # Day-first reading jumps between years; year-first is chronological.
    text = "21/03/30, 10:00 - Ana: a\n21/04/02, 10:00 - Ana: b\n21/04/15, 10:00 - Ana: c\n"
    items, _ = _parse(tmp_path / "x", owner, text)
    assert [i.timestamp.date().isoformat() for i in items] == [
        "2021-03-30",
        "2021-04-02",
        "2021-04-15",
    ]


def test_whatsapp_more_languages(tmp_path, owner):
    text = (
        "12.03.21, 14:05 - Ana: Hallo\n"
        "12.03.21, 14:06 - Ana hat Bob hinzugefügt\n"
        "12.03.21, 14:07 - Ana: <Medien ausgeschlossen>\n"
        "12.03.21, 14:08 - Ana: Diese Nachricht wurde gelöscht\n"
        "12/3/21, 2:05 p. m. - Ana: hola\n"
        "12/03/2021 14:06 - Ana saiu\n"
        "12/03/2021 14:08 - Ana: <Arquivo de mídia oculto>\n"
        "[12/03/21, 14:06:00] Ana: ‎immagine omessa\n"
        "[12/03/21, 14:07:00] Ana Ruiz: ‎Ana ha aggiunto Bob\n"
    )
    items, report = _parse(tmp_path, owner, text)
    assert [(i.text, i.timestamp.hour) for i in items] == [("Hallo", 14), ("hola", 14)]
    assert report.unparsed_lines == 0
