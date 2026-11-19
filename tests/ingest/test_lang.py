"""Offline language detection."""

from __future__ import annotations

from repartee.config import SourceConfig
from repartee.ingest import ingest_source
from repartee.ingest.lang import detect_language

from .helpers import write


def test_detect_language():
    assert detect_language("I think we should go to the beach this weekend") == "en"
    assert detect_language("Je pense qu'on devrait aller à la plage ce week-end") == "fr"
    assert detect_language("Creo que deberíamos ir a la playa este fin de semana") == "es"
    assert detect_language("Ich glaube, wir sollten am Wochenende an den Strand gehen") == "de"
    assert detect_language("Penso che dovremmo andare al mare questo fine settimana") == "it"
    assert detect_language("Acho que devíamos ir à praia neste fim de semana") == "pt"
    assert detect_language("Ik denk dat we dit weekend naar het strand moeten gaan") == "nl"
    assert detect_language("ok") is None
    assert detect_language("") is None
    assert detect_language("😂😂") is None
    assert detect_language("12:30") is None


def test_detect_language_by_script():
    assert detect_language("Καλημέρα, τι κάνεις σήμερα;") == "el"
    assert detect_language("오늘 날씨가 정말 좋네요") == "ko"
    assert detect_language("今日はとても良い天気ですね") == "ja"


def test_short_messages_inherit_language(tmp_path, owner):
    lines = [
        "12/03/2021, 14:00 - Ana: je pense qu'on peut y aller demain matin",
        "12/03/2021, 14:01 - Cam: ok",
        "12/03/2021, 14:02 - Ana: tu es sûre que c'est une bonne idée ?",
        "12/03/2021, 14:03 - Cam: oui, je suis sûre, on y va",
    ]
    write(tmp_path / "chat.txt", "\n".join(lines) + "\n")
    items, _ = ingest_source(SourceConfig(type="whatsapp", path="."), tmp_path, owner)
    assert [i.lang for i in items] == ["fr", "fr", "fr", "fr"]
    assert items[1].meta.get("lang_inherited") is True
    assert "lang_inherited" not in items[0].meta


def test_stray_language_on_short_message_is_corrected(tmp_path, owner):
    # "Parfois. Ha ha" alone looks Italian to a stopword detector; in a French chat it is French.
    french = [
        "je pense qu'on peut y aller demain matin",
        "tu es sûre que c'est une bonne idée ?",
        "oui, je suis sûre, on y va",
        "il faut que je passe chez le médecin avant",
        "on se retrouve devant la gare vers midi",
        "je ne sais pas encore ce que je vais faire",
    ]
    lines = [f"12/03/2021, 14:{i:02d} - Ana: {french[i % len(french)]}" for i in range(30)]
    lines.insert(15, "12/03/2021, 14:15 - Maman: Parfois. Ha ha")
    write(tmp_path / "chat.txt", "\n".join(lines) + "\n")
    items, _ = ingest_source(SourceConfig(type="whatsapp", path="."), tmp_path, owner)
    stray = next(i for i in items if i.text == "Parfois. Ha ha")
    assert stray.lang == "fr"
    assert stray.meta.get("lang_inherited") is True


def test_long_message_in_another_language_is_kept(tmp_path, owner):
    french = "je pense qu'on peut y aller demain matin si tu veux"
    lines = [f"12/03/2021, 14:{i:02d} - Ana: {french}" for i in range(30)]
    english = "honestly I think we should just stay here and wait for the others to arrive"
    lines.insert(15, f"12/03/2021, 14:15 - Cam: {english}")
    write(tmp_path / "chat.txt", "\n".join(lines) + "\n")
    items, _ = ingest_source(SourceConfig(type="whatsapp", path="."), tmp_path, owner)
    kept = next(i for i in items if i.text == english)
    assert kept.lang == "en"
