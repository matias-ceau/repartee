"""Checks on the fictional sample home in examples/persona."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import regex
import yaml

from repartee.config import HomeConfig

PERSONA = Path(__file__).resolve().parents[2] / "examples" / "persona"
WHATSAPP = PERSONA / "sources" / "whatsapp"
VAULT = PERSONA / "sources" / "vault"

# Chat file -> (header regex, name of Camille in that export, language register).
CHATS = {
    "Léa Fontaine/_chat.txt": (
        r"^‎?\[\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2}\] ",
        "Cam",
        "fr",
    ),
    "Copenhagen crew/_chat.txt": (
        r"^‎?\[\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2}\] ",
        "Camille",
        "en",
    ),
    "Discussion WhatsApp avec Maman.txt": (
        r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2} - ",
        "Camille",
        "fr",
    ),
    "WhatsApp Chat with Sam Whitaker.txt": (
        r"^\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2} [AP]M - ",
        "Camille Martin",
        "en",
    ),
}

NOT_TEXT = re.compile(
    r"omitted|absente|omis|retiré|supprimé|deleted|<Médias omis>|<Media omitted>"
    r"|chiffrés|encrypted|security code|code de sécurité|created group|changed the group"
)
EDITED = re.compile(r"\s*‎?<(This message was edited|Ce message a été modifié)>")


def parse_chat(rel: str) -> list[tuple[str, str]]:
    """(author, text) for every message line, continuation lines joined; system lines skipped."""
    header = re.compile(CHATS[rel][0])
    messages: list[tuple[str, str]] = []
    for line in (WHATSAPP / rel).read_text(encoding="utf-8").splitlines():
        match = header.match(line)
        if match:
            rest = line[match.end():]
            author, sep, text = rest.partition(": ")
            messages.append((author, text) if sep else ("", rest))
        elif messages:
            author, text = messages[-1]
            messages[-1] = (author, text + "\n" + line)
    return messages


def text_messages(rel: str, author: str | None = None, exclude: str | None = None) -> list[str]:
    """Plain text messages by one author (or everyone but `exclude`), markers removed."""
    out = []
    for who, text in parse_chat(rel):
        text = text.replace("‎", "")
        if not who or NOT_TEXT.search(text):
            continue
        if (author is not None and who != author) or (exclude is not None and who == exclude):
            continue
        out.append(EDITED.sub("", text).strip())
    return out


def lowercase_start_rate(texts: list[str]) -> float:
    letters = [t for t in texts if t[:1].isalpha()]
    return sum(t[0].islower() for t in letters) / len(letters)


def test_config_loads():
    config = HomeConfig.load(PERSONA / "repartee.yaml")
    assert config.owner.name == "Camille Martin"
    assert config.owner.is_self("Cam")
    assert config.owner.is_self("Camille")
    assert config.owner.is_self("Camille Martin")
    assert not config.owner.is_self("Léa")
    by_type = {source.type: source for source in config.sources}
    assert by_type["whatsapp"].path == "sources/whatsapp"
    assert by_type["vault"].exclude == ["Templates/**"]
    assert by_type["vault"].kinds == {
        "Journal/**": "journal",
        "Health/**": "medical",
        "Essays/**": "essay",
    }
    assert by_type["tastes"].path == "sources/tastes"


def test_layout():
    for rel in CHATS:
        assert (WHATSAPP / rel).is_file(), rel
    notes = [p for p in VAULT.rglob("*.md") if "Templates" not in p.parts]
    assert len(notes) >= 25
    journal = list((VAULT / "Journal").glob("*.md"))
    assert len(journal) == 15
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}\.md", p.name) for p in journal)
    assert len(list((VAULT / "Essays").glob("*.md"))) == 3
    assert len(list((VAULT / "Health").glob("*.md"))) == 3
    assert list((VAULT / "Templates").glob("*.md"))
    assert (PERSONA / "sources" / "tastes" / "music.csv").is_file()
    assert (PERSONA / "sources" / "tastes" / "books.md").is_file()
    assert (PERSONA / "interviews" / "2025-06-01-childhood.md").is_file()
    assert (PERSONA / "FACTS.md").is_file()
    assert "fictional" in (PERSONA / "README.md").read_text(encoding="utf-8").lower()


def test_vault_notes_have_frontmatter():
    for path in VAULT.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n"), path
        assert "\n---\n" in text[4:], path


def test_essays_are_about_600_words():
    for path in (VAULT / "Essays").glob("*.md"):
        words = len(path.read_text(encoding="utf-8").split("\n---\n", 1)[1].split())
        assert 500 <= words <= 750, (path.name, words)


def test_recall_file():
    entries = yaml.safe_load((PERSONA / "recall.yaml").read_text(encoding="utf-8"))
    assert len(entries) == 25
    for entry in entries:
        assert entry["question"].strip()
        assert len(entry["expected"]) >= 1
        assert all(str(fact).strip() for fact in entry["expected"])
        assert entry["sources"]
        for source in entry["sources"]:
            assert (PERSONA / source).is_file(), source


def test_interview_transcript_format():
    text = (PERSONA / "interviews" / "2025-06-01-childhood.md").read_text(encoding="utf-8")
    front = yaml.safe_load(text.split("---\n")[1])
    assert front["kind"] == "interview"
    assert front["mode"] == "interview"
    turns = re.findall(r"^\*\*(me|mirror):\*\* ", text, flags=re.M)
    assert turns[0] == "mirror"
    assert turns.count("me") >= 10
    assert set(turns) == {"me", "mirror"}


@pytest.mark.parametrize("rel", list(CHATS))
def test_chat_sizes(rel):
    header = re.compile(CHATS[rel][0])
    lines = (WHATSAPP / rel).read_text(encoding="utf-8").splitlines()
    count = sum(1 for line in lines if header.match(line))
    assert 250 <= count <= 450


@pytest.mark.parametrize("rel", list(CHATS))
def test_chat_export_features(rel):
    raw = (WHATSAPP / rel).read_text(encoding="utf-8")
    messages = parse_chat(rel)
    assert any(not who or "encrypted" in text or "chiffrés" in text for who, text in messages)
    assert re.search(r"omitted|absente|omis|<Médias omis>|<Media omitted>", raw)
    assert re.search(r"(This message was|You) deleted|supprimé ce message|a été supprimé", raw)
    assert EDITED.search(raw)
    assert any("\n" in text for _, text in messages), "no multi-line message"


@pytest.mark.parametrize("rel", list(CHATS))
def test_chat_timestamps_increase(rel):
    header = re.compile(CHATS[rel][0])
    lines = [ln.lstrip("‎") for ln in (WHATSAPP / rel).read_text(encoding="utf-8").splitlines()]
    stamps = []
    for line in lines:
        if not header.match(line):
            continue
        if line.startswith("["):
            d, m, y, hh, mm, ss = map(int, re.findall(r"\d+", line[: line.index("]")]))
            stamps.append((y, m, d, hh, mm, ss))
        elif " " in line:
            mo, d, y, hh, mm = map(int, re.findall(r"\d+", line[: line.index(" ")]))
            hh = hh % 12 + (12 if " PM" in line else 0)
            stamps.append((y, mo, d, hh, mm))
        else:
            d, m, y, hh, mm = map(int, re.findall(r"\d+", line[:16]))
            stamps.append((y, m, d, hh, mm))
    assert stamps == sorted(stamps)
    years = {s[0] % 100 for s in stamps}
    assert max(years) == 25


def test_camille_french_chat_style():
    """FACTS.md: French chat is lowercase, never ends on a full stop, laughs 'ahah'."""
    for rel in ("Léa Fontaine/_chat.txt", "Discussion WhatsApp avec Maman.txt"):
        me = text_messages(rel, author=CHATS[rel][1])
        others = text_messages(rel, exclude=CHATS[rel][1])
        assert lowercase_start_rate(me) >= 0.95
        assert lowercase_start_rate(others) <= 0.1
        assert not any(t.endswith(".") and not t.endswith("..") for t in me)
        joined = " ".join(me).lower()
        assert "ahah" in joined
        assert not re.search(r"\b(mdr|ptdr|haha)", joined)
    lea_me = text_messages("Léa Fontaine/_chat.txt", author="Cam")
    lea = text_messages("Léa Fontaine/_chat.txt", author="Léa")
    median = sorted(len(t.split()) for t in lea_me)[len(lea_me) // 2]
    assert median <= 6
    emoji = regex.compile(r"\p{Extended_Pictographic}")
    mine = [e for t in lea_me for e in emoji.findall(t)]
    assert set(mine) <= {"🙃", "🫠"}
    assert len(mine) / len(lea_me) < len(emoji.findall(" ".join(lea))) / len(lea)
    joined = " ".join(lea_me)
    for phrase in ("en vrai", "bon bref", "ça me va", "t'inquiète"):
        assert phrase in joined, phrase


def test_camille_english_chat_style():
    """FACTS.md: English chat is capitalised, says 'haha', and keeps the space before '?'."""
    for rel in ("Copenhagen crew/_chat.txt", "WhatsApp Chat with Sam Whitaker.txt"):
        me = text_messages(rel, author=CHATS[rel][1])
        assert lowercase_start_rate(me) <= 0.1
        assert sum(t.endswith(".") for t in me) / len(me) < 0.1
        joined = " ".join(me)
        assert "haha" in joined.lower()
        assert " ?" in joined
        assert not re.search(r"\w\?", joined)
        others = " ".join(text_messages(rel, exclude=CHATS[rel][1]))
        assert " ?" not in others
    english = " ".join(
        " ".join(text_messages(rel, author=CHATS[rel][1]))
        for rel in ("Copenhagen crew/_chat.txt", "WhatsApp Chat with Sam Whitaker.txt")
    )
    for phrase in ("to be fair", "honestly", "fair enough", "I mean", ":)"):
        assert phrase in english.lower() or phrase in english, phrase


def test_other_participants_have_their_own_habits():
    crew = "Copenhagen crew/_chat.txt"
    jonas = text_messages(crew, author="Jonas Lund")
    assert sum(t.endswith(".") for t in jonas) / len(jonas) >= 0.9
    assert "jaja" in " ".join(text_messages(crew, author="Mateo Vidal")).lower()
    assert lowercase_start_rate(text_messages(crew, author="Priya Shah")) >= 0.8
    sam = " ".join(text_messages("WhatsApp Chat with Sam Whitaker.txt", author="Sam Whitaker"))
    assert "lol" in sam or "lmao" in sam
    maman = " ".join(text_messages("Discussion WhatsApp avec Maman.txt", author="Françoise Martin"))
    assert "Bisous, Maman" in maman and "Ha ha" in maman
    lea = " ".join(text_messages("Léa Fontaine/_chat.txt", author="Léa")).lower()
    assert "mdr" in lea
