"""WhatsApp chat exports (`_chat.txt`, loose, in a folder or in a zip).

Every format decision is a table below, so a real export in an unseen variant
needs a table entry, not new code. Lines the parser cannot place are counted
in the report with a few samples; nothing is guessed silently.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from repartee.config import OwnerConfig, SourceConfig
from repartee.ingest.base import (
    MAX_SAMPLES_PER_FILE,
    ParseReport,
    decode_bytes,
    relative,
    walk_files,
)
from repartee.schema import Item, Kind, make_item_id

# Invisible direction and formatting marks that exports scatter around.
MARKS = "‎‏‪‫‬‭‮⁦⁧⁨⁩﻿​"
_SPACES = " \t  "

# ------------------------------------------------------------------ line headers

_DATE = r"(?P<date>\d{1,4}\s?[./-]\s?\d{1,2}\s?[./-]\s?\d{1,4})\.?"
# Words some locales put between date and time ("12/03/2021 à 14:05").
_CONNECT = r"(?:,[   ]*|[   ]+)(?:(?:à|a las|um|às|alle|om|kl\.?|klo)[   ]+)?"
_AMPM_WORDS = r"[AaPp]\.?[   ]?[Mm]\.?|vorm\.|nachm\.|午前|午後|오전|오후|ص|م"
_TIME = (
    rf"(?:(?P<ampm1>午前|午後|오전|오후)[   ]?)?"
    r"(?P<time>\d{1,2}[:.]\d{2}(?:[:.]\d{2})?)"
    rf"(?:[   ]*(?P<ampm>{_AMPM_WORDS}))?"
)
#: iOS: "[12/03/2021, 14:05:33] Ana: text"
_BRACKET = re.compile(rf"\[{_DATE}{_CONNECT}{_TIME}\][   ]?(?P<rest>.*)")
#: Android: "12/03/2021, 14:05 - Ana: text"
_DASH = re.compile(rf"{_DATE}{_CONNECT}{_TIME}[   ]+[-–—][   ]?(?P<rest>.*)")

_AM = {"am", "a.m.", "vorm.", "午前", "오전", "ص"}
_PM = {"pm", "p.m.", "nachm.", "午後", "오후", "م"}

_AUTHOR = re.compile(r"(?P<author>[^:\n]{1,100}?)[   ]?:(?:[   ]|$)(?P<text>.*)")

DATE_ORDERS = ("dmy", "mdy", "ymd")

# ------------------------------------------------------------------------ tables
#
# Each table is a list of (language, regex). Matching is case-insensitive and
# against the whole text (after invisible marks are stripped). Extend freely.

#: An actor at the start of a system line: a display name, never containing ":".
_A = r"[^:\n]{1,80}?"

#: Sentences no person types; recognised wherever they appear.
SYSTEM_FIXED: list[tuple[str, str]] = [
    ("*", r"(?:\w+ )?(?:messages|les messages|los mensajes|nachrichten|i messaggi|as mensagens"
          r"|berichten)\b[^\n]{0,80}(?:end-to-end|bout en bout|extremo a extremo|ende-zu-ende"
          r"|ponta a ponta)\b.*"),
    ("en", r"(?:your )?security code with .+ changed.*"),
    ("en", rf"{_A}'s security code changed.*"),
    ("en", r"this (?:chat|business) (?:is with|uses) .*"),
    ("en", r"you (?:blocked|unblocked) this contact.*"),
    ("en", r"waiting for this message.*"),
    ("fr", r"(?:le|votre) code de sécurité (?:de|avec) .+ a changé.*"),
    ("fr", r"vous avez (?:bloqué|débloqué) ce contact.*"),
    ("fr", r"en attente de ce message.*"),
    ("es", r"(?:el|tu) código de seguridad (?:con|de) .+ cambió.*"),
    ("es", r"(?:bloqueaste|desbloqueaste) a este contacto.*"),
    ("es", r"esperando (?:este|el) mensaje.*"),
    ("de", rf"{_A}sicherheitsnummer[^:\n]{{0,80}}geändert.*"),
    ("de", r"du hast diesen kontakt (?:blockiert|freigegeben).*"),
    ("de", r"warte(?:n)? auf diese nachricht.*"),
    ("it", r"il (?:tuo )?codice di sicurezza .+ è cambiato.*"),
    ("it", r"hai (?:bloccato|sbloccato) questo contatto.*"),
    ("it", r"in attesa di questo messaggio.*"),
    ("pt", r"(?:o|seu) código de segurança .+ (?:mudou|foi alterado).*"),
    ("pt", r"você (?:bloqueou|desbloqueou) (?:este|esse) contato.*"),
    ("pt", r"aguardando (?:esta|essa) mensagem.*"),
    ("nl", r"je beveiligingscode .+ is gewijzigd.*"),
]

#: Group events and calls. Recognised on lines without an author, and on authored
#: lines only when marked as system text (leading direction mark, or the author is
#: the chat itself), so "Ana: Bob left" stays a message.
SYSTEM_EVENTS: list[tuple[str, str]] = [
    ("en", rf"{_A} created (?:the |this )?group.*"),
    ("en", rf"{_A} (?:added|removed) .+"),
    ("en", rf"{_A} (?:left|joined)"),
    ("en", rf"{_A} joined (?:using|via|from) .*"),
    ("en", rf"{_A} (?:changed|deleted|reset) (?:the|this|their|your) .*"),
    ("en", rf"{_A} changed to .+"),
    ("en", rf"{_A}(?:'re| is| are) now an admin.*"),
    ("en", rf"{_A} (?:were|was) (?:added|removed).*"),
    ("en", rf"{_A} (?:pinned|unpinned) a message.*"),
    ("en", rf"{_A} turned (?:on|off) disappearing messages.*"),
    ("en", rf"{_A} (?:started|ended) a (?:voice |video )?call.*"),
    ("en", r"(?:missed )?(?:group )?(?:voice|video) call\b.*"),
    ("en", r"disappearing messages .*"),
    ("fr", rf"{_A} (?:a|avez) créé (?:le |ce )?groupe.*"),
    ("fr", rf"{_A} (?:a|avez) (?:ajouté|retiré|supprimé|exclu|changé|modifié|réinitialisé) .+"),
    ("fr", rf"{_A} (?:vous|t'|t’) ?a (?:ajouté|retiré)e?"),
    ("fr", rf"{_A} (?:a|avez|ont) été (?:ajouté|retiré)e?s?.*"),
    ("fr", rf"{_A} (?:est parti|est partie|a quitté|avez quitté|êtes parti|êtes partie).*"),
    ("fr", rf"{_A} (?:a|avez|êtes|est) rejoint.*"),
    ("fr", rf"{_A} (?:est|êtes) (?:maintenant|désormais) (?:un |une )?admin.*"),
    ("fr", rf"{_A} (?:a|avez) (?:épinglé|désépinglé) un message.*"),
    ("fr", rf"{_A} (?:a|avez) (?:activé|désactivé) les messages éphémères.*"),
    ("fr", r"appel (?:vocal|vidéo)(?: de groupe)?(?: manqué| en absence)?\b.*"),
    ("es", rf"{_A} (?:creó|ha creado|has creado) (?:el |este )?grupo.*"),
    ("es", r"(?:creaste|añadiste|agregaste|eliminaste|saliste|cambiaste|te uniste)\b.*"),
    ("es", rf"{_A} (?:añadió|agregó|eliminó|quitó|cambió) .+"),
    ("es", rf"{_A} te (?:añadió|agregó|eliminó)"),
    ("es", rf"{_A} salió(?: del grupo)?"),
    ("es", rf"{_A} se unió.*"),
    ("es", rf"{_A} (?:ahora es|ahora eres|es ahora) (?:un )?admin.*"),
    ("es", rf"{_A} (?:fijó|fijaste) un mensaje.*"),
    ("es", rf"{_A} (?:activó|desactivó) los mensajes temporales.*"),
    ("es", r"llamada (?:de voz|de video|de vídeo)(?: perdida)?\b.*"),
    ("de", rf"{_A} (?:hat|hast) (?:die )?gruppe .*erstellt.*"),
    ("de", rf"{_A} (?:hat|hast) .+ (?:hinzugefügt|entfernt|geändert|gelöscht|angeheftet)"),
    ("de", rf"{_A} (?:hat|hast) die gruppe verlassen"),
    ("de", rf"{_A} (?:ist|bist) .*beigetreten"),
    ("de", rf"{_A} (?:ist|bist) jetzt (?:ein )?admin.*"),
    ("de", rf"{_A} (?:wurde|wurdest) (?:hinzugefügt|entfernt).*"),
    ("de", r"(?:verpasster )?(?:sprach|video|gruppen)anruf\b.*"),
    ("it", rf"{_A} (?:ha|hai) creato (?:il |questo )?gruppo.*"),
    ("it", r"hai (?:aggiunto|rimosso|creato|cambiato|modificato|abbandonato) .+"),
    ("it", rf"{_A} (?:ha|hai) (?:aggiunto|rimosso|cambiato|modificato|fissato) .+"),
    ("it", rf"{_A} ti ha (?:aggiunto|rimosso)"),
    ("it", rf"{_A} (?:è uscito|è uscita|sei uscito|sei uscita|ha abbandonato).*"),
    ("it", rf"{_A} si è unit[oa].*"),
    ("it", rf"{_A} (?:è|sei) ora (?:un )?amministratore.*"),
    ("it", r"chiamata (?:vocale|video)(?: persa)?\b.*"),
    ("pt", rf"{_A} criou (?:o |este )?grupo.*"),
    ("pt", rf"{_A} (?:adicionou|removeu|mudou|alterou|apagou|fixou) .+"),
    ("pt", rf"{_A} (?:te )?adicionou(?: você)?"),
    ("pt", rf"{_A} saiu(?: do grupo)?"),
    ("pt", rf"{_A} entrou\b.*"),
    ("pt", rf"{_A} (?:agora é|é agora) (?:um )?admin.*"),
    ("pt", r"(?:chamada|ligação) (?:de voz|de vídeo)(?: perdida)?\b.*"),
    ("nl", rf"{_A} (?:heeft|hebt) (?:de )?groep .*aangemaakt"),
    ("nl", rf"{_A} (?:heeft|hebt) .+ (?:toegevoegd|verwijderd|gewijzigd)"),
    ("nl", rf"{_A} (?:heeft|hebt) de groep verlaten"),
    ("nl", rf"{_A} is .*(?:toegetreden|lid geworden)"),
    ("nl", rf"{_A} is nu (?:een )?beheerder"),
]

_MEDIA_NOUN = (
    r"image|photo|video|vidéo|vídeo|audio|sticker|autocollant|gif|document|documento|"
    r"contact card|carte de contact|fiche contact|imagen|imagem|figurinha|bild|foto|"
    r"immagine|kontaktkarte|scheda contatto|tarjeta de contacto|cartão de contato|afbeelding|"
    r"pegatina|voice message|message vocal|mensaje de voz|sprachnachricht|messaggio vocale|"
    r"mensagem de voz|spraakbericht"
)
_MEDIA_ADJ = (
    r"omitted|omis|omise|absente?|manquante?|retirée?|omitid[oa]|omessa|omesso|"
    r"weggelassen|ausgelassen|ocultad[oa]|ocult[oa]|weggelaten"
)
#: Media placeholders. Matched against the first line; a caption after it is kept.
MEDIA: list[tuple[str, str]] = [
    ("*", r"<(?:media|médias?|medias|multimedia|mídia|medien|archivo|arquivo|ficheiro)"
          r"[^<>]{0,40}>"),
    ("*", rf"(?:{_MEDIA_NOUN})\s+(?:{_MEDIA_ADJ})"),
    ("*", r"<(?:attached|pièce jointe|piece jointe|adjunto|adjunta|anhang|allegato|anexo"
          r"|bijlage)\s*:\s*[^<>]+>"),
    ("*", r".+\.\w{1,5}\s*\((?:file attached|fichier joint|archivo adjunto|datei angehängt"
          r"|file allegato|arquivo anexado|ficheiro anexado|bestand bijgevoegd)\)"),
    ("*", r"(?:location|localisation|position|ubicación|standort|posizione|localização"
          r"|locatie)\s*:\s*https?://\S+"),
    ("*", r"(?:live location shared|position en direct partagée"
          r"|ubicación en tiempo real compartida|live-standort geteilt)"),
    ("*", r"null"),
]

#: Deleted-message placeholders (the message produces no item).
DELETED: list[tuple[str, str]] = [
    ("en", r"(?:this message was deleted(?: by .+)?|you deleted this message)"),
    ("fr", r"(?:ce message a été supprimé|vous avez supprimé ce message|message supprimé)"),
    ("es", r"(?:se eliminó este mensaje|eliminaste este mensaje|este mensaje fue eliminado"
           r"|mensaje eliminado)"),
    ("de", r"(?:diese nachricht wurde gelöscht|du hast diese nachricht gelöscht"
           r"|nachricht gelöscht)"),
    ("it", r"(?:questo messaggio è stato eliminato|hai eliminato questo messaggio"
           r"|messaggio eliminato)"),
    ("pt", r"(?:esta mensagem foi apagada|esta mensagem foi eliminada|você apagou esta mensagem"
           r"|apagaste esta mensagem|mensagem apagada)"),
    ("nl", r"(?:dit bericht is verwijderd|je hebt dit bericht verwijderd)"),
]

#: Edited markers appended to a message (stripped; meta["edited"] = True).
EDITED: list[tuple[str, str]] = [
    ("en", r"this message was edited"),
    ("fr", r"ce message a été modifié|message modifié"),
    ("es", r"se editó este mensaje|este mensaje (?:fue|ha sido) editado|mensaje editado"),
    ("de", r"diese nachricht wurde bearbeitet|nachricht bearbeitet"),
    ("it", r"questo messaggio è stato modificato|messaggio modificato"),
    ("pt", r"esta mensagem foi editada|mensagem editada"),
    ("nl", r"dit bericht is bewerkt|bericht bewerkt"),
]

#: File-name prefixes WhatsApp puts before the chat name.
CHAT_PREFIXES = [
    "WhatsApp Chat with ",
    "WhatsApp Chat - ",
    "WhatsApp Chat mit ",
    "WhatsApp-Chat mit ",
    "Discussion WhatsApp avec ",
    "Chat de WhatsApp con ",
    "Chat WhatsApp con ",
    "Chat del WhatsApp con ",
    "Conversa do WhatsApp com ",
    "WhatsApp-chat met ",
]


def _table(entries: list[tuple[str, str]]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{rx})" for _, rx in entries), re.IGNORECASE | re.DOTALL)


_SYSTEM_FIXED = _table(SYSTEM_FIXED)
_SYSTEM_EVENTS = _table(SYSTEM_EVENTS)
_MEDIA = _table(MEDIA)
_DELETED = re.compile(rf"(?:{_table(DELETED).pattern})\.?", re.IGNORECASE)
_EDITED = re.compile(
    rf"[{_SPACES}{MARKS}]*<\s*(?:{_table(EDITED).pattern})\.?\s*>[{_SPACES}{MARKS}]*\Z",
    re.IGNORECASE,
)


def _strip_marks(text: str) -> str:
    return text.strip(MARKS + _SPACES)


def chat_name_for(file_name: str) -> str:
    """Chat name from an export file name: extension and WhatsApp prefixes removed."""
    stem = Path(file_name).stem if Path(file_name).suffix.lower() in {".txt", ".zip"} else file_name
    stem = _strip_marks(stem)
    for prefix in CHAT_PREFIXES:
        if stem.casefold().startswith(prefix.casefold()):
            return stem[len(prefix) :].strip() or stem
    return stem


# --------------------------------------------------------------------- the parse


@dataclass
class _Header:
    line: int
    fields: tuple[int, int, int]
    first_len: int
    time: str
    ampm: str | None
    rest: str


@dataclass
class _Message:
    header: _Header
    when: datetime
    lines: list[str] = field(default_factory=list)


def _match_header(line: str) -> _Header | None:
    stripped = line.lstrip(MARKS + _SPACES)
    if not stripped or not (stripped[0] == "[" or stripped[0].isdigit()):
        return None
    m = _BRACKET.match(stripped) if stripped[0] == "[" else _DASH.match(stripped)
    if not m:
        return None
    parts = re.split(r"\s?[./-]\s?", m["date"])
    try:
        fields = (int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return None
    ampm = m["ampm"] or m["ampm1"]
    return _Header(0, fields, len(parts[0]), m["time"], ampm, m["rest"])


def _year(value: int, digits_hint: int = 4) -> int:
    return 2000 + value if value < 100 else value


def _make_date(h: _Header, order: str) -> date | None:
    a, b, c = h.fields
    if h.first_len >= 3:
        order = "ymd"
    if order == "dmy":
        d, mo, y = a, b, c
    elif order == "mdy":
        mo, d, y = a, b, c
    else:
        y, mo, d = a, b, c
    if y >= 100 and y < 1990:
        return None
    try:
        return date(_year(y), mo, d)
    except ValueError:
        return None


def _make_time(h: _Header) -> tuple[int, int, int] | None:
    parts = [int(p) for p in re.split(r"[:.]", h.time)]
    hour, minute = parts[0], parts[1]
    second = parts[2] if len(parts) > 2 else 0
    if h.ampm:
        key = h.ampm.replace(" ", "").replace(" ", "").replace(" ", "").lower()
        key = {"a.m": "a.m.", "p.m": "p.m.", "am.": "am", "pm.": "pm"}.get(key, key)
        if key in _PM and 1 <= hour <= 11:
            hour += 12
        elif key in _AM and hour == 12:
            hour = 0
    if hour > 23 or minute > 59 or second > 59:
        return None
    return hour, minute, second


def _backwards(dates: list[date | None]) -> int:
    count, last = 0, None
    for d in dates:
        if d is None:
            continue
        if last is not None and d < last:
            count += 1
        last = d
    return count


def _choose_order(
    headers: list[_Header], forced: str | None, label: str, report: ParseReport
) -> str:
    if forced:
        return forced
    ambiguous = [h for h in headers if h.first_len < 3]
    if not ambiguous:
        return "dmy"
    n = len(ambiguous)
    dmy = [_make_date(h, "dmy") for h in ambiguous]
    mdy = [_make_date(h, "mdy") for h in ambiguous]
    ok_dmy = sum(d is not None for d in dmy)
    ok_mdy = sum(d is not None for d in mdy)
    if ok_dmy == n and ok_mdy < n:
        return "dmy"
    if ok_mdy == n and ok_dmy < n:
        return "mdy"
    if ok_dmy < n and ok_mdy < n:
        order = "mdy" if ok_mdy > ok_dmy else "dmy"
        report.warn(
            f"{label}: day-first and month-first dates both appear; read as {order}, "
            f"lines that do not fit are left unplaced (set options.date_order to force)"
        )
        return order
    # Both readings are valid everywhere.
    if all(h.fields[0] == h.fields[1] for h in ambiguous):
        return "dmy"
    back_dmy, back_mdy = _backwards(dmy), _backwards(mdy)
    if back_mdy < back_dmy:
        report.warn(
            f"{label}: day/month order ambiguous; read as month-first because dates are then "
            f"in chronological order (set options.date_order to force)"
        )
        return "mdy"
    how = "chronological order" if back_dmy < back_mdy else "no evidence either way"
    report.warn(
        f"{label}: day/month order ambiguous; read as day-first ({how}); "
        f"set options.date_order to 'mdy' if that is wrong"
    )
    return "dmy"


def _sample(text: str) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= 80 else text[:77] + "..."


def parse_chat(
    text: str,
    chat: str,
    owner: OwnerConfig,
    report: ParseReport,
    *,
    date_order: str | None = None,
    label: str = "",
    meta_path: str = "",
) -> list[Item]:
    """Parse the text of one export into message items, updating the report."""
    label = label or chat
    lines = text.split("\n")
    headers: list[_Header] = []
    for idx, line in enumerate(lines):
        h = _match_header(line)
        if h is not None:
            h.line = idx
            headers.append(h)
    order = _choose_order(headers, date_order, label, report)

    samples = 0

    def unplaced(lineno: int, line: str, why: str) -> None:
        nonlocal samples
        report.unparsed_lines += 1
        if samples < MAX_SAMPLES_PER_FILE:
            report.warn(f"{label} line {lineno + 1}: {why}: {_sample(line)!r}")
            samples += 1

    by_line = {h.line: h for h in headers}
    messages: list[_Message] = []
    current: _Message | None = None
    for idx, line in enumerate(lines):
        h = by_line.get(idx)
        if h is not None:
            d = _make_date(h, order)
            t = _make_time(h)
            if d is None or t is None:
                unplaced(idx, line, "date or time not understood")
                current = None
                continue
            current = _Message(h, datetime(d.year, d.month, d.day, *t), [h.rest])
            messages.append(current)
        elif current is not None:
            current.lines.append(line)
        elif _strip_marks(line):
            unplaced(idx, line, "not a message line")

    if len(messages) >= 20:
        back = _backwards([m.when.date() for m in messages])
        if back > max(2, len(messages) // 100):
            report.warn(
                f"{label}: dates go backwards {back} times; the date order may be misread "
                f"(set options.date_order)"
            )

    chat_key = _strip_marks(chat).casefold()
    items: list[Item] = []
    ordinals: dict[tuple[str, str, str], int] = {}
    for msg in messages:
        rest = _strip_marks(msg.lines[0])
        if not rest:
            unplaced(msg.header.line, lines[msg.header.line], "empty line header")
            continue
        if _SYSTEM_FIXED.fullmatch(rest) or _SYSTEM_EVENTS.fullmatch(rest):
            report.skip("system")
            continue
        m = _AUTHOR.fullmatch(rest)
        if not m:
            unplaced(msg.header.line, lines[msg.header.line], "line with no author (unknown "
                     "system wording?)")
            continue
        author = _strip_marks(m["author"]).lstrip("~").strip(MARKS + _SPACES)
        raw = "\n".join([m["text"], *msg.lines[1:]])
        marked = raw.lstrip(_SPACES)[:1] in MARKS if raw.strip(_SPACES) else False
        body = raw.strip(MARKS + _SPACES + "\n")
        meta: dict[str, object] = {"path": meta_path} if meta_path else {}

        edited = _EDITED.search(body)
        if edited:
            body = body[: edited.start()].rstrip(MARKS + _SPACES + "\n")
            meta["edited"] = True
        if _SYSTEM_FIXED.fullmatch(body) or (
            (marked or author.casefold() == chat_key) and _SYSTEM_EVENTS.fullmatch(body)
        ):
            report.skip("system")
            continue
        if _DELETED.fullmatch(body):
            report.skip("deleted")
            continue
        first, _, remainder = body.partition("\n")
        if _MEDIA.fullmatch(_strip_marks(first)):
            report.skip("media")
            body = remainder.strip(MARKS + _SPACES + "\n")
            if not body:
                continue
            meta["media"] = True
        elif marked:
            # iOS marks system and media lines this way; this one matched no table.
            report.skip("unrecognised_marked")
            if samples < MAX_SAMPLES_PER_FILE:
                report.warn(f"{label} line {msg.header.line + 1}: marked like a system line but "
                            f"matched no known wording; kept as text: {_sample(body)!r}")
                samples += 1
        if not body:
            report.skip("empty")
            continue

        stamp = msg.when.isoformat()
        key = (stamp, author, body)
        ordinal = ordinals.get(key, 0)
        ordinals[key] = ordinal + 1
        items.append(
            Item(
                id=make_item_id("wa", chat, stamp, author, body, ordinal),
                source="whatsapp",
                kind=Kind.MESSAGE,
                container=chat,
                author=author,
                is_self=owner.is_self(author),
                timestamp=msg.when,
                text=body,
                seq=len(items),
                meta=meta,
            )
        )
    report.items += len(items)
    return items


class WhatsAppAdapter:
    """WhatsApp `.txt` exports, loose, in export folders, or zipped."""

    name = "whatsapp"

    def discover(self, root: Path, source: SourceConfig) -> list[Path]:
        """Every .txt and .zip under root, minus excluded globs."""
        return walk_files(root, {".txt", ".zip"}, source.exclude)

    def parse(
        self,
        path: Path,
        root: Path,
        source: SourceConfig,
        owner: OwnerConfig,
        report: ParseReport,
    ) -> Iterator[Item]:
        """Parse one export file (or zip) into message items."""
        order = source.options.get("date_order")
        if order is not None and order not in DATE_ORDERS:
            report.warn(f"options.date_order {order!r} is not one of {DATE_ORDERS}; inferring")
            order = None
        rel = relative(path, root)
        if path.suffix.lower() == ".zip":
            yield from self._parse_zip(path, rel, owner, report, order)
            return
        name = chat_name_for(path.name)
        if Path(path.name).stem == "_chat":
            parent = path.parent if path != root else path.parent
            name = chat_name_for(parent.name)
        text = decode_bytes(path.read_bytes(), rel, report)
        yield from parse_chat(text, name, owner, report, date_order=order, label=rel,
                              meta_path=rel)

    def _parse_zip(self, path, rel, owner, report, order) -> Iterator[Item]:
        try:
            with zipfile.ZipFile(path) as zf:
                members = [
                    n for n in zf.namelist()
                    if n.lower().endswith(".txt") and not n.startswith("__MACOSX")
                ]
                if not members:
                    report.warn(f"{rel}: zip holds no .txt chat export")
                for member in members:
                    stem = Path(member).stem
                    name = chat_name_for(path.name) if stem == "_chat" else chat_name_for(member)
                    text = decode_bytes(zf.read(member), f"{rel}!{member}", report)
                    yield from parse_chat(text, name, owner, report, date_order=order,
                                          label=f"{rel}!{member}", meta_path=rel)
        except zipfile.BadZipFile:
            report.warn(f"{rel}: not a readable zip file")
