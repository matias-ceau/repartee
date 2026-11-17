"""Exported conversations with AI assistants (Claude.ai and ChatGPT `conversations.json`).

The owner's turns are the owner's items. The assistant's turns are kept as items too,
so that a chunk reads as a conversation, but they are never the owner's: `is_self` is
false, extraction may not cite them (Kind.ASSISTANT is owner-only evidence), chunks
carry only a clipped version of them, and voice never sees them.

Accepted inputs: a `.zip` export, an extracted export folder, or a bare `.json` file.
Every `.json` found is read; files that hold no conversations (`users.json`,
`projects.json`, ...) are counted as skipped. Unknown fields are ignored; a
conversation or message that cannot be read is counted and sampled in the report.

Ids are keyed by the conversation id and the message id (or, when a message has no
id, by a hash of its role and text), never by file path, so the same conversation in
a zip and in its extracted folder, or in an older and a newer export, has the same ids.

Owner turns are often pasted material (code, logs, articles). The item text stays
complete for search; :func:`pasted_reason` flags turns that are mostly pasted
(`meta["pasted"]`), and turns with fenced code get `meta["code"]`, so that voice can
skip the first and strip code from the second (see voice/registers.py).
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from repartee.config import OwnerConfig, SourceConfig
from repartee.ingest.base import (
    MAX_SAMPLES_PER_FILE,
    ParseReport,
    decode_bytes,
    relative,
    to_datetime,
    walk_files,
)
from repartee.ingest.lang import detect_language
from repartee.schema import ID_PREFIX, Item, Kind, make_item_id, stable_hash

SOURCE = "assistant"
#: Display name of the other speaker, per export format.
ASSISTANT_NAMES = {"claude": "Claude", "chatgpt": "ChatGPT"}

# ------------------------------------------------------------------ pasted text

#: A fenced code block (``` or ~~~, three or more); an unclosed fence runs to the end.
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})[^\n]*\n?.*?(?:^[ \t]*\1[ \t]*$|\Z)",
                    re.MULTILINE | re.DOTALL)
_URL = re.compile(r"https?://\S+|www\.\S+")
_WORD = re.compile(r"[^\W\d_]{2,}")
_LOG_LINE = re.compile(r"^\s*(?:\[?\d{4}-\d{2}-\d{2}[ T]\d|\d{2}:\d{2}:\d{2}\b|Traceback \(|"
                       r"File \"|at [\w.$]+\(|\$ |>>> |[A-Z]\w*(?:Error|Exception)\b)")
_CODE_CHARS = set("{}[]()<>;=_/\\|$#*`@^~")

#: A line is code-like when this share of its visible characters are code symbols.
CODE_SYMBOL_SHARE = 0.15
#: Under this many visible characters a turn is never judged "mostly pasted" by share.
PASTE_MIN_CHARS = 300
#: A turn is mostly pasted when the owner's own lines are under this share of it.
OWN_SHARE_MIN = 0.5
#: Fewer of the owner's own words than this, next to pasted material: nothing to learn from.
OWN_WORDS_MIN = 4
#: Own-looking prose longer than this is taken for a pasted document (about 1,000 words).
LONG_PROSE_CHARS = 6000


def strip_code_blocks(text: str, placeholder: str = "") -> str:
    """The text without fenced code blocks (each replaced by `placeholder`), tidied."""
    out = _FENCE.sub(lambda _: placeholder, text)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _visible(text: str) -> int:
    return len("".join(text.split()))


def _pasted_line(line: str) -> bool:
    """A line that looks copied rather than typed: quoted, indented code, a log, a link, code."""
    if line.lstrip().startswith(">") or line.startswith(("    ", "\t")):
        return True
    if _LOG_LINE.match(line):
        return True
    visible = "".join(line.split())
    if not visible:
        return False
    if sum(len(u) for u in _URL.findall(line)) * 2 > len(visible):
        return True
    return sum(1 for ch in visible if ch in _CODE_CHARS) > CODE_SYMBOL_SHARE * len(visible)


def pasted_reason(text: str) -> str | None:
    """Why an owner turn is mostly pasted material, or None when it reads as their own words.

    The heuristic, in order:

    1. Fenced code blocks are pasted. In what is left, a line is pasted when it is
       quoted (`>`), indented by four spaces or a tab, shaped like a log or traceback
       line, mostly a link, or dense in code symbols. Every other line is the owner's.
    2. "pasted": the owner's lines hold fewer than 4 words while something was pasted.
    3. "mostly pasted": over 300 visible characters, and the owner's lines are under
       half of them.
    4. "very long": the owner's lines are over 6,000 characters (about 1,000 words);
       a turn that long is almost always a pasted article or document.
    """
    total = _visible(text)
    if total == 0:
        return None
    prose = strip_code_blocks(text)
    own_lines = [ln for ln in prose.split("\n") if ln.strip() and not _pasted_line(ln)]
    own_text = "\n".join(own_lines)
    own = _visible(own_text)
    if len(_WORD.findall(_URL.sub(" ", own_text))) < OWN_WORDS_MIN and own < total:
        return "pasted"
    if total >= PASTE_MIN_CHARS and own < OWN_SHARE_MIN * total:
        return "mostly pasted"
    if len(own_text) > LONG_PROSE_CHARS:
        return "very long"
    return None


# ----------------------------------------------------------------------- parsing


class _Turn:
    __slots__ = ("mine", "text", "msg_id", "when", "extra")

    def __init__(self, mine: bool, text: str, msg_id: str | None, when: datetime | None,
                 extra: dict[str, Any] | None = None) -> None:
        self.mine = mine
        self.text = text
        self.msg_id = msg_id
        self.when = when
        self.extra = extra or {}


def _epoch(value: Any) -> datetime | None:
    """Naive UTC datetime from epoch seconds, else from an ISO string."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC).replace(tzinfo=None)
        except (OverflowError, OSError, ValueError):
            return None
    return _iso(value)


def _iso(value: Any) -> datetime | None:
    """Naive UTC datetime from an ISO string (exports write UTC)."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return to_datetime(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


class _Malformed(Exception):
    """A conversation or message whose shape is not one the parser knows."""


def _claude_text(message: dict[str, Any]) -> str:
    """Text blocks of `content`, joined; the flat `text` field when there are none."""
    content = message.get("content")
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
    if parts:
        return "\n\n".join(parts)
    text = message.get("text")
    return text.strip() if isinstance(text, str) else ""


def _claude_order(messages: list[Any], leaf: Any) -> list[Any]:
    """The visible branch when the export records parents and a current leaf, else list order."""
    by_id = {m.get("uuid"): m for m in messages if isinstance(m, dict) and m.get("uuid")}
    if not isinstance(leaf, str) or leaf not in by_id:
        return messages
    if not any(isinstance(m, dict) and m.get("parent_message_uuid") for m in messages):
        return messages
    path: list[Any] = []
    seen: set[str] = set()
    node: Any = by_id[leaf]
    while isinstance(node, dict) and node.get("uuid") not in seen:
        seen.add(node.get("uuid"))
        path.append(node)
        node = by_id.get(node.get("parent_message_uuid"))
    return list(reversed(path))


def _claude_turns(conv: dict[str, Any], report: ParseReport) -> list[_Turn]:
    messages = conv.get("chat_messages")
    if not isinstance(messages, list):
        raise _Malformed("chat_messages is not a list")
    turns: list[_Turn] = []
    for message in _claude_order(messages, conv.get("current_leaf_message_uuid")):
        if not isinstance(message, dict):
            report.skip("malformed_message")
            continue
        sender = message.get("sender")
        if sender not in ("human", "assistant"):
            report.skip("other_sender")
            continue
        text = _claude_text(message)
        if not text:
            report.skip("empty")
            continue
        extra: dict[str, Any] = {}
        attached = sum(len(v) for v in (message.get("attachments"), message.get("files"))
                       if isinstance(v, list))
        if attached:
            extra["attachments"] = attached  # pasted files: not the owner's words, not kept
        turns.append(_Turn(sender == "human", text, _str(message.get("uuid")),
                           _iso(message.get("created_at")), extra))
    return turns


#: ChatGPT content types whose parts are what a person or the model wrote.
_GPT_TEXT_TYPES = {"text", "multimodal_text"}


def _gpt_path(mapping: dict[str, Any], current: Any) -> list[dict[str, Any]]:
    """Nodes from the root to the current node (the thread the person last saw)."""
    if not isinstance(current, str) or current not in mapping:
        leaves = [k for k, n in mapping.items()
                  if isinstance(n, dict) and not n.get("children")]
        if not leaves:
            raise _Malformed("no current_node and no leaf")

        def stamp(key: str) -> float:
            msg = mapping[key].get("message")
            t = msg.get("create_time") if isinstance(msg, dict) else None
            return float(t) if isinstance(t, int | float) else float("-inf")

        current = max(leaves, key=stamp)  # the latest leaf; ties go to the last listed
    path: list[dict[str, Any]] = []
    seen: set[str] = set()
    key: Any = current
    while isinstance(key, str) and key in mapping and key not in seen:
        seen.add(key)
        node = mapping[key]
        if not isinstance(node, dict):
            raise _Malformed(f"node {key!r} is not an object")  # the thread is broken
        path.append(node | {"_key": key})
        key = node.get("parent")
    return list(reversed(path))


def _gpt_turns(conv: dict[str, Any], report: ParseReport) -> list[_Turn]:
    mapping = conv.get("mapping")
    if not isinstance(mapping, dict):
        raise _Malformed("mapping is not an object")
    turns: list[_Turn] = []
    for node in _gpt_path(mapping, conv.get("current_node")):
        message = node.get("message")
        if message is None:
            continue  # the root
        if not isinstance(message, dict):
            report.skip("malformed_message")
            continue
        author = message.get("author")
        role = author.get("role") if isinstance(author, dict) else None
        if role not in ("user", "assistant"):
            report.skip(str(role or "unknown_role"))
            continue
        meta = message.get("metadata")
        if isinstance(meta, dict) and meta.get("is_visually_hidden_from_conversation"):
            report.skip("hidden")
            continue
        recipient = message.get("recipient")
        if role == "assistant" and isinstance(recipient, str) and recipient != "all":
            report.skip("tool")  # a call to a tool, not a reply
            continue
        content = message.get("content")
        ctype = content.get("content_type") if isinstance(content, dict) else None
        if ctype not in _GPT_TEXT_TYPES:
            report.skip("non_text")
            continue
        parts = content.get("parts")
        texts = [p.strip() for p in parts if isinstance(p, str) and p.strip()] if isinstance(
            parts, list) else []
        if not texts:
            report.skip("media" if parts else "empty")
            continue
        turns.append(_Turn(role == "user", "\n\n".join(texts),
                           _str(message.get("id")) or _str(node.get("_key")),
                           _epoch(message.get("create_time"))))
    return turns


def _conversations(data: Any) -> list[Any] | None:
    """The list of conversations in a decoded file, or None when it holds none."""
    if isinstance(data, dict):
        data = data.get("conversations")
    if not isinstance(data, list):
        return None
    if not any(isinstance(c, dict) and ("chat_messages" in c or "mapping" in c) for c in data):
        return None
    return data


def parse_conversations(
    data: Any,
    owner: OwnerConfig,
    report: ParseReport,
    *,
    label: str,
    meta_path: str = "",
) -> list[Item] | None:
    """Items from a decoded export; None when the data holds no conversations at all."""
    convs = _conversations(data)
    if convs is None:
        return None
    prefix = ID_PREFIX[SOURCE]
    items: list[Item] = []
    samples = 0
    for index, conv in enumerate(convs):
        conv_id = None
        try:
            if not isinstance(conv, dict):
                raise _Malformed("not an object")
            fmt = "claude" if "chat_messages" in conv else "chatgpt" if "mapping" in conv else None
            if fmt is None:
                raise _Malformed("neither chat_messages nor mapping")
            conv_id = (_str(conv.get("uuid")) or _str(conv.get("conversation_id"))
                       or _str(conv.get("id")))
            title = _str(conv.get("name")) or _str(conv.get("title"))
            started = (_iso(conv.get("created_at")) if fmt == "claude"
                       else _epoch(conv.get("create_time")))
            turns = (_claude_turns if fmt == "claude" else _gpt_turns)(conv, report)
        except _Malformed as exc:
            report.skip("malformed_conversation")
            if samples < MAX_SAMPLES_PER_FILE:
                which = f"conversation {index + 1}" + (f" ({conv_id})" if conv_id else "")
                report.warn(f"{label}: {which} skipped: {exc}")
                samples += 1
            continue
        if not turns:
            report.skip("empty_conversation")
            continue
        if conv_id is None:
            conv_id = "h-" + stable_hash(title, started, turns[0].text)
        container = f"{title} ({conv_id[:8]})" if title else conv_id
        other = ASSISTANT_NAMES[fmt]
        ordinals: dict[tuple[bool, str], int] = {}
        for seq, turn in enumerate(turns):
            msg_key = turn.msg_id
            if msg_key is None:
                n = ordinals.get((turn.mine, turn.text), 0)
                ordinals[(turn.mine, turn.text)] = n + 1
                msg_key = "h-" + stable_hash(turn.mine, turn.text, n)
            meta: dict[str, Any] = {"role": "me" if turn.mine else "assistant",
                                    "conversation": conv_id, "format": fmt, **turn.extra}
            if meta_path:
                meta["path"] = meta_path
            lang = None
            if turn.mine:
                prose = strip_code_blocks(turn.text)
                if prose != turn.text.strip():
                    meta["code"] = True
                    lang = detect_language(prose) if prose else None
                reason = pasted_reason(turn.text)
                if reason:
                    meta["pasted"] = reason
            items.append(Item(
                id=make_item_id(prefix, conv_id, msg_key),
                source=SOURCE,
                kind=Kind.ASSISTANT,
                container=container,
                author=owner.name if turn.mine else other,
                is_self=turn.mine,
                timestamp=turn.when or started,
                text=turn.text,
                title=title,
                lang=lang,
                seq=seq,
                meta=meta,
            ))
    return items


class AssistantAdapter:
    """Claude.ai and ChatGPT exports: `conversations.json`, loose, in a folder, or zipped."""

    name = SOURCE

    def discover(self, root: Path, source: SourceConfig) -> list[Path]:
        """Every .json and .zip under root, minus excluded globs."""
        return walk_files(root, {".json", ".zip"}, source.exclude, source.include)

    def parse(
        self,
        path: Path,
        root: Path,
        source: SourceConfig,
        owner: OwnerConfig,
        report: ParseReport,
    ) -> Iterator[Item]:
        """Parse one export file (or every .json in a zip) into turn items."""
        rel = relative(path, root)
        if path.suffix.lower() == ".zip":
            yield from self._parse_zip(path, rel, owner, report)
            return
        yield from self._parse_bytes(path.read_bytes(), rel, rel, owner, report)

    def _parse_bytes(self, data: bytes, label: str, meta_path: str, owner: OwnerConfig,
                     report: ParseReport) -> list[Item]:
        name = Path(label.rsplit("!", 1)[-1]).name.lower()
        try:
            decoded = json.loads(decode_bytes(data, label, report))
        except json.JSONDecodeError as exc:
            report.warn(f"{label}: not valid JSON ({exc.msg}, line {exc.lineno})")
            return []
        items = parse_conversations(decoded, owner, report, label=label, meta_path=meta_path)
        if items is None:
            if name.startswith("conversations"):
                report.warn(f"{label}: holds no conversations in a known format")
            else:
                report.skip("other_file")  # users.json, projects.json and the like
            return []
        report.items += len(items)
        return items

    def _parse_zip(self, path: Path, rel: str, owner: OwnerConfig,
                   report: ParseReport) -> Iterator[Item]:
        try:
            with zipfile.ZipFile(path) as zf:
                members = [n for n in zf.namelist()
                           if n.lower().endswith(".json") and not n.startswith("__MACOSX")]
                found = False
                for member in members:
                    items = self._parse_bytes(zf.read(member), f"{rel}!{member}", rel, owner,
                                              report)
                    found = found or bool(items)
                    yield from items
                if not found:
                    report.warn(f"{rel}: zip holds no assistant conversations")
        except zipfile.BadZipFile:
            report.warn(f"{rel}: not a readable zip file")
