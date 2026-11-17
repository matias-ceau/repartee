"""Self-model pages, modelled as their original lines and edited line by line.

A page is never parsed to a tree and re-rendered: ``render()`` joins the lines it
was given, so a page that was not edited comes back byte for byte. Edits replace,
insert or remove single lines and touch nothing else.

Grammar (see spec section 10.2):

- A *claim line* is a line outside frontmatter, code fences, comments and the
  generated block that ends in `` ^c-<8 hex>``. When it is a top-level ``- `` or
  ``* `` item it is a regular claim; anywhere else (nested, numbered, quoted) the
  owner has put it there and it stays a claim, but the pipeline may not edit it.
- The section of a line is the text of the last level-2 heading above it; a
  level-1 heading ends the section, deeper headings do not.
- A top-level ``- ``/``* `` item with no block id in a pipeline section is the
  owner's new claim; anything else on a page is owner text and never touched.
"""

from __future__ import annotations

import secrets
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import regex
import yaml
from pydantic import BaseModel

from repartee.schema import (
    FACT_SLOTS,
    PAGE_DIRS,
    PAGE_PROPERTIES,
    PIPELINE_SECTIONS,
    FactValue,
    PageKind,
    page_path,
)

GEN_START = "<!-- repartee:generated:start -->"
GEN_END = "<!-- repartee:generated:end -->"
CORE_KEY = "core"
QUESTIONS_KEY = "open-questions"
EARLIER = "Earlier"

_CLAIM = regex.compile(
    r"^(?P<prefix>[ \t>]*(?:(?:[-*+]|\d{1,9}[.)])[ \t]+)?)(?P<text>.*?\S)"
    r"(?P<sep>[ \t]+)\^(?P<id>c-[0-9a-f]{8})(?P<trail>[ \t]*)$"
)
_BARE_MARKER = regex.compile(r"^(?:[-*+]|\d{1,9}[.)])$")
_STRICT_PREFIX = regex.compile(r"^[-*][ \t]+$")
_TOP_ITEM = regex.compile(r"^[-*][ \t]+\S")
_ANY_ITEM = regex.compile(r"^[ \t>]*(?:[-*+]|\d{1,9}[.)])(?:[ \t]|$)")
_COL0_ITEM = regex.compile(r"^(?:[-*+]|\d{1,9}[.)])(?:[ \t]|$)")
_BLOCK_ID_END = regex.compile(r"[ \t]\^[A-Za-z0-9-]+[ \t]*$")
_BAD_TEXT = regex.compile(r"\^[cq]-[0-9a-f]{8}|[ \t]\^[A-Za-z0-9-]+$")
_THEMATIC = regex.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_ATX = regex.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_SETEXT = regex.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_FENCE_OPEN = regex.compile(r"^[ \t>]*(`{3,}|~{3,})(.*)$")
_FENCE_CLOSE = regex.compile(r"^[ \t>]*(`{3,}|~{3,})[ \t]*$")
_QUESTION = regex.compile(
    r"^[-*][ \t]+\[(?P<state>[ xX])\][ \t]+(?P<text>.*?\S)"
    r"(?:[ \t]+\(page: (?P<page>[^()]*)\))?[ \t]+\^(?P<id>q-[0-9a-f]{8})[ \t]*$"
)
_IDLESS_TASK = regex.compile(r"^[-*][ \t]+\[[ xX]\][ \t]+\S")
_NAME = regex.compile(r"^[\p{L}\p{N}][\p{L}\p{N}._-]*$")
_FRONT_KEY = regex.compile(r"^(?P<key>[^\s#:'\"?\-][^:]*?)[ \t]*:(?:[ \t]|$)")
_DATE_VALUE = regex.compile(r"^\d{4}(?:-\d{2}(?:-\d{2})?)?$")
_WIKILINK = regex.compile(r"^\[\[([^\[\]|]+)(?:\|([^\[\]]+))?\]\]$")


class ClaimLine(BaseModel):
    """A claim (or an owner-added candidate claim) as found on a page."""

    claim_id: str | None  # None for a list item in a pipeline section with no block id
    section: str
    text: str
    line_no: int  # 0-based index into the page's lines
    top_level: bool = True  # False: the owner nested, numbered or quoted it; pipeline hands off


@dataclass
class _Info:
    region: str  # "front", "code", "gen", "body"
    heading: int = 0  # heading level, 0 when not a heading
    heading_text: str = ""
    section: str | None = None


# ----------------------------------------------------------------------- helpers


def new_claim_id() -> str:
    """A fresh random claim id: "c-" + 8 hex."""
    return "c-" + secrets.token_hex(4)


def new_question_id() -> str:
    """A fresh random question id: "q-" + 8 hex."""
    return "q-" + secrets.token_hex(4)


def clean_text(text: object) -> str | None:
    """Pipeline or review text made safe for one claim line; None when unusable."""
    if not isinstance(text, str):
        return None
    out = " ".join(text.split())
    if not out or _BAD_TEXT.search(out):
        return None
    return out


def same_text(a: str, b: str) -> bool:
    """True when two texts differ only in whitespace or Unicode normal form."""
    return _ws(a) == _ws(b)


def norm_text(text: str) -> str:
    """Normal form for duplicate and rejection checks: case, punctuation and spacing folded."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    out = regex.sub(r"[^\p{L}\p{N}]+", " ", folded).strip()
    return out or " ".join(folded.split())


def has_facts(kind: PageKind) -> bool:
    """True for page kinds whose frontmatter properties are key facts (subject pages)."""
    return kind in PIPELINE_SECTIONS


def _prop_item(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return "true" if raw else "false"
    if isinstance(raw, date):  # datetime is a date too
        return raw.isoformat()
    if isinstance(raw, str | int | float):
        return " ".join(str(raw).split()) or None
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], str | int | float):
        return f"[[{raw[0]}]]"  # an unquoted [[link]] reads as a nested list
    return _yaml_value(raw)


def property_value(raw: object, multi: bool | None = None) -> FactValue | None:
    """A frontmatter value as a fact value: a string or a list of strings; None when empty.

    ``multi`` True makes a single value a list, False makes a one-element list a single
    value; None (an owner's own property) keeps the shape found.
    """
    if isinstance(raw, list):
        one_link = len(raw) == 1 and isinstance(raw[0], list) and len(raw[0]) == 1 \
            and isinstance(raw[0][0], str | int | float)  # `key: [[Theo]]`, unquoted
        items = [x for x in (_prop_item(v) for v in ([raw[0]] if one_link else raw)) if x]
        if not items:
            return None
        if len(items) == 1 and (multi is False or (multi is None and one_link)):
            return items[0]
        return items
    item = _prop_item(raw)
    if item is None:
        return None
    return [item] if multi else item


def link_text(value: str) -> str:
    """What a value says: the label of a [[target|label]] link, or its target, or the text."""
    m = _WIKILINK.match(value.strip())
    if not m:
        return value
    return (m.group(2) or m.group(1)).strip()


def value_key(value: FactValue) -> tuple[str, ...]:
    """Normal form of a fact value for comparisons: link brackets, case and order folded."""
    items = value if isinstance(value, list) else [value]
    return tuple(sorted({norm_text(link_text(v)) for v in items}))


def mentions(text: str, names: Sequence[str]) -> bool:
    """True when ``text`` names any of ``names`` as a whole word (case folded)."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    for name in names:
        want = " ".join(unicodedata.normalize("NFKC", name).casefold().split())
        if len(want) < 2:
            continue
        pattern = rf"(?<![\p{{L}}\p{{N}}]){regex.escape(want)}(?![\p{{L}}\p{{N}}])"
        if regex.search(pattern, folded):
            return True
    return False


def _prop_scalar(value: str) -> str:
    return value if _DATE_VALUE.match(value) else _yaml_value(value)


def _prop_lines(name: str, value: FactValue) -> list[str]:
    """A property in the style Obsidian writes: ``name: value`` or a block list."""
    if isinstance(value, list):
        return [f"{name}:"] + [f"  - {_prop_scalar(v)}" for v in value]
    return [f"{name}: {_prop_scalar(value)}"]


def _ws(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def canonical_section(kind: PageKind, name: str) -> str | None:
    """The pipeline section of this kind matching ``name`` loosely, else None."""
    key = _ws(name).casefold()
    for section in PIPELINE_SECTIONS.get(kind, []):
        if section.casefold() == key:
            return section
    return None


def kind_for_key(key: str) -> PageKind:
    """Page kind implied by a page key's location."""
    if key == CORE_KEY:
        return PageKind.CORE
    head = key.split("/", 1)[0]
    if "/" in key:
        for kind, folder in PAGE_DIRS.items():
            if folder == head:
                return kind
    return PageKind.META


def _split(text: str) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    start = 0
    while start < len(text):
        i = text.find("\n", start)
        if i < 0:
            lines.append((text[start:], ""))
            break
        line = text[start:i]
        if line.endswith("\r"):
            lines.append((line[:-1], "\r\n"))
        else:
            lines.append((line, "\n"))
        start = i + 1
    return lines


def _yaml_value(value: object) -> str:
    out = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=10**9)
    out = out.strip()
    if out.endswith("\n..."):
        out = out[: -len("\n...")].strip()
    return out


def _scan_code(contents: Sequence[str], start: int = 0) -> tuple[list[int], str | None]:
    """Indexes of lines in fences or comments, and the closer still owed at the end."""
    code: list[int] = []
    fence: str | None = None
    comment: str | None = None
    for i in range(start, len(contents)):
        line = contents[i]
        if fence:
            code.append(i)
            m = _FENCE_CLOSE.match(line)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
            continue
        if comment:
            code.append(i)
            if (comment == "-->" and "-->" in line) or (
                    comment == "%%" and line.count("%%") % 2 == 1):
                comment = None
            continue
        m = _FENCE_OPEN.match(line)
        if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
            code.append(i)
            fence = m.group(1)
            continue
        stripped = line.lstrip()
        if stripped.startswith("<!--") and "-->" not in stripped[4:]:
            code.append(i)
            comment = "-->"
        elif stripped.startswith("%%") and line.count("%%") % 2 == 1:
            code.append(i)
            comment = "%%"
    return code, fence or comment


def _classify(contents: Sequence[str]) -> list[_Info]:
    n = len(contents)
    infos = [_Info("body") for _ in range(n)]
    start = 0
    if n and contents[0].lstrip("﻿").rstrip() == "---":
        for j in range(1, n):
            if contents[j].rstrip() in ("---", "..."):
                for k in range(j + 1):
                    infos[k].region = "front"
                start = j + 1
                break

    code, _ = _scan_code(contents, start)
    for i in code:
        infos[i].region = "code"

    pending: int | None = None
    for i in range(start, n):
        if infos[i].region != "body":
            continue
        s = contents[i].strip()
        if s == GEN_START:
            pending = i
        elif s == GEN_END and pending is not None:
            for k in range(pending, i + 1):
                if infos[k].region == "body":
                    infos[k].region = "gen"
            pending = None

    section: str | None = None
    for i in range(start, n):
        info = infos[i]
        if info.region != "body":
            info.section = section
            continue
        line = contents[i]
        m = _ATX.match(line)
        level, title = 0, ""
        if m:
            level, title = len(m.group(1)), (m.group(2) or "").strip()
        elif _SETEXT.match(line) and _is_setext_target(contents, infos, i):
            level = 1 if line.strip()[0] == "=" else 2
            title = contents[i - 1].strip()
            infos[i - 1].heading, infos[i - 1].heading_text = level, title
        if level:
            info.heading, info.heading_text = level, title
            if level == 1:
                section = None
            elif level == 2:
                section = title
        info.section = section
    return infos


def _is_setext_target(contents: Sequence[str], infos: list[_Info], i: int) -> bool:
    j = i - 1
    if j < 0 or infos[j].region != "body" or not contents[j].strip() or infos[j].heading:
        return False
    while j >= 0 and infos[j].region == "body" and contents[j].strip():
        line = contents[j]
        if _ANY_ITEM.match(line) or _ATX.match(line) or _THEMATIC.match(line) or infos[j].heading:
            return False
        j -= 1
    return True


# -------------------------------------------------------------------------- page


class Page:
    """One markdown page of the self-model, held as its lines."""

    def __init__(self, key: str, lines: list[tuple[str, str]]):
        self.key = key
        self._lines = lines
        self._infos: list[_Info] | None = None

    @classmethod
    def parse(cls, key: str, text: str) -> Page:
        """Page from its file text; ``render()`` gives the same text back."""
        return cls(key, _split(text))

    def render(self) -> str:
        """The page text: byte-identical to the parsed text when nothing changed."""
        return "".join(c + e for c, e in self._lines)

    # ---------------------------------------------------------------- metadata
    @property
    def kind(self) -> PageKind:
        return kind_for_key(self.key)

    def frontmatter(self) -> dict:
        """Parsed frontmatter, {} when absent or not valid YAML."""
        infos = self._info()
        front = [c for (c, _), inf in zip(self._lines, infos, strict=True) if inf.region == "front"]
        if len(front) < 2:
            return {}
        try:
            data = yaml.safe_load("\n".join(front[1:-1]))
        except yaml.YAMLError:
            return {}
        return data if isinstance(data, dict) else {}

    def frontmatter_ok(self) -> bool:
        """False when the page has frontmatter that is not a valid YAML mapping."""
        bounds = self._front_bounds()
        if bounds is None:
            return True
        text = "\n".join(c for c, _ in self._lines[bounds[0] + 1: bounds[1]])
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            return False
        return data is None or isinstance(data, dict)

    def properties(self) -> dict[str, FactValue]:
        """Frontmatter properties about the page's subject (key facts), values normalized.

        Page metadata (kind, title, aliases, updated) is left out, and so are empty values.
        """
        out: dict[str, FactValue] = {}
        for raw_name, raw in self.frontmatter().items():
            name = str(raw_name).strip()
            if not name or name in PAGE_PROPERTIES:
                continue
            slot = FACT_SLOTS.get(name)
            value = property_value(raw, slot.multi if slot else None)
            if value is not None:
                out[name] = value
        return out

    def set_property(self, name: str, value: FactValue) -> None:
        """Write one property, replacing its lines only; adds frontmatter when there is none."""
        lines = _prop_lines(name, value)
        span = self._property_span(name)
        if span:
            self._delete(*span)
            self._insert(span[0], lines)
            return
        bounds = self._front_bounds()
        if bounds:
            self._insert(bounds[1], lines)
        else:
            self._insert(0, ["---", *lines, "---"])

    def remove_property(self, name: str) -> None:
        """Remove one property's lines; nothing else in the frontmatter changes."""
        span = self._property_span(name)
        if span:
            self._delete(*span)

    def _front_bounds(self) -> tuple[int, int] | None:
        front = [i for i, inf in enumerate(self._info()) if inf.region == "front"]
        return (front[0], front[-1]) if len(front) >= 2 else None

    def _property_span(self, name: str) -> tuple[int, int] | None:
        bounds = self._front_bounds()
        if bounds is None:
            return None
        for i in range(bounds[0] + 1, bounds[1]):
            m = _FRONT_KEY.match(self._lines[i][0])
            if not m or m.group("key").strip() != name:
                continue
            j = i + 1
            while j < bounds[1]:
                line = self._lines[j][0]
                if _FRONT_KEY.match(line) or line.startswith("#"):
                    break
                j += 1
            while j > i + 1 and not self._lines[j - 1][0].strip():
                j -= 1
            return i, j
        return None

    @property
    def title(self) -> str:
        title = self.frontmatter().get("title")
        if isinstance(title, str | int | float) and str(title).strip():
            return str(title).strip()
        for inf in self._info():
            if inf.region == "body" and inf.heading == 1 and inf.heading_text:
                return inf.heading_text
        return self.key.rsplit("/", 1)[-1]

    @property
    def aliases(self) -> list[str]:
        raw = self.frontmatter().get("aliases")
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        return [str(a).strip() for a in raw if isinstance(a, str | int | float) and str(a).strip()]

    def body(self) -> str:
        """Page text without frontmatter, LF line endings."""
        infos = self._info()
        return "\n".join(
            c for (c, _), inf in zip(self._lines, infos, strict=True) if inf.region != "front"
        )

    # ------------------------------------------------------------------ claims
    def claim_lines(self) -> list[ClaimLine]:
        """Claim lines and owner-added candidates (claim_id None), in page order."""
        out: list[ClaimLine] = []
        kind = self.kind
        for i, ((content, _), inf) in enumerate(zip(self._lines, self._info(), strict=True)):
            if inf.region != "body" or inf.heading:
                continue
            section = inf.section or ""
            m = _CLAIM.match(content)
            if m and _BARE_MARKER.match(m.group("text")):
                m = None  # "- ^c-..": the owner emptied the claim
            if m:
                top = bool(_STRICT_PREFIX.match(m.group("prefix")))
                canon = canonical_section(kind, section) or section
                out.append(ClaimLine(claim_id=m.group("id"), section=canon, text=m.group("text"),
                                     line_no=i, top_level=top))
            elif self._is_candidate(content, inf):
                text = content.split(None, 1)[1].strip()
                out.append(ClaimLine(claim_id=None, section=canonical_section(kind, section) or "",
                                     text=text, line_no=i))
        return out

    def _is_candidate(self, content: str, inf: _Info) -> bool:
        if inf.section is None or canonical_section(self.kind, inf.section) is None:
            return False
        if not _TOP_ITEM.match(content) or _THEMATIC.match(content):
            return False
        return not _BLOCK_ID_END.search(content)

    def find_claim(self, claim_id: str) -> int | None:
        """Line index of the first claim line with this id."""
        for cl in self.claim_lines():
            if cl.claim_id == claim_id:
                return cl.line_no
        return None

    def claim_line(self, claim_id: str) -> ClaimLine | None:
        """The first claim line with this id."""
        return next((c for c in self.claim_lines() if c.claim_id == claim_id), None)

    def add_claim(self, section: str, text: str, claim_id: str) -> None:
        """Insert a claim line at the end of the section's claims, creating the section."""
        self._insert_item(section, [f"- {text} ^{claim_id}"])

    def set_claim_text(self, claim_id: str, text: str) -> None:
        """Replace the text of a claim line, keeping its marker, id and trailing spaces."""
        i = self._require(claim_id)
        m = _CLAIM.match(self._lines[i][0])
        assert m is not None
        content = m.group("prefix") + text + m.group("sep") + "^" + m.group("id") + m.group("trail")
        self._set(i, content)

    def move_claim(self, claim_id: str, section: str) -> None:
        """Move a claim line, with the owner lines attached under it, to another section."""
        i = self._require(claim_id)
        end = self._item_end(i)
        block = [c for c, _ in self._lines[i:end]]
        self._delete(i, end)
        self._insert_item(section, block)

    def remove_claim(self, claim_id: str) -> None:
        """Remove the claim line only; owner lines around it stay."""
        i = self._require(claim_id)
        self._delete(i, i + 1)

    def set_line_id(self, line_no: int, claim_id: str) -> None:
        """Give a claim line a new id, or an id-less item its first one."""
        content = self._lines[line_no][0]
        m = _CLAIM.match(content)
        if m:
            content = content[: m.start("id")] + claim_id + content[m.end("id") :]
        else:
            core = content.rstrip(" \t")
            content = f"{core} ^{claim_id}{content[len(core):]}"
        self._set(line_no, content)

    # --------------------------------------------------------------- generated
    def _gen_pair(self) -> tuple[int, int] | None:
        infos = self._info()
        start = None
        for i, inf in enumerate(infos):
            if inf.region != "gen":
                continue
            s = self._lines[i][0].strip()
            if s == GEN_START and start is None:
                start = i
            elif s == GEN_END and start is not None:
                return start, i
        return None

    def generated(self) -> str:
        """Content between the generated markers ("" when there is none)."""
        pair = self._gen_pair()
        if not pair:
            return ""
        return "\n".join(c for c, _ in self._lines[pair[0] + 1 : pair[1]]).strip("\n")

    def set_generated(self, markdown: str) -> None:
        """Replace the content between the generated markers, adding them when missing."""
        new = []
        if markdown.strip():
            new = [c.rstrip("\r") for c in markdown.strip("\r\n").split("\n")]
        new = [c for c in new if c.strip() not in (GEN_START, GEN_END)]
        new += _closers(new)
        pair = self._gen_pair()
        if pair:
            self._delete(pair[0] + 1, pair[1])
            self._insert(pair[0] + 1, new)
            return
        block = ["## Generated", GEN_START, *new, GEN_END]
        if self._lines and self._lines[-1][0].strip():
            block.insert(0, "")
        self._insert(len(self._lines), block)

    def section_text(self, name: str) -> str:
        """Owner text of a section (LF, outer blank lines stripped), "" when absent."""
        bounds = self._section_bounds(name)
        if not bounds:
            return ""
        infos = self._info()
        lines = [self._lines[i][0] for i in range(bounds[0] + 1, bounds[1])
                 if infos[i].region != "gen"]
        return "\n".join(lines).strip("\n")

    # ---------------------------------------------------------------- questions
    def question_lines(self) -> list[tuple[int, str | None, bool, str, str | None]]:
        """(line_no, id or None, resolved, text, page) for each question item."""
        out = []
        for i, ((content, _), inf) in enumerate(zip(self._lines, self._info(), strict=True)):
            if inf.region != "body":
                continue
            m = _QUESTION.match(content)
            if m:
                page = (m.group("page") or "").strip() or None
                out.append((i, m.group("id"), m.group("state") != " ", m.group("text"), page))
            elif _IDLESS_TASK.match(content) and not _BLOCK_ID_END.search(content):
                text = content.split("]", 1)[1].strip()
                out.append((i, None, content.split("[", 1)[1][0] != " ", text, None))
        return out

    def append_question(self, content: str) -> None:
        """Append a question line after the last question (or at the end)."""
        qs = self.question_lines()
        if qs:
            self._insert(qs[-1][0] + 1, [content])
            return
        block = [content]
        if self._lines and self._lines[-1][0].strip():
            block.insert(0, "")
        self._insert(len(self._lines), block)

    def tick_question(self, line_no: int) -> None:
        """Mark the question on this line resolved: ``[ ]`` becomes ``[x]``."""
        content = self._lines[line_no][0]
        i = content.index("[")
        self._set(line_no, content[: i + 1] + "x" + content[i + 2 :])

    # ----------------------------------------------------------------- internals
    def _info(self) -> list[_Info]:
        if self._infos is None:
            self._infos = _classify([c for c, _ in self._lines])
        return self._infos

    def _require(self, claim_id: str) -> int:
        i = self.find_claim(claim_id)
        if i is None:
            raise KeyError(claim_id)
        return i

    def _eol(self) -> str:
        crlf = sum(1 for _, e in self._lines if e == "\r\n")
        lf = sum(1 for _, e in self._lines if e == "\n")
        return "\r\n" if crlf > lf else "\n"

    def _set(self, i: int, content: str) -> None:
        self._lines[i] = (content, self._lines[i][1])
        self._infos = None

    def _insert(self, index: int, contents: list[str]) -> None:
        if not contents:
            return
        eol = self._eol()
        new = [(c, eol) for c in contents]
        if index == len(self._lines) and self._lines and self._lines[-1][1] == "":
            self._lines[-1] = (self._lines[-1][0], eol)
            new[-1] = (new[-1][0], "")
        self._lines[index:index] = new
        self._infos = None

    def _delete(self, start: int, end: int) -> None:
        if start >= end:
            return
        at_end = end == len(self._lines)
        no_final = at_end and self._lines[-1][1] == ""
        del self._lines[start:end]
        if no_final and self._lines:
            self._lines[-1] = (self._lines[-1][0], "")
        self._infos = None

    def _section_bounds(self, name: str) -> tuple[int, int] | None:
        want = _ws(name).casefold()
        infos = self._info()
        start = None
        for i, inf in enumerate(infos):
            if inf.region != "body" or not inf.heading or inf.heading > 2:
                continue
            if start is not None:
                return start, i
            if inf.heading == 2 and _ws(inf.heading_text).casefold() == want:
                start = i
        return (start, len(infos)) if start is not None else None

    def _item_end(self, i: int) -> int:
        """End (exclusive) of the list item starting at line i: its attached lines."""
        n = len(self._lines)
        infos = self._info()
        j = i + 1
        while j < n:
            content, inf = self._lines[j][0], infos[j]
            if inf.region in ("front", "gen"):
                break
            if not content.strip():
                k = j
                while k < n and not self._lines[k][0].strip():
                    k += 1
                if k < n and infos[k].region not in ("front", "gen") \
                        and self._lines[k][0][:1] in (" ", "\t"):
                    j = k
                    continue
                break
            if inf.heading or _COL0_ITEM.match(content) or _THEMATIC.match(content):
                break
            if inf.region == "code" and content[:1] not in (" ", "\t", ">") and (
                    j == 0 or infos[j - 1].region != "code"):
                break
            j += 1
        return j

    def _insert_item(self, section: str, block: list[str]) -> None:
        bounds = self._section_bounds(section) or self._create_section(section)
        h, e = bounds
        infos = self._info()
        claims = [
            cl.line_no for cl in self.claim_lines()
            if cl.claim_id and cl.top_level and h < cl.line_no < e
        ]
        if claims:
            self._insert(self._item_end(claims[-1]), block)
            return
        last = None
        for i in range(h + 1, e):
            if self._lines[i][0].strip():
                last = i
        if last is None:
            self._insert(h + 1, block)
            return
        content = self._lines[last][0]
        pad = [] if (_ANY_ITEM.match(content) and infos[last].region == "body") else [""]
        self._insert(last + 1, pad + block)

    def _create_section(self, section: str) -> tuple[int, int]:
        name = canonical_section(self.kind, section) or section.strip()
        order = PIPELINE_SECTIONS.get(self.kind, [])
        pos = len(self._lines)
        if name in order:
            idx = order.index(name)
            found = False
            for later in order[idx + 1 :]:
                b = self._section_bounds(later)
                if b:
                    pos, found = b[0], True
                    break
            if not found:
                for earlier in reversed(order[:idx]):
                    b = self._section_bounds(earlier)
                    if b:
                        pos = b[1]
                        break
        lines = [f"## {name}"]
        if pos > 0 and self._lines[pos - 1][0].strip():
            lines.insert(0, "")
        if pos < len(self._lines):
            lines.append("")
        self._insert(pos, lines)
        b = self._section_bounds(name)
        assert b is not None
        return b


def _closers(lines: list[str]) -> list[str]:
    """Lines that close a fence or comment left open by ``lines``."""
    _, owed = _scan_code(lines)
    return [owed] if owed else []


def new_page(kind: PageKind, slug: str, title: str, aliases: Sequence[str] = ()) -> Page:
    """A fresh page for ``kind`` with its frontmatter, title and empty sections."""
    if not isinstance(slug, str) or not _NAME.match(slug) or ".." in slug:
        raise ValueError(f"not a usable page name: {slug!r}")
    kind = PageKind(kind)
    if kind is PageKind.CORE:
        key = CORE_KEY
    elif kind is PageKind.META:
        key = slug
    else:
        key = page_path(kind, slug)
    title = " ".join(str(title).split()) or slug
    front = [
        "---",
        f"kind: {kind.value}",
        f"title: {_yaml_value(title)}",
        f"aliases: {_yaml_value([' '.join(str(a).split()) for a in aliases if str(a).strip()])}",
        f"updated: {date.today().isoformat()}",
        "---",
        f"# {title}",
        "",
    ]
    if kind is PageKind.CORE:
        body = ["## Pinned", "", "## Generated", GEN_START, GEN_END]
    elif kind is PageKind.VOICE:
        body = ["## Generated", GEN_START, GEN_END, "", "## Notes"]
    elif kind is PageKind.META:
        body = []
    else:
        body = []
        for section in PIPELINE_SECTIONS.get(kind, []):
            body += [f"## {section}", ""]
        body.append("## Notes")
    lines = front + body
    if not body:
        lines = front[:-1]
    return Page(key, [(c, "\n") for c in lines])
