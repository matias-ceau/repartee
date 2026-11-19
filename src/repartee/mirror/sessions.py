"""Session transcripts in the internal format that ``ingest.parse_transcript`` reads back.

::

    ---
    kind: mirror              # interview for interviews
    date: 2026-11-14T10:32:00
    mode: chat
    title: "what did I think about moving"
    ---
    **me:** what did I think about moving?

    **mirror:** I went back and forth all spring.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import regex

from repartee.schema import slugify

Speaker = Literal["me", "mirror"]

# A line inside a turn that would be read as the start of a new turn.
_TURN_LIKE = regex.compile(r"^\*\*(?:me|mirror)(?::\*\*|\*\*\s*:)", regex.IGNORECASE)


def _yaml_str(text: str) -> str:
    # JSON strings are valid YAML double-quoted scalars.
    return json.dumps(text, ensure_ascii=False)


def title_from(text: str, words: int = 8) -> str:
    """A short title from the first words of a message."""
    first = " ".join(text.split()[:words])
    return first.rstrip(" ?!.,;:") or "session"


class SessionLog:
    """Turns of one session, written to ``directory`` as one markdown file."""

    def __init__(self, directory: Path, mode: str, title: str | None = None,
                 started: datetime | None = None):
        self.directory = Path(directory)
        self.mode = mode
        self.title = title
        self.started = (started or datetime.now()).replace(microsecond=0)
        self.turns: list[tuple[Speaker, str]] = []
        self.path: Path | None = None
        self._written = 0

    def add(self, speaker: Speaker, text: str) -> None:
        """Append a turn; empty text is ignored."""
        text = text.strip()
        if text:
            self.turns.append((speaker, text))

    def render(self) -> str:
        """The whole transcript as markdown."""
        kind = "interview" if self.mode == "interview" else "mirror"
        title = self.title or next((t for s, t in self.turns if s == "me"), "") or self.mode
        lines = ["---", f"kind: {kind}", f"date: {self.started.isoformat()}",
                 f"mode: {self.mode}", f"title: {_yaml_str(title_from(title))}", "---"]
        for speaker, text in self.turns:
            body = [("\\" + line if _TURN_LIKE.match(line) else line)
                    for line in text.split("\n")]
            lines += ["", f"**{speaker}:** {body[0]}", *body[1:]]
        return "\n".join(lines) + "\n"

    def write(self) -> Path | None:
        """Write (or rewrite) the file; None when there is nothing to write."""
        if not self.turns:
            return None
        if self.path is None:
            title = self.title or next((t for s, t in self.turns if s == "me"), self.mode)
            stem = f"{self.started:%Y-%m-%d-%H%M%S}-{slugify(title_from(title, 6))[:48]}"
            self.directory.mkdir(parents=True, exist_ok=True)
            path, n = self.directory / f"{stem}.md", 1
            while path.exists():
                n += 1
                path = self.directory / f"{stem}-{n}.md"
            self.path = path
        self.path.write_text(self.render(), encoding="utf-8")
        self._written = len(self.turns)
        return self.path

    @property
    def unsaved(self) -> bool:
        """True when turns were added since the last write."""
        return len(self.turns) > self._written
