"""What the model sees besides tool results: the system blocks and the per-session date note.

The system blocks are the mode's instructions and the core page. Both stay byte-identical
across requests so the provider can cache them; anything that changes per request (today's
date above all) goes into the first user turn instead.
"""

from __future__ import annotations

from datetime import datetime

from repartee.llm.base import SystemBlock
from repartee.mirror import prompts
from repartee.schema import Claim, ClaimOrigin, ClaimStatus

_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December")


def build_system(mode: str, owner: str, core_text: str) -> list[SystemBlock]:
    """Instructions, then the core page; the second block marks the end of the cached prefix."""
    return [
        SystemBlock(text=prompts.instructions(mode, owner)),
        SystemBlock(text=prompts.core_block(owner, core_text), cache=True),
    ]


def date_note(now: datetime) -> str:
    """Today's date, in words, for the first user turn of a session."""
    return f"[Today is {_DAYS[now.weekday()]}, {now.day} {_MONTHS[now.month - 1]} {now.year}.]"


# ------------------------------------------------------------- formatting for tool output


def fmt_when(when: datetime | None) -> str:
    """2022-04-02, or 2022-04-02 09:03 when the time is known."""
    if when is None:
        return "undated"
    if (when.hour, when.minute, when.second) == (0, 0, 0):
        return f"{when:%Y-%m-%d}"
    return f"{when:%Y-%m-%d %H:%M}"


def fmt_span(start: datetime | None, end: datetime | None) -> str:
    """A compact date range."""
    if start is None or end is None or start == end:
        return fmt_when(start or end)
    if start.date() == end.date() and (start.hour, start.minute) != (0, 0):
        return f"{fmt_when(start)}–{end:%H:%M}"
    return f"{start:%Y-%m-%d} – {end:%Y-%m-%d}"


def validity(claim: Claim) -> str:
    """'from 2022', 'until 2021', '2019 – 2021' or ''."""
    if claim.valid_from and claim.valid_to:
        return f"{claim.valid_from} – {claim.valid_to}"
    if claim.valid_from:
        return f"from {claim.valid_from}"
    if claim.valid_to:
        return f"until {claim.valid_to}"
    return ""


def claim_status(claim: Claim) -> str:
    """Status in a few words: 'confirmed', 'inferred 0.45, until 2021', ..."""
    if claim.status == ClaimStatus.CONFIRMED:
        parts = ["confirmed" + (", written by the owner" if claim.origin == ClaimOrigin.USER
                                else "")]
    else:
        source = ", from an interview" if claim.origin == ClaimOrigin.INTERVIEW else ""
        parts = [f"{claim.status} {claim.confidence:.2f}{source}"]
    if valid := validity(claim):
        parts.append(valid)
    return " · ".join(parts)


def clip(text: str, limit: int, *, flat: bool = True) -> str:
    """Trim to `limit` characters with an ellipsis; `flat` folds whitespace first."""
    if flat:
        text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def indent(text: str, prefix: str = "  ") -> str:
    """Indent every line."""
    return "\n".join(prefix + line if line else line for line in text.split("\n"))
