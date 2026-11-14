"""The pieces of the mirror's screen: conversation turns, drafts, the lookups panel, review, eval.

Every widget keeps the plain text it shows in ``text`` (or a few named attributes), so tests
and the app can read it back without parsing what Textual renders.
"""

from __future__ import annotations

from collections.abc import Sequence

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static

REFS_SHOWN = 5


class Line(Static):
    """A Static that shows text literally (no markup) and remembers it."""

    def __init__(self, text: str = "", **kwargs) -> None:
        super().__init__(text, markup=False, **kwargs)
        self.text = text

    def set_text(self, text: str) -> None:
        """Replace what the line shows."""
        self.text = text
        self.update(text)


class ModeBar(Horizontal):
    """The top line: the app name and the current mode."""

    def __init__(self, mode: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._title = Line(self._label(mode), id="modebar-title")

    @staticmethod
    def _label(mode: str) -> str:
        return f"repartee · {mode}"

    @property
    def text(self) -> str:
        return self._title.text

    def compose(self) -> ComposeResult:
        yield self._title
        yield Line("/help", id="modebar-help")

    def set_mode(self, mode: str) -> None:
        """Show another mode."""
        self._title.set_text(self._label(mode))


class Turn(Horizontal):
    """One turn of the conversation: 'you' (the owner) or 'me' (the mirror)."""

    def __init__(self, who: str, text: str = "", **kwargs) -> None:
        super().__init__(classes=f"turn turn-{who}", **kwargs)
        self.who = who
        self._body = Line(text, classes="turn-body")

    @property
    def text(self) -> str:
        return self._body.text

    def compose(self) -> ComposeResult:
        yield Line(self.who, classes="turn-who")
        yield self._body

    def append(self, text: str) -> None:
        """Add streamed text to the end of the turn."""
        self._body.set_text(self._body.text + text)


class Notice(Line):
    """A short line from the program itself (not the mirror): status, errors, help."""

    def __init__(self, text: str, *, error: bool = False) -> None:
        super().__init__(text, classes="notice error" if error else "notice")


class HelpBlock(Vertical):
    """The commands, one per row, the description wrapping in its own column."""

    def __init__(self, rows: Sequence[tuple[str, str]]) -> None:
        super().__init__(classes="help")
        self.rows = list(rows)

    @property
    def text(self) -> str:
        return "\n".join(f"{cmd}  {what}" for cmd, what in self.rows)

    def compose(self) -> ComposeResult:
        for cmd, what in self.rows:
            with Horizontal(classes="help-row"):
                yield Line(cmd, classes="help-cmd")
                yield Line(what, classes="help-what")


class DraftBlock(Vertical):
    """A draft, each message of the burst as its own block, the way it would be sent."""

    def __init__(self, heading: str, variants: Sequence[Sequence[str]]) -> None:
        super().__init__(classes="draft")
        self.heading = heading
        self.messages = [list(v) for v in variants if v]

    def compose(self) -> ComposeResult:
        yield Line(self.heading, classes="draft-heading")
        for n, variant in enumerate(self.messages):
            if n:
                yield Line("or", classes="draft-or")
            for message in variant:
                yield Line(message, classes="bubble")


class LookupEntry(Vertical):
    """One lookup: its summary, then what it found, one line each."""

    def __init__(self, summary: str, refs: Sequence[str], *, error: bool = False) -> None:
        super().__init__(classes="lookup error" if error else "lookup")
        self.summary = summary
        self.refs = list(refs)

    @property
    def text(self) -> str:
        return "\n".join([self.summary, *(f"  {ref}" for ref in self.refs)])

    def compose(self) -> ComposeResult:
        yield Line(self.summary, classes="lookup-summary")
        shown = self.refs[:REFS_SHOWN]
        for ref in shown:
            yield Line(ref, classes="lookup-ref")
        if len(self.refs) > len(shown):
            yield Line(f"+{len(self.refs) - len(shown)} more", classes="lookup-ref")


class LookupPanel(Vertical):
    """The 'looked up' panel: each lookup behind the current answer."""

    EMPTY = "Nothing looked up for this answer."

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.entries: list[LookupEntry] = []
        self._empty = Line(self.EMPTY, classes="lookups-empty")
        self._list = VerticalScroll(classes="lookups-list")

    def compose(self) -> ComposeResult:
        yield Line("looked up", classes="lookups-title")
        with self._list:
            yield self._empty

    def clear(self) -> None:
        """Forget the previous answer's lookups."""
        for entry in self.entries:
            entry.remove()
        self.entries = []
        self._empty.display = True

    def scroll_to_top(self) -> None:
        """Show the first lookup (after an answer, read from the top)."""
        self._list.scroll_home(animate=False)

    def add(self, summary: str, refs: Sequence[str], *, error: bool = False) -> LookupEntry:
        """List one lookup with the things it found (the same thing listed once)."""
        entry = LookupEntry(summary, list(dict.fromkeys(refs)), error=error)
        self.entries.append(entry)
        self._empty.display = False
        self._list.mount(entry)
        self._list.scroll_end(animate=False)
        return entry


class ReviewView(VerticalScroll, can_focus=True):
    """One claim at a time with the material behind it; y, e, n, s act on it."""

    BINDINGS = [
        Binding("y", "act('y')", "confirm"),
        Binding("e", "act('e')", "edit"),
        Binding("n", "act('n')", "reject"),
        Binding("s", "act('s')", "skip"),
        Binding("slash", "act('/')", "command", show=False),
    ]

    class Act(Message):
        """A key pressed in review."""

        def __init__(self, key: str) -> None:
            super().__init__()
            self.key = key

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.remaining = 0
        self.card_text = ""
        self._count = Line("", classes="view-status")
        self._card = Vertical(classes="card")
        self._message = Line("", classes="view-message")

    @property
    def message_text(self) -> str:
        return self._message.text

    def compose(self) -> ComposeResult:
        yield self._count
        yield self._card
        yield self._message

    def action_act(self, key: str) -> None:
        self.post_message(self.Act(key))

    def show(self, lines: Sequence[tuple[str, str]], remaining: int) -> None:
        """Show a card as (text, class) lines, and how many claims are left."""
        self.remaining = remaining
        self.card_text = "\n".join(text for text, _ in lines)
        self._count.set_text(f"{remaining} left to review" if remaining
                             else "Review finished")
        self._card.remove_children()
        self._card.mount_all([Line(text, classes=cls) for text, cls in lines])
        self.scroll_home(animate=False)

    def say(self, text: str) -> None:
        """A line under the card: what the last key did, or an error."""
        self._message.set_text(text)


class EvalView(VerticalScroll, can_focus=True):
    """The blind voice test: two texts, A and B; which did the owner write?"""

    BINDINGS = [
        Binding("a", "pick('a')", "A"),
        Binding("b", "pick('b')", "B"),
        Binding("slash", "pick('/')", "command", show=False),
    ]

    class Pick(Message):
        """A key pressed in the blind test."""

        def __init__(self, key: str) -> None:
            super().__init__()
            self.key = key

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.text_a = ""
        self.text_b = ""
        self._status = Line("", classes="view-status")
        self._a = Line("", classes="eval-text")
        self._b = Line("", classes="eval-text")
        self._result = Line("", classes="view-message")
        self._boxes: list[Vertical] = []

    @property
    def result_text(self) -> str:
        return self._result.text

    @property
    def status_text(self) -> str:
        return self._status.text

    def compose(self) -> ComposeResult:
        yield self._status
        with Vertical(classes="eval-box") as box_a:
            yield Line("A", classes="eval-label")
            yield self._a
        with Vertical(classes="eval-box") as box_b:
            yield Line("B", classes="eval-label")
            yield self._b
        self._boxes = [box_a, box_b]
        yield self._result

    def action_pick(self, key: str) -> None:
        self.post_message(self.Pick(key))

    def show_pair(self, status: str, a: Sequence[str], b: Sequence[str]) -> None:
        """Show the next two texts."""
        self.text_a, self.text_b = "\n\n".join(a), "\n\n".join(b)
        self._status.set_text(status)
        self._a.set_text(self.text_a)
        self._b.set_text(self.text_b)
        for box in self._boxes:
            box.display = True
        self.scroll_home(animate=False)

    def show_message(self, status: str, result: str = "") -> None:
        """No pair on screen: preparing, finished, or nothing to test."""
        self.text_a = self.text_b = ""
        self._status.set_text(status)
        for box in self._boxes:
            box.display = False
        self._result.set_text(result)

    def say(self, text: str) -> None:
        """The line under the texts: the last answer and the running score."""
        self._result.set_text(text)
