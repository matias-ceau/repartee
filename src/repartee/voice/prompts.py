"""Prompts and structured-output models for the voice subsystem (style cards, drafts, revision).

The output models stay flat: the real client sends them as JSON Schema to an API that
accepts only a subset of it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, Field

from repartee.schema import Item, ReplyPair

#: Bump when the prompts change in a way that should not reuse cached responses.
PROMPT_VERSION = "voice-1"


class StyleCardOut(BaseModel):
    """What a `style_card` request returns."""

    description: str = Field(description="The description of how the person writes, only.")


class DraftOut(BaseModel):
    """What a `draft` or `revise` request returns."""

    variants: list[list[str]] = Field(
        description="One entry per version; each version is the list of its separate "
        "messages (chat) or paragraphs (longer text), text only."
    )


# ------------------------------------------------------------------- style card

STYLE_CARD_SYSTEM = """\
You describe how one person writes, for a ghostwriter who will have to imitate them. \
Numbers about their habits are measured separately; your job is what numbers miss."""


def style_card_prompt(owner: str, register_name: str, chat: bool, measured: Sequence[str],
                      texts: Sequence[str]) -> str:
    """The user message of a `style_card` request."""
    what = "chat messages (consecutive messages shown together)" if chat else "texts"
    lines = [
        f"Below are {len(texts)} {what} that {owner} wrote in the register "
        f'"{register_name}", picked evenly from everything they wrote in it, and what was '
        "measured over all of it.",
        "",
        "<measured>",
        *(f"- {line}" for line in measured),
        "</measured>",
        "",
        "<texts>",
        *(f"<text>\n{t}\n</text>" for t in texts),
        "</texts>",
        "",
        f"Write a description of how {owner} writes here, in 4 to 8 sentences, in the first "
        f'person as if {owner} were describing their own habits ("I ..."), in English, quoting '
        "characteristic words and expressions in their original language. Cover what the "
        "measurements cannot: tone and attitude, how direct or warm, humour, how people are "
        "addressed, what gets left out, and whatever a ghostwriter would get wrong by "
        "default. Describe the manner, not the content: do not summarise what the texts are "
        "about and do not repeat personal details from them.",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------------ draft


def draft_system(owner: str) -> str:
    """The stable system prompt of `draft` and `revise` requests."""
    return f"""\
You are ghostwriting for {owner}. You write texts that {owner} will send or publish as their \
own: a message to a friend, an email, a passage of an essay. {owner} reads every draft before \
it goes anywhere.

What matters is that the text sounds like {owner} wrote it, so that someone who knows them \
well could not tell otherwise. That is a different goal from writing well. Models drift \
toward a polished, friendly, complete, evenly paced assistant voice; real people are terser \
or more rambling, less tidy, less explicit, and they have tics. When {owner}'s habits and \
good style disagree, follow {owner}: their length, capitalisation, punctuation, emoji, \
laughter, abbreviations, mixing of languages, how they open and close, and how much they \
bother to explain.

You get what a good ghostwriter would ask for:
- who the text is for and what the relationship is, from {owner}'s own notes;
- a style card for this kind of text: habits measured over everything {owner} wrote in it, \
a short description, and notes {owner} wrote themselves (their notes outrank everything else);
- real texts by {owner} in the same register. They are there so you can absorb the rhythm, \
the vocabulary and the attitude. They are about other days and other things: do not copy \
their sentences and do not take facts from them;
- what {owner} answered in similar situations before, when there are some;
- the conversation being answered, when there is one;
- what {owner} wants to say, in their own words.

The content is {owner}'s, not yours. Everything the text asserts (plans, facts, reasons, \
feelings, promises, news) must come from what {owner} wants to say or from the conversation. \
Do not invent an excuse, a time, a detail or an emotion to make the text feel complete; if \
the intent is vague, stay as vague as {owner} would. The only things you may add are the \
small ones that commit {owner} to nothing and that their texts show they habitually add, \
such as a greeting, a reaction to what was just said, or a laugh.

Write in the language of the register unless the intent or the conversation clearly calls \
for another. Return only the text itself: no preamble, no explanation, no quotation marks \
around it, and no signature unless {owner} signs this kind of text."""


def _when(dt: datetime | None) -> str:
    return f' date="{dt:%Y-%m-%d}"' if dt else ""


def render_sample(item: Item, chat: bool) -> str:
    """One sample of the owner's writing, tagged with where and when it was written."""
    if chat:
        where = f' chat="{item.container}"'
    else:
        where = f' title="{item.title}"' if item.title else ""
    return f"<sample{where}{_when(item.timestamp)}>\n{item.text}\n</sample>"


def render_pair(pair: ReplyPair) -> str:
    """A past situation: what was said, and the messages the owner sent back."""
    answered = "\n".join(f"<message>{m}</message>" for m in pair.reply)
    return (f"<situation{_when(pair.timestamp)}>\n<they_wrote>\n{pair.context}\n</they_wrote>\n"
            f"<i_answered>\n{answered}\n</i_answered>\n</situation>")


def _versions(n: int) -> str:
    return "1 version" if n == 1 else f"{n} versions"


def draft_prompt(*, owner: str, register_name: str, chat: bool, intent: str,
                 recipient: str | None = None, person_page: str | None = None,
                 card: str | None = None, recipient_habits: Sequence[str] = (),
                 samples: Sequence[str] = (), pairs: Sequence[str] = (),
                 context: str | None = None, burst_hint: str | None = None,
                 variants: int = 1) -> str:
    """The user message of a `draft` request; every part is optional except the intent."""
    parts: list[str] = []
    if recipient:
        about = (f"\n\nWhat {owner} knows about them, from their self-model:\n"
                 f"<person>\n{person_page}\n</person>" if person_page else "")
        parts.append(f"{owner} is writing to {recipient}.{about}")
    if card:
        parts.append(f'{owner}\'s style card for "{register_name}":\n<style_card>\n{card}\n'
                     "</style_card>")
    if recipient and recipient_habits:
        listed = "\n".join(f"- {line}" for line in recipient_habits)
        parts.append(f"Measured on {owner}'s messages to {recipient} specifically (where this "
                     f"differs from the card, follow this):\n<with_them>\n{listed}\n</with_them>")
    if samples:
        parts.append(f"Real texts by {owner} in this register, to absorb and not to copy:\n"
                     "<samples>\n" + "\n".join(samples) + "\n</samples>")
    if pairs:
        parts.append(f"What {owner} answered in similar situations before:\n"
                     "<similar_situations>\n" + "\n".join(pairs) + "\n</similar_situations>")
    if context:
        parts.append(f"The conversation {owner} is answering, most recent last:\n"
                     f"<conversation>\n{context}\n</conversation>")
    parts.append(f"What {owner} wants to say, in their own words:\n<intent>\n{intent}\n</intent>")

    if chat:
        task = (f"Write {_versions(variants)} of what {owner} would send now. Each version is "
                f"a list of separate messages, in the order {owner} would send them, split "
                "where they would split rather than into tidy sentences.")
        if burst_hint:
            task += " " + burst_hint
    else:
        task = (f"Write {_versions(variants)} of the text. Each version is a list of its "
                "paragraphs.")
    if variants > 1:
        task += " The versions should differ in wording and shape, not in what they say."
    parts.append(task)
    return "\n\n".join(parts)


def burst_hint(owner: str, words_median: float | None, burst_mean: float | None) -> str | None:
    """One sentence on how many messages, of what length, the owner usually sends."""
    if words_median is None and burst_mean is None:
        return None
    bits = []
    if burst_mean is not None:
        n = max(1, round(burst_mean))
        bits.append("one message at a time" if n == 1 else f"about {n} messages in a row")
    if words_median is not None:
        bits.append(f"of about {max(1, round(words_median))} words each")
    return f"In this register {owner} usually sends " + ", ".join(bits) + "."


def revise_prompt(owner: str, flagged: Sequence[tuple[int, Sequence[str]]]) -> str:
    """The follow-up message of a `revise` request: each off version with its deviations."""
    blocks = []
    for number, messages in flagged:
        blocks.append(f"Version {number}:\n" + "\n".join(f"- {m}" for m in messages))
    order = ", ".join(str(n) for n, _ in flagged)
    one = len(flagged) == 1
    return (
        f"I measured the draft against everything {owner} has written in this register, and "
        f"{'it reads' if one else 'these versions read'} differently from how they actually "
        "write:\n\n" + "\n\n".join(blocks) + "\n\n"
        f"Rewrite {'that version so it reads' if one else 'those versions so they read'} like "
        f"{owner}. Keep what is said exactly the same and keep whatever already sounds like "
        f"them; change only the form. Return {_versions(len(flagged))}, in this order: {order}."
    )
