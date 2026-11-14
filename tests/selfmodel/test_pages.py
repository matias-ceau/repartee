"""Page grammar: lines in, the same lines out; edits touch only what they must."""

from __future__ import annotations

import pytest
import yaml

from repartee.schema import PIPELINE_SECTIONS, PageKind
from repartee.selfmodel.pages import GEN_END, GEN_START, Page, new_claim_id, new_page

from .conftest import ANA, ANA_KEY, C_LYON, C_NURSE, C_TALK, NURSE_LINE

NESTED = """# Ana Ruiz

## Who
- Friend from university. ^c-11111111
  - nested owner note
    - deeper ^c-22222222
- Works nights. ^c-33333333
1. numbered owner item
+ plus owner item

## Notes
- owner bullet in notes
"""

CODE = """# T

## Who
- real claim ^c-aaaaaaaa

```markdown
- looks like a claim ^c-bbbbbbbb
## Not a heading
```

~~~
- tilde fenced ^c-cccccccc
~~~
"""

TABLE = """# T

## Who
| a | b |
|---|---|
| x | ^c-dddddddd |
- claim after table ^c-eeeeeeee
"""

TRAILING = "# T\n\n## Who\n- claim with spaces ^c-12345678   \n- hard break  \ntext\t\n"

FIXTURES = {
    "lf": ANA,
    "crlf": ANA.replace("\n", "\r\n"),
    "no_final_newline": ANA.rstrip("\n"),
    "nested": NESTED,
    "code": CODE,
    "table": TABLE,
    "trailing_spaces": TRAILING,
    "mixed_eol": "# T\r\n\n## Who\r\n- a ^c-12345678\n- b ^c-87654321\r\n",
    "bom": "﻿---\ntitle: T\n---\n# T\n",
    "empty": "",
    "only_newlines": "\n\n\n",
    "only_frontmatter": "---\nkind: person\n---",
    "unclosed_frontmatter": "---\ntitle: x\n# T\n",
    "unclosed_fence": "# T\n\n## Who\n```\n- a ^c-12345678\n",
    "setext": "T\n===\n\nWho\n---\n- a ^c-12345678\n",
    "lone_cr": "# T\r## Who\r- a ^c-12345678\r",
    "unicode": "# Ünïcode ✨\n\n## Who\n- שלום עולם, 你好 🙂 ^c-12345678\n- odd separator\n",
    "comments": "# T\n\n## Who\n<!--\n- hidden ^c-12345678\n-->\n%%\n- hidden2 ^c-87654321\n%%\n",
    "bad_yaml": "---\ntitle: [unclosed\n---\n# T\n",
}


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_round_trip_identical_bytes(name):
    text = FIXTURES[name]
    assert Page.parse("people/x", text).render() == text


def test_code_block_id_is_not_a_claim():
    page = Page.parse("people/x", CODE)
    ids = [c.claim_id for c in page.claim_lines()]
    assert ids == ["c-aaaaaaaa"]


def test_unclosed_fence_hides_claims_below():
    page = Page.parse("people/x", FIXTURES["unclosed_fence"])
    assert page.claim_lines() == []


def test_html_and_obsidian_comments_hide_claims():
    assert Page.parse("people/x", FIXTURES["comments"]).claim_lines() == []


def test_claim_grammar_top_level_and_nested():
    page = Page.parse("people/x", NESTED)
    lines = {c.claim_id: c for c in page.claim_lines() if c.claim_id}
    assert lines["c-11111111"].top_level and lines["c-11111111"].section == "Who"
    assert lines["c-11111111"].text == "Friend from university."
    assert not lines["c-22222222"].top_level  # nested: kept alive, never edited
    assert lines["c-22222222"].text == "deeper"
    # no id-less candidates: nested, numbered and "+" items are owner text; Notes is owner's
    assert [c for c in page.claim_lines() if c.claim_id is None] == []


def test_idless_item_in_pipeline_section_is_candidate():
    text = "# T\n\n## Who\n- owner wrote this\n- owner ref ^my-ref\n---\n* star item\n"
    page = Page.parse("people/x", text)
    cands = [c for c in page.claim_lines() if c.claim_id is None]
    assert [c.text for c in cands] == ["owner wrote this", "star item"]


def test_setext_heading_changes_section():
    page = Page.parse("people/x", FIXTURES["setext"])
    assert [c.section for c in page.claim_lines()] == ["Who"]


def test_section_under_h3_stays_h2_and_h1_resets():
    text = "## Who\n### detail\n- a ^c-12345678\n# Other\n- b ^c-87654321\n"
    page = Page.parse("people/x", text)
    assert [c.section for c in page.claim_lines()] == ["Who", ""]


def test_title_and_aliases():
    page = Page.parse(ANA_KEY, ANA)
    assert page.kind is PageKind.PERSON
    assert page.title == "Ana Ruiz"
    assert page.aliases == ["Ana", "Anita"]
    assert Page.parse("topics/x", "# From heading\n").title == "From heading"
    assert Page.parse("topics/some-slug", "no heading").title == "some-slug"
    assert Page.parse("topics/x", FIXTURES["bad_yaml"]).title == "T"


def test_new_claim_id_format():
    cid = new_claim_id()
    assert cid.startswith("c-") and len(cid) == 10 and int(cid[2:], 16) >= 0


def test_new_page_layouts():
    person = new_page(PageKind.PERSON, "ana-ruiz", "Ana: the one", ["Ana", "Anita"])
    assert person.key == "people/ana-ruiz"
    text = person.render()
    front = yaml.safe_load(text.split("---\n")[1])
    assert front["kind"] == "person" and front["title"] == "Ana: the one"
    assert front["aliases"] == ["Ana", "Anita"] and "updated" in front
    headings = [ln for ln in text.splitlines() if ln.startswith("## ")]
    assert headings == [f"## {s}" for s in PIPELINE_SECTIONS[PageKind.PERSON]] + ["## Notes"]
    assert person.title == "Ana: the one" and person.claim_lines() == []

    core = new_page(PageKind.CORE, "core", "Core")
    assert core.key == "core"
    ctext = core.render()
    assert "## Pinned" in ctext and "## Generated" in ctext
    assert ctext.index(GEN_START) < ctext.index(GEN_END)
    assert core.generated() == ""

    voice = new_page(PageKind.VOICE, "whatsapp.fr", "whatsapp.fr")
    assert voice.key == "voice/whatsapp.fr"
    vtext = voice.render()
    assert vtext.index("## Generated") < vtext.index(GEN_START) < vtext.index("## Notes")

    meta = new_page(PageKind.META, "open-questions", "Open questions")
    assert meta.key == "open-questions" and meta.kind is PageKind.META


def test_new_page_rejects_path_tricks():
    for slug in ["../x", "a/b", "", ".hidden"]:
        with pytest.raises(ValueError):
            new_page(PageKind.TOPIC, slug, "x")


def test_add_claim_after_last_claim_keeps_children_attached():
    page = Page.parse("people/x", NESTED)
    page.add_claim("Who", "New one.", "c-44444444")
    lines = page.render().splitlines()
    i = lines.index("- Works nights. ^c-33333333")
    assert lines[i + 1] == "- New one. ^c-44444444"
    # the nested children of the first claim are still right under it
    j = lines.index("- Friend from university. ^c-11111111")
    assert lines[j + 1 : j + 3] == ["  - nested owner note", "    - deeper ^c-22222222"]


def test_add_claim_skips_nested_children_of_last_claim():
    text = "## Who\n- a ^c-11111111\n  - owner child\n\n    more child text\n\nOwner para.\n"
    page = Page.parse("people/x", text)
    page.add_claim("Who", "b", "c-22222222")
    assert page.render() == (
        "## Who\n- a ^c-11111111\n  - owner child\n\n    more child text\n"
        "- b ^c-22222222\n\nOwner para.\n"
    )


def test_add_claim_to_section_with_only_owner_paragraph_separates_it():
    text = "## Who\nOwner intro paragraph.\n\n## Notes\n"
    page = Page.parse("people/x", text)
    page.add_claim("Who", "a", "c-11111111")
    assert page.render() == "## Who\nOwner intro paragraph.\n\n- a ^c-11111111\n\n## Notes\n"


def test_add_claim_to_empty_section():
    page = new_page(PageKind.PERSON, "x", "X")
    page.add_claim("Relationship", "a", "c-11111111")
    assert "## Relationship\n- a ^c-11111111\n\n## How we talk" in page.render()


def test_add_claim_creates_missing_section_in_order():
    text = "# X\n\n## Who\n- a ^c-11111111\n\n## Shared history\n\n## Notes\nmine\n"
    page = Page.parse("people/x", text)
    page.add_claim("How we talk", "b", "c-22222222")
    out = page.render()
    assert out.index("## Who") < out.index("## How we talk") < out.index("## Shared history")
    assert "## How we talk\n- b ^c-22222222\n" in out
    assert out.endswith("## Notes\nmine\n")


def test_add_claim_case_insensitive_section():
    page = Page.parse("people/x", "## who\n- a ^c-11111111\n")
    page.add_claim("Who", "b", "c-22222222")
    assert page.render() == "## who\n- a ^c-11111111\n- b ^c-22222222\n"


def test_add_claim_crlf_and_no_final_newline():
    page = Page.parse("people/x", "## Who\r\n- a ^c-11111111")
    page.add_claim("Who", "b", "c-22222222")
    assert page.render() == "## Who\r\n- a ^c-11111111\r\n- b ^c-22222222"


def test_set_claim_text_keeps_marker_and_trailing():
    page = Page.parse("people/x", TRAILING)
    page.set_claim_text("c-12345678", "new words")
    assert page.render() == TRAILING.replace("claim with spaces", "new words")


def test_move_claim_carries_children():
    page = Page.parse("people/x", NESTED)
    page.move_claim("c-11111111", "Earlier")
    out = page.render()
    assert out.index("## Who") < out.index("## Earlier") < out.index("## Notes")
    earlier = out[out.index("## Earlier") :]
    assert earlier.startswith(
        "## Earlier\n- Friend from university. ^c-11111111\n  - nested owner note\n"
    )
    assert out.count("nested owner note") == 1
    assert [c.section for c in page.claim_lines() if c.claim_id == "c-11111111"] == ["Earlier"]


def test_remove_claim_removes_only_its_line():
    page = Page.parse(ANA_KEY, ANA)
    page.remove_claim(C_NURSE)
    assert page.render() == ANA.replace(NURSE_LINE, "")
    assert [c.claim_id for c in page.claim_lines()] == [C_LYON, C_TALK]


def test_remove_last_line_keeps_no_final_newline_style():
    page = Page.parse("people/x", "## Who\n- a ^c-11111111\n- b ^c-22222222")
    page.remove_claim("c-22222222")
    assert page.render() == "## Who\n- a ^c-11111111"


def test_set_line_id_appends_before_trailing_space():
    page = Page.parse("people/x", "## Who\n- owner  \n")
    (cand,) = page.claim_lines()
    page.set_line_id(cand.line_no, "c-11111111")
    assert page.render() == "## Who\n- owner ^c-11111111  \n"


def test_generated_round_trip_and_replace():
    page = new_page(PageKind.CORE, "core", "Core")
    page.set_generated("- I like trains.\n- I live by the sea.\n")
    assert page.generated() == "- I like trains.\n- I live by the sea."
    page.set_generated("Short.")
    assert page.generated() == "Short."
    assert page.render().count(GEN_START) == 1
    # generated lines are never claims, even with ids
    page.set_generated("- x ^c-12345678")
    assert page.claim_lines() == []


def test_generated_missing_markers_appended_and_owner_text_kept():
    text = "# Core\n\n## Pinned\nMine.\n"
    page = Page.parse("core", text)
    page.set_generated("Gen.")
    out = page.render()
    assert out.startswith(text)
    assert page.generated() == "Gen."


def test_generated_orphan_marker_does_not_swallow_owner_text():
    text = f"# Core\n{GEN_START}\nowner text after orphan start\n"
    page = Page.parse("core", text)
    assert page.generated() == ""
    page.set_generated("Gen.")
    page.set_generated("Gen2.")
    assert "owner text after orphan start" in page.render()
    assert page.generated() == "Gen2."


def test_generated_unbalanced_fence_is_closed():
    page = new_page(PageKind.CORE, "core", "Core")
    page.set_generated("```\nunclosed")
    page.set_generated("again")  # markers are still found
    assert page.generated() == "again"


def test_generated_markers_inside_code_are_ignored():
    text = f"```\n{GEN_START}\nx\n{GEN_END}\n```\n"
    page = Page.parse("core", text)
    assert page.generated() == ""
