"""Exported AI assistant conversations (Claude.ai and ChatGPT). Invented fixtures only."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime

from repartee.config import ChunkingConfig, SourceConfig
from repartee.distill.extract import (
    ExtractedObservation,
    ExtractionResult,
    clean_observations,
    render_chunk,
)
from repartee.ingest import ingest_source
from repartee.ingest.assistant import AssistantAdapter, pasted_reason, strip_code_blocks
from repartee.ingest.chunking import ASSISTANT_CONTEXT_CHARS, chunk_items
from repartee.schema import Kind

from .helpers import parse_with, write

SRC = SourceConfig(type="assistant", path="sources/assistant")

OWN_FR = ("Je voudrais organiser un pique-nique au bord du canal dimanche, mais je ne sais "
          "pas quoi apporter pour huit personnes. Tu as des idées simples ?")
OWN_EN = ("I keep putting off repotting the fig tree on my balcony. Is late autumn a bad "
          "time to do it, or does it not matter much for a plant that size?")
LONG_REPLY = ("Here is a detailed plan for the picnic. " * 60).strip()


def claude_export() -> list[dict]:
    """Two invented Claude.ai conversations."""
    return [
        {
            "uuid": "c1a2b3c4-0000-4000-8000-000000000001",
            "name": "Pique-nique au canal",
            "created_at": "2025-05-02T08:00:00.000000Z",
            "updated_at": "2025-05-02T08:10:00.000000Z",
            "account": {"uuid": "ignored"},
            "chat_messages": [
                {"uuid": "m-001", "sender": "human", "text": "flat text, not used",
                 "content": [{"type": "text", "text": OWN_FR}],
                 "created_at": "2025-05-02T08:00:05.000000Z", "attachments": [], "files": []},
                {"uuid": "m-002", "sender": "assistant", "text": LONG_REPLY,
                 "content": [{"type": "thinking", "thinking": "hidden"},
                             {"type": "text", "text": LONG_REPLY}],
                 "created_at": "2025-05-02T08:00:30.000000Z"},
                {"uuid": "m-003", "sender": "human", "text": "Merci, je vais faire des wraps.",
                 "content": [], "created_at": "2025-05-02T08:05:00Z",
                 "attachments": [{"file_name": "menu.txt", "extracted_content": "x"}]},
            ],
        },
        {
            "uuid": "c1a2b3c4-0000-4000-8000-000000000002",
            "name": "",
            "created_at": "2025-06-10T19:00:00Z",
            "chat_messages": [
                {"uuid": "m-101", "sender": "human",
                 "content": [{"type": "text", "text": OWN_EN}],
                 "created_at": "2025-06-10T19:00:01Z"},
                {"uuid": "m-102", "sender": "assistant",
                 "content": [{"type": "text", "text": "Late autumn is fine for a fig."}],
                 "created_at": "2025-06-10T19:00:09Z"},
            ],
        },
    ]


def _node(nid, parent, children, role, parts, t, ctype="text", **extra):
    message = None
    if role is not None:
        message = {"id": nid, "author": {"role": role, "name": None},
                   "content": {"content_type": ctype, "parts": parts},
                   "create_time": t, "status": "finished_successfully", **extra}
    return {"id": nid, "message": message, "parent": parent, "children": children}


def chatgpt_export() -> list[dict]:
    """One invented ChatGPT conversation with an edited (abandoned) branch."""
    t = 1714550400.0  # 2024-05-01 08:00 UTC
    mapping = {
        "root": _node("root", None, ["sys"], None, None, None),
        "sys": _node("sys", "root", ["u1"], "system", [""], None),
        "u1": _node("u1", "sys", ["a1"], "user",
                    ["How should I pace a first half marathon if I usually run 10k in 55 "
                     "minutes?", {"content_type": "image_asset_pointer", "asset": "x"}],
                    t, ctype="multimodal_text"),
        "a1": _node("a1", "u1", ["u2-old", "u2"], "assistant",
                    ["Aim for about 6 minutes per kilometre."], t + 20),
        "u2-old": _node("u2-old", "a1", ["a2-old"], "user", ["ABANDONED owner question"],
                        t + 60),
        "a2-old": _node("a2-old", "u2-old", [], "assistant", ["ABANDONED reply"], t + 70),
        "u2": _node("u2", "a1", ["tool-call"], "user",
                    ["And what should I eat the night before? I do not like pasta much."],
                    t + 90),
        "tool-call": _node("tool-call", "u2", ["tool-out"], "assistant", ["search(...)"],
                           t + 95, ctype="code", recipient="browser"),
        "tool-out": _node("tool-out", "tool-call", ["a2"], "tool", ["results"], t + 96),
        "a2": _node("a2", "tool-out", [], "assistant",
                    ["Rice or potatoes work just as well."], t + 100, recipient="all"),
    }
    return [{"id": "6a1b-gpt-conv-0001", "title": "Half marathon pacing", "create_time": t,
             "update_time": t + 100, "mapping": mapping, "current_node": "a2",
             "moderation_results": []}]


def _parse(tmp_path, owner, data, name="conversations.json"):
    path = write(tmp_path / name, json.dumps(data))
    return parse_with(AssistantAdapter(), path, tmp_path, SRC, owner)


# ------------------------------------------------------------------- formats


def test_claude_export_owner_and_assistant_turns(tmp_path, owner):
    items, report = _parse(tmp_path, owner, claude_export())
    assert [i.meta["role"] for i in items] == ["me", "assistant", "me", "me", "assistant"]
    assert all(i.source == "assistant" and i.kind == Kind.ASSISTANT for i in items)
    assert all(i.id.startswith("ai-") for i in items)
    first, reply, thanks, en, _ = items
    assert first.text == OWN_FR  # content blocks win over the flat text
    assert first.is_self and first.author == "Camille Martin"
    assert not reply.is_self and reply.author == "Claude"
    assert "hidden" not in reply.text  # thinking blocks are not text
    assert thanks.text == "Merci, je vais faire des wraps."  # no text block: flat text
    assert thanks.meta["attachments"] == 1
    assert first.container == "Pique-nique au canal (c1a2b3c4)"
    assert first.title == "Pique-nique au canal"
    assert en.container == "c1a2b3c4-0000-4000-8000-000000000002"  # no title: the id
    assert first.timestamp == datetime(2025, 5, 2, 8, 0, 5)
    assert [i.seq for i in items[:3]] == [0, 1, 2]
    assert report.items == 5 and not report.warnings


def test_chatgpt_export_follows_the_visible_branch(tmp_path, owner):
    items, report = _parse(tmp_path, owner, chatgpt_export())
    texts = [i.text for i in items]
    assert not any("ABANDONED" in t for t in texts)
    assert [i.meta["role"] for i in items] == ["me", "assistant", "me", "assistant"]
    assert texts[0].startswith("How should I pace")  # the image part is dropped
    assert texts[3] == "Rice or potatoes work just as well."
    assert [i.author for i in items] == ["Camille Martin", "ChatGPT"] * 2
    assert items[0].timestamp == datetime(2024, 5, 1, 8, 0)
    assert items[0].container == "Half marathon pacing (6a1b-gpt)"
    assert report.skipped.get("system") == 1
    assert report.skipped.get("tool") == 2  # the call to a tool and the tool's output
    assert not report.warnings


def test_chatgpt_without_current_node_takes_the_latest_leaf(tmp_path, owner):
    data = chatgpt_export()
    del data[0]["current_node"]
    items, _ = _parse(tmp_path, owner, data)
    assert items[-1].text == "Rice or potatoes work just as well."


def test_claude_branches_follow_the_current_leaf(tmp_path, owner):
    conv = {"uuid": "conv-branch", "name": "Branches", "current_leaf_message_uuid": "b2",
            "chat_messages": [
                {"uuid": "q", "sender": "human", "text": "first question here please",
                 "parent_message_uuid": "00000000-0000-4000-8000-000000000000"},
                {"uuid": "b1", "sender": "assistant", "text": "ABANDONED answer",
                 "parent_message_uuid": "q"},
                {"uuid": "b2", "sender": "assistant", "text": "kept answer",
                 "parent_message_uuid": "q"},
            ]}
    items, _ = _parse(tmp_path, owner, [conv])
    assert [i.text for i in items] == ["first question here please", "kept answer"]


# -------------------------------------------------------------------- inputs


def test_file_folder_and_zip_give_the_same_ids(tmp_path, owner):
    data = claude_export() + chatgpt_export()
    payload = json.dumps(data)
    bare = write(tmp_path / "bare" / "conversations.json", payload)
    folder = tmp_path / "folder"
    write(folder / "export-2025" / "conversations.json", payload)
    write(folder / "export-2025" / "users.json", json.dumps([{"uuid": "u", "full_name": "x"}]))
    zipped = tmp_path / "zip" / "data-export.zip"
    zipped.parent.mkdir()
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.writestr("conversations.json", payload)
        zf.writestr("projects.json", "[]")

    results = {}
    for name, root in (("bare", bare), ("folder", folder), ("zip", zipped.parent)):
        items, report = ingest_source(SRC, root, owner)
        assert not report.warnings, (name, report.warnings)
        results[name] = [i.id for i in items]
    assert results["bare"] == results["folder"] == results["zip"]
    assert len(results["bare"]) == 9

    _, report = ingest_source(SRC, folder, owner)
    assert report.skipped.get("other_file") == 1  # users.json


def test_zip_and_its_extracted_folder_together_are_deduplicated(tmp_path, owner):
    payload = json.dumps(claude_export())
    write(tmp_path / "export" / "conversations.json", payload)
    with zipfile.ZipFile(tmp_path / "export.zip", "w") as zf:
        zf.writestr("export/conversations.json", payload)
    items, report = ingest_source(SRC, tmp_path, owner)
    assert len(items) == 5
    assert any("duplicate" in w for w in report.warnings)


def test_ids_are_stable_across_reingest_and_renames(tmp_path, owner):
    first, _ = _parse(tmp_path, owner, claude_export())
    again, _ = _parse(tmp_path, owner, claude_export())
    assert [i.id for i in first] == [i.id for i in again]

    renamed = claude_export()
    renamed[0]["name"] = "Renamed later"
    moved, _ = _parse(tmp_path, owner, renamed, name="sub/conversations-new.json")
    assert [i.id for i in moved] == [i.id for i in first]


def test_messages_without_ids_get_content_hash_ids(tmp_path, owner):
    data = claude_export()
    for message in data[0]["chat_messages"]:
        del message["uuid"]
    data[0]["chat_messages"].append(dict(data[0]["chat_messages"][2]))  # same text twice
    a, _ = _parse(tmp_path, owner, data)
    b, _ = _parse(tmp_path, owner, data)
    assert [i.id for i in a] == [i.id for i in b]
    assert len({i.id for i in a}) == len(a)


def test_holdout_never_touches_assistant_turns(tmp_path, owner):
    write(tmp_path / "conversations.json", json.dumps(claude_export() + chatgpt_export()))
    items, _ = ingest_source(SRC, tmp_path, owner, holdout_percent=100)
    assert all(i.holdout for i in items if i.is_self)
    assert not any(i.holdout for i in items if not i.is_self)


# ----------------------------------------------------------------- malformed


def test_malformed_input_is_reported_not_raised(tmp_path, owner):
    good = claude_export()[0]
    data = [
        good,
        "not a conversation",
        {"uuid": "conv-bad-1", "name": "Bad", "chat_messages": "oops"},
        {"id": "conv-bad-2", "title": "Bad map", "mapping": ["not", "a", "dict"]},
        {"uuid": "conv-nothing", "name": "Neither"},
        {"uuid": "conv-partly", "name": "Partly", "chat_messages": [
            42, {"uuid": "p1", "sender": "human", "text": "a message that is fine"},
            {"uuid": "p2", "sender": "system", "text": "ignored"}]},
        {"id": "conv-gpt-loop", "title": "Loop", "current_node": "a",
         "mapping": {"a": {"message": None, "parent": "b"}, "b": {"message": None,
                                                                 "parent": "a"}}},
    ]
    items, report = _parse(tmp_path, owner, data)
    assert {i.container for i in items} == {"Pique-nique au canal (c1a2b3c4)",
                                            "Partly (conv-par)"}
    assert report.skipped["malformed_conversation"] == 4
    assert report.skipped["malformed_message"] == 1
    assert report.skipped["other_sender"] == 1
    assert report.skipped["empty_conversation"] == 1  # the loop holds no message
    assert any("conv-bad-1" in w for w in report.warnings)


def test_bad_files_are_warnings(tmp_path, owner):
    write(tmp_path / "conversations.json", "{ not json")
    write(tmp_path / "other" / "conversations.json", json.dumps({"hello": "world"}))
    (tmp_path / "broken.zip").write_bytes(b"not a zip")
    items, report = ingest_source(SRC, tmp_path, owner)
    assert items == []
    text = "\n".join(report.warnings)
    assert "not valid JSON" in text
    assert "no conversations in a known format" in text
    assert "not a readable zip" in text


# -------------------------------------------------------------------- pasted


def test_pasted_reason_heuristic():
    assert pasted_reason(OWN_EN) is None
    code = "```python\n" + "def f(x):\n    return x * 2\n" * 30 + "```"
    assert pasted_reason(code) == "pasted"
    assert pasted_reason("fix this please\n" + code) == "pasted"  # under 4 words of own
    assert pasted_reason("why does this fail?\n" + code) == "mostly pasted"
    small = OWN_EN + "\n\n```\nx = 1\n```\n\nAny idea?"
    assert pasted_reason(small) is None
    log = "\n".join(f"2025-01-0{n % 9 + 1} 12:00:0{n % 9} ERROR worker {n} timed out"
                    for n in range(40))
    assert pasted_reason("Can you explain this log to me?\n" + log) == "mostly pasted"
    quoted = "\n".join("> " + OWN_EN for _ in range(6))
    assert pasted_reason("What do you think of what she wrote?\n" + quoted) == "mostly pasted"
    article = ("The committee met on Tuesday to review the proposal in detail. " * 120)
    assert pasted_reason("Summarise this for me:\n\n" + article) == "very long"
    assert pasted_reason("https://example.org/a/very/long/link/to/something?x=1") == "pasted"
    assert pasted_reason("ok thanks") is None  # short, but nothing pasted


def test_code_flag_and_language_ignore_code(tmp_path, owner):
    text = (OWN_FR + "\n\n```bash\nfind . -name '*.tmp' -delete && echo done\n```\n\n"
            "Est-ce que cette commande est dangereuse pour mes photos ?")
    conv = {"uuid": "conv-code", "name": "Commande",
            "chat_messages": [{"uuid": "x1", "sender": "human", "text": text}]}
    items, _ = _parse(tmp_path, owner, [conv])
    (item,) = items
    assert item.text == text  # kept whole for search
    assert item.meta["code"] is True and "pasted" not in item.meta
    assert item.lang == "fr"
    assert "find ." not in strip_code_blocks(item.text)


def test_mostly_pasted_turns_are_flagged(tmp_path, owner):
    code = "```js\n" + "const a = b.map(x => x + 1);\n" * 20 + "```"
    conv = {"uuid": "conv-paste", "name": "Bug", "chat_messages": [
        {"uuid": "x1", "sender": "human", "text": "fix this\n" + code},
        {"uuid": "x2", "sender": "assistant", "text": "```js\n" + "y\n" * 50 + "```"}]}
    items, _ = _parse(tmp_path, owner, [conv])
    assert items[0].meta["pasted"] == "pasted"
    assert "pasted" not in items[1].meta  # only the owner's turns are judged


# ------------------------------------------------------- chunks and extraction


def _chunks(tmp_path, owner):
    write(tmp_path / "conversations.json", json.dumps(claude_export()))
    items, _ = ingest_source(SRC, tmp_path, owner)
    return items, chunk_items(items, ChunkingConfig())


def test_chunks_clip_assistant_turns_and_keep_owner_turns(tmp_path, owner):
    items, chunks = _chunks(tmp_path, owner)
    picnic = next(c for c in chunks if c.container.startswith("Pique-nique"))
    assert picnic.kind == Kind.ASSISTANT
    assert OWN_FR in picnic.text  # whole
    assert LONG_REPLY not in picnic.text
    assert "Claude: Here is a detailed plan" in picnic.text and "[…]" in picnic.text
    reply_part = picnic.text.split("Claude: ", 1)[1].split("\n\n")[0]
    assert len(reply_part) <= ASSISTANT_CONTEXT_CHARS + 4
    assert "Here is a detailed plan" not in picnic.self_text
    assert OWN_FR in picnic.self_text
    assert picnic.participants == ["Camille Martin", "Claude"]


def test_assistant_turns_are_never_evidence(tmp_path, owner):
    items, chunks = _chunks(tmp_path, owner)
    picnic = next(c for c in chunks if c.container.startswith("Pique-nique"))
    members = [i for i in items if i.id in picnic.item_ids]
    mine = next(i for i in members if i.is_self)
    theirs = next(i for i in members if not i.is_self)

    rendered = render_chunk(picnic, members, owner)
    assert f"[{theirs.id}]" in rendered and "assistant (context only)" in rendered
    assert LONG_REPLY not in rendered  # clipped as in the chunk
    assert "context only and never evidence" in rendered

    result = ExtractionResult(observations=[
        ExtractedObservation(subject_kind="topic", subject="picnic", evidence=[theirs.id],
                             text="I am planning a picnic for eight people."),
        ExtractedObservation(subject_kind="topic", subject="picnic",
                             evidence=[mine.id, theirs.id],
                             text="I am organising a picnic by the canal on Sunday."),
    ])
    kept, dropped = clean_observations(picnic, members, result)
    assert dropped == 1
    assert [o.evidence for o in kept] == [[mine.id]]
