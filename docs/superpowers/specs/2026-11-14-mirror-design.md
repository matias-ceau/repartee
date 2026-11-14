# Repartee Mirror — design

Date: 2026-11-14
Status: approved direction, architecture decided by the builder (see "Calls made" at the end)

## 1. Intent

Repartee becomes a **mirror of its owner**: one person's memory, behaviour and
writing voice, built from their own material. The program and all the data stay
on the owner's machine; the thinking is done by a cloud model.

The owner uses it for two things:

- **Talking with it.** It answers in the first person, as the owner would, from
  what the owner actually lived, wrote and thinks.
- **Writing.** It drafts texts in the owner's voice (a message to a friend, an
  email, an essay passage). The owner proofreads everything before it goes
  anywhere.

It replaces the previous goal of a general multi-provider chat client, which no
longer fills a gap.

## 2. Goals, non-goals, anti-goals

**Goals**

- Build a readable, correctable **self-model** from: WhatsApp archives, own
  writing and essays, an Obsidian vault, a journal, medical observations, music
  and other tastes, and long interviews.
- Know facts (what happened, who people are), stances (what the owner thinks),
  behaviour (how the owner reacts and decides) and voice (how the owner writes,
  per language and per kind of text).
- Every claim in the self-model points to the material it came from.
- The owner can read the self-model in Obsidian and fix it by editing text.
- The mirror interviews the owner to fill gaps.
- Measure voice with a blind test against the owner's real texts.
- Work incrementally: new material updates the model without redoing everything.

**Non-goals (v1)**

- Multiple users, hosting, or exposing the mirror to other people. One owner,
  one terminal.
- Audience or sharing rules. The owner is the only reader and the only filter.
- Spoken voice (audio). Writing voice only.
- Fine-tuning a model. Voice comes from measured style plus real samples.
- Providers other than Anthropic. The LLM layer is a small protocol, so another
  provider is one file later.
- Media in chats (images, audio, video). Text only.

**Anti-goals** — it has failed if:

- Personal data ends up in the public code repository.
- It invents memories. Saying "I don't know" must be normal.
- The self-model is a black box the owner cannot read or correct.
- A re-run of the pipeline overwrites something the owner wrote or confirmed.

## 3. Constraints

- Python 3.12+, managed with uv. Object-oriented where it clarifies, minimal
  otherwise. PEP 8.
- Code is public; all personal data lives in a separate **private** repository
  (the "home"). XDG locations for config and cache.
- The owner accepts that every kind of material (chats, journal, medical notes)
  is sent to the model provider for processing. The one privacy requirement is
  that nothing personal reaches the public repository. There is no "local
  only" class of data in v1.
- One machine writes to a home at a time. The owner may push and pull the home
  between machines by hand; the cache is per machine and rebuilds itself.
- API keys only in environment variables.
- Must be buildable and testable with **no real data and no API key**: a
  fictional sample persona, a scripted fake model, a deterministic fake embedder.
- Languages are not assumed. Everything text-related is Unicode-aware and works
  for mixed-language material.

## 4. Approaches considered

1. **Search only.** Index everything, pull snippets into each conversation.
   This is what the old code did. Least work, but there is no stable picture of
   the person: stances drift between conversations and the writing sounds like
   a generic model quoting notes.
2. **Written self-model plus search (chosen).** A pipeline distils the sources
   into readable pages of claims with evidence; the runtime loads a compact
   core, looks up pages and raw material on demand, and imitates real samples
   of the owner's writing. Trade-off: most of the effort is distillation
   quality, and voice is imitated, not trained in.
3. **Fine-tuned model.** Best casual voice, but weaker reasoning, needs a GPU
   and retraining, still needs search for facts, and a wrong belief cannot be
   fixed by editing a file. Kept as a possible later add-on for casual
   registers only, if the blind test says imitation is not good enough.

The owner chose 2.

## 5. System overview

```
 sources (private)            derived (private)                 runtime
┌──────────────────┐  ingest ┌──────────────┐  search   ┌─────────────────────┐
│ WhatsApp exports │────────▶│ corpus index │◀──────────│ mirror agent        │
│ Obsidian vault   │         │ items+chunks │           │  modes: chat, draft,│
│ journal, essays  │         │ FTS + vectors│           │  interview, review  │
│ medical, tastes  │         └──────┬───────┘           │  tools: search_self,│
│ interviews       │                │ distill (LLM)     │  search_memory, ... │
└──────────────────┘                ▼                   └────────┬────────────┘
        ▲                    ┌──────────────┐  read/cite         │
        │ interviews,        │ self-model   │◀───────────────────┤
        │ sessions, drafts   │ markdown +   │                    ▼
        └────────────────────│ claims       │            TUI / CLI (owner)
                             └──────┬───────┘
                                    │ voice (stats + LLM)
                                    ▼
                             style cards, exemplars
```

Four layers, each usable on its own:

- **Ingest** turns each source into normalized `Item`s and `Chunk`s. Pure, no LLM.
- **Store** indexes chunks for keyword and meaning search.
- **Self-model** is a folder of markdown pages; the **distill** pipeline writes
  to it through validated operations; the owner edits it freely.
- **Mirror** is the runtime agent with tools over the store and the self-model,
  plus the **voice** subsystem for drafting.

## 6. Where things live

### 6.1 The home (private repository)

Resolved in this order: `--home` option, `$REPARTEE_HOME`, `home:` in
`$XDG_CONFIG_HOME/repartee/config.yaml`, else `$XDG_DATA_HOME/repartee`.

```
<home>/
  repartee.yaml        owner identity, source list, model and embedding settings
  sources/             raw exports the owner drops in (whatsapp/, writing/, ...)
  interviews/          interview transcripts (written by the mirror, also a source)
  conversations/       logs of sessions with the mirror (also a source)
  drafts/              each accepted draft: what the mirror wrote, what the owner kept
  self/                the self-model (an Obsidian-compatible folder)
  derived/             durable pipeline outputs: observations, voice statistics, state
  evals/               blind-test and recall results over time
  .gitignore
```

A source may also point at a path outside the home (for example the live
Obsidian vault) instead of being copied in.

`repartee init` refuses a directory that already has files but no
`repartee.yaml` (for example data left by the old Repartee), unless forced. It
does not migrate old data.

Everything in the home is text (markdown, YAML, JSON, JSONL), so it versions
well in git. When `git.autocommit` is on (default), Repartee commits the home
after each pipeline run, interview, and review session. It never pushes.

### 6.2 Cache (rebuildable)

`$XDG_CACHE_HOME/repartee/<hash of home path>/`: the SQLite corpus index with
embeddings, and the LLM response cache. Deleting it loses nothing that cannot
be recomputed. Pipeline outputs that cost money to recompute (extracted
observations) are **not** here; they are in `<home>/derived/`.

### 6.3 `repartee.yaml`

```yaml
owner:
  name: Camille Martin
  aliases: [Camille, Cam]        # author names in exports that mean "me"
sources:
  - type: whatsapp
    path: sources/whatsapp
  - type: vault
    path: ~/Documents/vault      # may live outside the home
    exclude: ["Templates/**"]
    kinds:                       # glob -> kind; frontmatter `kind:` wins over this
      "Journal/**": journal
      "Health/**": medical
      "Essays/**": essay
  - type: writing
    path: sources/writing
  - type: tastes
    path: sources/tastes
models:   {bulk: claude-opus-5-5, smart: claude-opus-5-5, chat: claude-opus-5-5}
effort:   {bulk: low, smart: high, chat: medium}
embeddings: {enabled: true, model: <small multilingual model>}
git: {autocommit: true}
```

## 7. Data model

Defined once in `repartee/schema.py` (pydantic). Summary:

- **Item** — the smallest unit from a source: one chat message, one note
  section, one journal entry, one interview turn. Fields: `id`, `source`,
  `kind`, `container` (chat name, note path), `author`, `is_self`, `timestamp`,
  `text`, `title`, `lang`, `seq`, `holdout`, `meta`.
- **Chunk** — the unit for search and for extraction: a chat session, a note
  section, a journal entry. Fields: `id`, `item_ids`, `source`, `kind`,
  `container`, `start`, `end`, `participants`, `text` (rendered for reading),
  `self_text` (only what the owner wrote), `title`.
- **Observation** — one thing the extractor noticed in one chunk, written in
  the first person, with a subject (`person: Ana`, `topic: urban cycling`), an
  optional date, a confidence and the item ids that support it.
- **Claim** — one line of the self-model: `id`, `page`, `section`, `text`,
  `status` (`inferred`, `confirmed`, `rejected`), `origin` (`pipeline`, `user`,
  `interview`), `confidence`, `evidence` (item ids), `valid_from`, `valid_to`.
- **ClaimOp** — the only way the pipeline changes the self-model: `add`,
  `support`, `revise`, `supersede`, `question`, `create_page`.

**Ids are stable.** An item id is a short prefix plus a hash of its natural
key, never a row number.

- Chat messages never change, so their key includes the text: chat, timestamp,
  author, text, and an ordinal among identical messages. Re-importing a longer
  export of the same chat gives the same ids to the same messages.
- Notes, journal entries and other markdown get edited, so their key is their
  **location**: path, heading trail, and an ordinal. Editing the text keeps the
  id; the store notices the changed content and the chunk is extracted again.
- When an id disappears and an item with identical text appears under a new id
  (a renamed file), the store reports the pair and the self-model's evidence
  links are rewritten to the new id.

**When source material is deleted**, evidence ids that no longer exist are
dropped from claims at the next ingest. An `inferred` claim left with no
evidence is removed (and logged), so deleting something makes the mirror forget
what it only knew from there. `confirmed` claims stay: the owner vouched for them.

**Holdout.** A fixed 5% of the owner's own items (chosen by hash) is flagged
`holdout`. Those are never used for style statistics, style cards, samples or
reply pairs (memory search is unaffected).
They exist only so the blind voice test compares against texts the mirror has
not imitated from.

## 8. Ingest

Each adapter is a class with `discover(config) -> files` and
`parse(file) -> Iterable[Item]`. No LLM, no network.

- **WhatsApp** (`_chat.txt` exports, loose or inside a folder or zip). Handles
  the iOS bracket format and the Android dash format, 12- and 24-hour clocks,
  2- and 4-digit years, multi-line messages, invisible direction marks, system
  lines, media placeholders (dropped), deleted and edited markers. Day/month
  order is inferred per file from the values seen and can be forced in config.
  The chat name comes from the file or folder name. `is_self` comes from
  `owner.aliases`. System lines and placeholders are recognised from a table of
  known wordings in several languages; a line that matches nothing is kept as
  ordinary text, and the parse report counts what it could not place, so real
  exports show where the table needs extending.
- **Markdown** (Obsidian vault, journal, essays, medical notes, interview
  transcripts). Reads frontmatter, tags and wikilinks. Kind comes from
  frontmatter `kind:`, else the config glob rules, else `note`. Dates come from
  frontmatter, else a date in the file name, else none. Any path that resolves
  inside the home is skipped, so the self-model is never read back as a source
  even when `self/` is linked into the owner's vault.
- **The home's own folders** (`interviews/`, `conversations/`, `drafts/`) are
  always ingested, without being listed as sources. Transcripts use
  speaker-labelled turns (`**me:**` and `**mirror:**`) and become one item per
  turn. **Only the owner's turns count**: the mirror's turns are kept as
  context, flagged as not the owner's, and are never evidence for a claim and
  never a voice sample. From a draft file, only the owner's final text is used.
- **Tastes** (music and similar): CSV or JSON listening/library exports and
  plain markdown lists, normalized to `taste` items.

**Chunking**

- Chats: split by silence. A new chunk starts after a gap of 3 hours
  (configurable) or when the chunk passes about 1,500 tokens. Rendered as
  `[2021-03-12 14:05] Ana: ...` lines.
- Notes and essays: by headings, merged or split to stay near 800 tokens.
- Journal: one entry per chunk, split if very long.

**Language** is detected per item with a light offline detector and stored in
`lang`; short messages inherit the dominant language of their chunk.

## 9. Store and retrieval

One SQLite file in the cache directory.

- Tables for items and chunks, plus an FTS5 index over chunk text (Unicode
  tokenizer, diacritics folded).
- Embeddings for chunks from a local multilingual model (fastembed, ONNX, no
  GPU needed), stored as float32 blobs and searched by brute force with numpy.
  That is fast enough for the tens of thousands of chunks a lifetime of text
  produces.
- **Hybrid search**: keyword and meaning results merged by reciprocal rank
  fusion. Filters: source, kind, container, participant, author (self / other),
  date range. If embeddings are disabled or the model is unavailable,
  search degrades to keyword only and says so in `status`.
- The self-model's claims are indexed the same way so `search_self` can find
  lines, not just pages.
- Reply pairs (what the other person said, what the owner answered) are stored
  with an embedding of the incoming context, for "similar situation" lookups.

Embedding is behind an `Embedder` protocol with a real implementation and a
deterministic hash-based one for tests.

## 10. The self-model

### 10.1 Layout

```
self/
  core.md              always in context: who I am, compact (about 1,500 tokens)
  people/<slug>.md     one page per person who matters
  topics/<slug>.md     what I think and know about a subject
  tastes/<slug>.md     music, books, food, ...
  behaviour/<slug>.md  patterns: "when X, I tend to Y"
  timeline/<year>.md   dated events
  health/<slug>.md     medical observations
  voice/<register>.md  style cards (section 12)
  open-questions.md    what the mirror does not know yet; drives interviews
  .claims/             claim metadata, one JSON file per page (hidden in Obsidian)
```

### 10.2 Page format

```markdown
---
kind: person
title: Ana Ruiz
aliases: [Ana, Anita]
updated: 2026-11-14
---
# Ana Ruiz

## Who
- Friend from university; we met in Lyon in 2014. ^c-3fa91c20
- Works as a nurse, mostly night shifts. ^c-77be01aa

## How we talk
- We write in French, in short bursts, with a lot of teasing. ^c-0d4e9a13

## Notes
Anything I write here myself. The pipeline never touches it.
```

- A **claim** is a single-line list item ending in an Obsidian block id
  `^c-<8 hex>`. Claims are written in the first person.
- Each page has a fixed set of sections per kind that the pipeline may write
  to (for a person: Who, Relationship, How we talk, Shared history, Earlier).
  Everything else on the page belongs to the owner.
- Metadata for each claim (status, origin, confidence, evidence, validity
  dates, a hash of the text) lives in `self/.claims/<kind>/<slug>.json`, so the
  markdown stays clean.
- **Round-trip rule:** loading a page and saving it without operations yields
  the same bytes.

### 10.3 How the owner corrects it

Reconciliation runs whenever the self-model is loaded, by comparing pages with
their metadata:

| The owner did this in the markdown | Result |
|---|---|
| Edited the text of a claim | Claim becomes `confirmed`, origin `user`. The pipeline can no longer change it. |
| Added a list item with no id in a pipeline section | Gets an id; `confirmed`, origin `user`. |
| Deleted a claim line | Recorded as `rejected` with its text and evidence, so it is not inferred again. |
| Moved a claim line to another page | Follows the id; nothing else changes. |
| Wrote anything outside claim lines | Preserved untouched. |

The `review` mode does the same through the TUI: confirm, edit or reject, one
claim at a time, lowest confidence first.

Evidence is not shown in the markdown. `repartee why <claim-id>` and the review
mode print a claim with excerpts of the material behind it.

### 10.4 What the pipeline may do

Only through `ClaimOp`s, applied and validated by code:

- `add` a claim (to a known section, with at least one existing evidence id).
- `support` an existing claim with more evidence (raises confidence).
- `revise` the wording of an `inferred` claim.
- `supersede` an `inferred` claim when things changed over time: the old one
  gets `valid_to` and moves to the page's "Earlier" section; a new one is added.
- `question`: append to `open-questions.md` (used when new evidence
  contradicts a `confirmed` claim, or when something is unclear).
- `create_page`.

Rejected by validation: touching `confirmed` or `rejected` claims, unknown
evidence ids, re-adding text that matches a rejected claim, writing outside
pipeline sections. Rejected operations are logged, not fatal.

### 10.5 `core.md`

Has a `## Pinned` section the owner writes, and a generated block between
`<!-- repartee:generated:start -->` and `<!-- repartee:generated:end -->` that
the pipeline rewrites from the strongest claims. Both are loaded into every
conversation.

## 11. Distill pipeline

Runs in stages; each is incremental and restartable.

1. **Extract** (model tier `bulk`). For each chunk not yet processed, the model
   returns a list of `Observation`s as structured output. Guidance in the
   prompt: only things about the owner or about people as they relate to the
   owner; first person; one fact per observation; a date when the chunk gives
   one; cite the item ids. Results are stored as JSONL in `derived/`, keyed by
   chunk hash and prompt version, so a re-run costs nothing for unchanged chunks.
2. **Route.** Each observation's subject is matched to a page by slug, alias
   and fuzzy match. A new page needs at least three observations on the same
   subject (people, years and health subjects are exempt); the rest go to a
   per-kind catch-all page. A small model call may merge near-duplicate labels.
3. **Consolidate** (tier `smart`). Per page: the current claims plus the new
   observations go in; a list of `ClaimOp`s comes out as structured output and
   is applied. Large pages are processed in batches.
4. **Core.** Regenerate the generated block of `core.md`.
5. **Gaps.** Look for thin, stale or contradictory areas and for kinds of
   knowledge with little coverage; write questions to `open-questions.md`.

Confidence starts from the kind of source (an interview answer or a journal
entry outweighs a passing chat message) and rises with independent evidence.

**Interview answers** produce `inferred` claims with origin `interview` and
high confidence: the model paraphrased them, so they are not yet the owner's
words. Each interview ends with a short review of the claims it produced, where
the owner confirms, edits or rejects them.

**Cost control.** `repartee distill --dry-run` prints chunk counts, estimated
tokens and estimated cost per stage. A real run prints the same estimate first
and asks before spending more than `distill.confirm_above_usd` (default 5). `--limit`, `--since` and `--source` narrow
a run. Extraction can go through the Message Batches API at half price when
latency does not matter.

**State** for stages 2 and 3 (which observations each page has consumed) is
kept as text in `derived/`.

## 12. Voice

A **register** is a kind of text in a language: `whatsapp.fr`, `whatsapp.en`,
`essay.en`, `journal.fr`, `note.en`. Per-person differences are handled by
sample selection, not by more registers.

1. **Statistics** (no LLM), from the owner's non-holdout items per register,
   and per recipient when there is enough text:
   - chat: words per message, messages per burst, lowercase starts, final
     punctuation, ellipses, repeated marks, emoji rate and favourites, laughter
     forms, letter stretching, openers and closers, language mixing;
   - long form: sentence and paragraph length, first-person rate, questions,
     connectors, markdown habits;
   - **distinctive phrases**: word sequences the owner uses far more than the
     people they talk to.
2. **Style card** (`self/voice/<register>.md`): the statistics rendered as
   plain statements, a short qualitative description written by the model from
   a sample, and a `## Notes` section the owner can write in ("never use
   semicolons").
3. **Samples at drafting time**: real texts by the owner in the same register,
   preferring the same recipient and a similar subject, plus **reply pairs**
   from similar past situations.
4. **Draft, then check.** The draft is measured with the same statistics. If
   it deviates clearly from the register (three sentences where the owner
   sends six words), the model revises once with the deviations listed.
5. **Feedback.** `/accept` saves the draft as accepted unchanged. `/accept`
   followed by text saves that text as what the owner actually used (edited in
   the TUI, or pasted back later). Both the mirror's version and the owner's
   final are kept in `drafts/`; only the final counts as a new sample.

Chat-register drafts are returned as a burst of separate messages when that is
how the owner writes.

Interface used by the mirror and the evals:

```python
class DraftRequest(BaseModel):
    intent: str                    # what to say, in the owner's words
    recipient: str | None = None   # person slug or name
    register_name: str | None = None  # inferred from recipient and channel if absent
    context: str | None = None     # e.g. the thread being answered
    variants: int = 1

class DraftResult(BaseModel):
    register_name: str
    variants: list[list[str]]      # each variant is one or more messages/paragraphs
    used: list[str]                # ids of samples and pages used
    deviations: list[str]          # what the style check still flags, if anything

class Drafter:
    async def draft(self, request: DraftRequest) -> DraftResult: ...
```

## 13. Mirror runtime

One agent loop with tools; modes change the instructions, not the machinery.

**Modes**

- `chat` — answers as the owner, first person.
- `draft` — produces a text for someone, through the `Drafter`.
- `interview` — asks the owner one question at a time (from
  `open-questions.md` and coverage gaps), follows up, saves the transcript to
  `interviews/`, then ingests and distils it straight away.
- `review` — walks through unconfirmed claims with their evidence.

**Context.** The system prompt is the mode's instructions plus `core.md`,
cached. Everything else is fetched on demand through tools, so start-up stays
small:

| Tool | Does |
|---|---|
| `search_self(query, kind?)` | finds claims and pages in the self-model |
| `read_page(path)` | returns one page |
| `search_memory(query, person?, source?, after?, before?, author?)` | finds chunks of raw material |
| `read_memory(chunk_id, around?)` | returns a chunk with its neighbours |
| `get_voice(register?, recipient?, topic?)` | style card and samples |
| `similar_replies(context, recipient?)` | what the owner answered in similar situations |
| `note_gap(question)` | records something the mirror could not answer |

**Grounding rules** (in every mode's instructions):

- Facts and events must come from a tool result. If nothing is found, say so,
  and call `note_gap`.
- Extending the owner's views to a new question is allowed, and is marked as a
  guess in the answer.
- `confirmed` claims outrank `inferred` ones; recent outranks old unless the
  question is about the past.

**Events.** The loop yields text deltas, tool calls and tool results. The TUI
shows the lookups beside the answer, so the owner sees what an answer rests on.

**Sessions** are saved to `conversations/` as markdown and ingested like any
other source; the owner's turns count as their own writing.

## 14. LLM layer

A small async protocol in `repartee/llm/base.py`:

- `complete(request) -> response` and `stream(request) -> events`.
- A request names a **tier** (`bulk`, `smart`, `chat`), not a model; tiers map
  to models and effort levels in config.
- Supports tool definitions, structured output from a pydantic schema, and
  cacheable system blocks.
- Assistant turns keep the provider's raw content so thinking blocks are
  replayed unchanged.

Implementations:

- `AnthropicClient` on the official SDK. Defaults to `claude-opus-5-5` on every
  tier. Adaptive thinking (parameter omitted), effort set explicitly per tier,
  no sampling parameters, structured outputs through `output_config.format`,
  tool choice left on `auto`, streaming for chat, prompt caching on the stable
  prefix, typed error handling, refusals checked on every response with the
  server-side fallback enabled, and the Batches API for bulk extraction.
- `FakeLLM` for tests: scripted or rule-based responses, records every request.
- A disk cache wrapper for the pipeline stages, keyed by the full request.

## 15. Interfaces

**CLI** (click):

```
repartee init [PATH]         create a home (folders, repartee.yaml, .gitignore, git init)
repartee ingest              parse sources into the index
repartee distill             extract, route, consolidate, core, gaps
repartee voice               rebuild statistics and style cards
repartee sync                ingest, distill, voice, incrementally
repartee status              counts, coverage, unconfirmed claims, open questions
repartee why CLAIM_ID        a claim with the material behind it
repartee                     open the TUI
repartee ask "..."           one answer on stdout
repartee draft "..." --to X  one draft on stdout
repartee interview | review | eval voice | eval recall
```

**TUI** (Textual), one screen:

```
┌ repartee · chat ───────────────────────────────┬ looked up ────────────┐
│ you   what did I think about moving in 2022?   │ topics/moving-abroad  │
│                                                │ journal 2022-03-14    │
│ me    I went back and forth all spring. In     │ wa: Ana, 2022-04-02   │
│       March I wrote that ...                   │                       │
│                                                │                       │
├────────────────────────────────────────────────┴───────────────────────┤
│ > /draft @ana tell her I can't make it Friday                          │
└────────────────────────────────────────────────────────────────────────┘
```

Slash commands switch mode: `/chat`, `/draft`, `/interview`, `/review`,
`/eval`, `/accept`, `/sources`, `/quit`. Review and the blind test are simple
keyboard-driven views in the same app.

## 16. Evaluation

- **Blind voice test** (`eval voice`): for held-out real texts, the model
  writes a neutral brief of what each says; the mirror drafts from the brief;
  the owner sees the real and the generated text in random order and picks the
  real one. Score per register, saved over time. 50% means indistinguishable.
- **Style distance**: automatic, the same statistics on generated versus real
  text. Cheap proxy between blind tests.
- **Recall** (`eval recall`): a YAML file of questions with expected facts;
  the mirror answers; a model grades. The sample persona ships with one.

## 17. Keeping personal data out of the public repo

- Repartee refuses to use a home that sits inside a git work tree whose remote
  is not the home's own (in practice: inside the code checkout), except the
  bundled sample persona.
- `.gitignore` in the code repo covers common export and data names.
- `scripts/pre-commit` (installable hook) blocks commits containing chat
  export files, lines shaped like chat export messages, `.env`, `.pypirc`, and
  common key formats.
- All fixtures are fictional and marked as such.

## 18. Code layout

```
src/repartee/
  schema.py  config.py  home.py  cli.py
  ingest/      whatsapp.py  markdown.py  tastes.py  chunking.py  lang.py
  store/       corpus.py  embed.py  search.py
  selfmodel/   pages.py  model.py  ops.py
  distill/     extract.py  route.py  consolidate.py  core.py  gaps.py  pipeline.py
  voice/       stylometry.py  registers.py  pairs.py  cards.py  drafter.py
  llm/         base.py  anthropic_client.py  fake.py  cache.py
  mirror/      tools.py  context.py  agent.py  modes.py  sessions.py
  evals/       voice_blind.py  style_distance.py  recall.py
  tui/         app.py
  guard.py     gitops.py
examples/persona/     fictional person: sources, repartee.yaml, recall questions
scripts/pre-commit
tests/
```

The old provider classes, memory classes, hand-rolled MCP stub and empty UI
files are removed. Earlier planning documents move to `docs/archive/`.

## 19. Testing

- Unit tests per module with small inline fixtures.
- The sample persona exercises ingest, store, voice statistics and the
  pipeline end to end with `FakeLLM` and the hash embedder. No network.
- Textual's test pilot drives the TUI headlessly.
- Anything needing a real model or the real embedding model is opt-in (marked,
  skipped by default).

## 20. Build order

1. Contracts: schema, config, home paths, LLM protocol.
2. In parallel: sample persona; ingest; store; LLM clients; self-model pages
   and operations; voice statistics and reply pairs; home init, guard, git.
3. In parallel: distill pipeline; style cards and drafter; mirror agent.
4. In parallel: CLI and sync; TUI; evals. Then an end-to-end test and docs.

## 21. Later, not now

- An MCP server exposing the same tools, so other agents can ask the mirror.
- Importers for exported assistant conversations and email.
- A fine-tuned model for casual registers, if the blind test calls for it.
- Reply prediction scored against held-out reply pairs.
- Spoken voice.

## 22. Calls made by the builder

The owner handed over the architecture. These were decided without asking and
are easy to change:

- Anthropic only in v1, `claude-opus-5-5` on every tier, with refusal fallback
  enabled. The `bulk` tier can be pointed at a cheaper model in config.
- Local embeddings (fastembed), keyword search always available.
- Claims carry Obsidian block ids; their metadata sits in a hidden sidecar folder.
- Claims are written in the first person.
- The home auto-commits to git locally and never pushes.
- Textual for the TUI, click for the CLI.
- Old code is deleted (it remains in git history); old docs are archived.
- `requirements.txt` is dropped in favour of `uv.lock`.

Settled after an independent read of this document:

- All material may be sent to the model provider; nothing is local-only.
- Markdown items are identified by location, chat messages by content; renamed
  files keep their evidence links; deleting a source makes the mirror forget
  what it only inferred from it.
- The mirror's own words are never evidence and never a voice sample.
- Interview answers become high-confidence `inferred` claims, confirmed in a
  short review at the end of the interview.
- `self/` stays in the home and is never ingested; evidence is shown by
  `repartee why` and review mode, not in the markdown.
- `init` refuses to take over a non-empty directory and migrates nothing.
- The WhatsApp parser is built for many export variants and reports what it
  could not parse, because no real export was available while building.
- No cost ceiling is imposed; runs show an estimate and ask above a threshold.
- One machine writes to a home at a time.
- Drafts are accepted as-is or with the text the owner actually used.
