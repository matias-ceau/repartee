# Repartee

Repartee is a mirror of its owner. It reads one person's own material (chat
archives, notes, journal, essays, interviews), builds a readable model of what
they lived, think and how they write, and then:

- **answers as them**, in the first person, from their own record;
- **drafts in their voice**, for them to proofread and send.

It runs in a terminal, for one person, on their own machine. The thinking is
done by a Claude model through the Anthropic API, or by any model on OpenRouter
or an OpenAI-compatible server.

> Status: the whole pipeline has run end to end on a fictional sample person
> through OpenRouter with a small model. It has **not yet been run on real
> data or against the Anthropic API**. See [What is verified](#what-is-verified).

## How it works

```
your material ──ingest──▶ searchable index ──distill──▶ self-model (markdown you can edit)
                                │                              │
                                └───────────── mirror ◀────────┘
                                        chat · draft · interview · review
```

1. **Ingest** turns WhatsApp exports, an Obsidian vault, journal, essays,
   medical notes, taste lists and exported conversations with AI assistants
   (Claude.ai, ChatGPT) into normalized items. No model involved.
2. **Store** indexes them for keyword and meaning search (SQLite plus a local
   multilingual embedding model).
3. **Distill** has the model read the material and write a **self-model**:
   pages about people, topics, tastes, behaviour, health and a timeline. Every
   line is a claim that points to the material it came from.
4. **Voice** measures how you write in each kind of text and language (message
   length, punctuation, emoji, pet phrases) and keeps real samples.
5. **The mirror** answers and drafts using the self-model, the raw material and
   the voice measurements. It says "I don't know" when it has nothing to go on.

The self-model is a folder of markdown you can open in Obsidian. Edit a line
and it becomes confirmed; delete a line and the mirror will not infer it again;
anything you write yourself is never touched.

## Your data stays out of this repository

- This repository holds code only. Your material and everything derived from it
  live in a separate **home** directory, meant to be its own private git
  repository.
- Repartee refuses to use a home placed inside another repository, and ships a
  pre-commit hook (`scripts/pre-commit`) that blocks chat exports, home data
  and API keys from being committed here.
- Your material **is sent to the model provider** (Anthropic, or OpenRouter and
  whoever serves the model there) for processing. Nothing is
  kept local-only. If that is not acceptable for some of your data, do not add
  it as a source.

## Quick start

```bash
uv sync                       # or: uv pip install -e .
export ANTHROPIC_API_KEY=...  # only needed for the steps that call the model
```

To use OpenRouter instead, set `OPENROUTER_API_KEY` and add to `repartee.yaml`:

```yaml
provider: openrouter
models: {bulk: z-ai/glm-5.3-flash, smart: z-ai/glm-5.3-flash, chat: z-ai/glm-5.3-flash}
```

Try it on the bundled fictional person first:

```bash
cp -r examples/persona ~/repartee-demo
repartee --home ~/repartee-demo ingest          # parse the sources (no model)
repartee --home ~/repartee-demo status
repartee --home ~/repartee-demo distill --dry-run   # what a full run would cost
repartee --home ~/repartee-demo sync            # distill and build the voice
repartee --home ~/repartee-demo                 # open the mirror
```

Then make your own home:

```bash
repartee init ~/repartee-self        # creates the folders and a git repository
$EDITOR ~/repartee-self/repartee.yaml    # your name, the names you appear under, your sources
export REPARTEE_HOME=~/repartee-self
repartee sync
repartee
```

Put WhatsApp exports (`_chat.txt` files, folders or zips) under
`sources/whatsapp/`, or point a source at an existing folder such as your
Obsidian vault. Claude.ai and ChatGPT data exports (the zip, its folder, or
`conversations.json`) go under a `type: assistant` source: only your own turns
count as evidence or voice, the assistant's are kept as clipped context, and
turns that are mostly pasted code, logs or text are left out of the voice.

## Commands

| Command | Does |
|---|---|
| `repartee init [PATH]` | create a home |
| `repartee ingest` | parse sources into the index |
| `repartee distill` | extract, consolidate, rebuild the summary and open questions |
| `repartee voice` | rebuild writing statistics and style cards |
| `repartee sync` | all of the above, incrementally |
| `repartee status` | what is indexed, what the self-model holds, what is pending |
| `repartee` | open the mirror (chat) |
| `repartee ask "..."` | one answer on stdout |
| `repartee draft "..." --to NAME` | one draft on stdout |
| `repartee interview` | let the mirror ask you questions to fill gaps |
| `repartee review` | confirm, edit or reject what it inferred |
| `repartee why CLAIM_ID` | a claim with the material behind it |
| `repartee eval voice` | blind test: can you tell your text from the mirror's? |
| `repartee eval recall FILE` | score answers against known facts |
| `repartee eval style` | how far the mirror's drafts are from your measured style |

Inside the mirror: `/chat`, `/draft @name ...`, `/accept`, `/interview`,
`/review`, `/eval`, `/sources`, `/quit`.

## Cost

Every tier uses `claude-opus-5-5` by default. The first distill of a large
archive is the expensive step: `repartee distill --dry-run` prints an estimate,
and a real run asks before spending more than a threshold. To spend less, set
`models.bulk` in `repartee.yaml` to a cheaper model for the extraction pass, or
run `repartee distill --batch`.

If Claude declines a request on safety grounds, the API retries it on an older
model at that model's price (Anthropic's server-side fallback).

## What is verified

- Over 550 automated tests, all offline, including one that runs the sample
  person end to end: ingest, distill, voice, a chat turn, a draft, an interview.
- The real commands were run by hand on a copy of the sample person, up to the
  point where a model is needed: ingest (with the real embedding model),
  status, the cost estimate, the statistics-only style cards.
- The model is replaced by a scripted fake in those tests.
- The sample person was run for real through OpenRouter with `z-ai/glm-5.3-flash`:
  1,162 items, 69 pages, about 550 claims, 17 open questions, style cards for two
  registers, for about $0.09 and 11 minutes. Answers were grounded in the material
  and said so when something never happened; a draft to a friend came out in her
  casual French register. On OpenRouter the reported cost is the provider's own
  figure; the dry-run estimate uses the cheapest listed price and can be several
  times low.
- Prompts, costs and answer quality against the Anthropic API are untested.
- The WhatsApp parser handles the common export variants in several languages
  and reports any line it cannot place, but it was written without a real
  export to check against. Run `repartee ingest` and read the report.

## Development

```bash
uv sync
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
ln -s ../../scripts/pre-commit .git/hooks/pre-commit   # recommended
```

- Design: [docs/superpowers/specs/2026-11-14-mirror-design.md](docs/superpowers/specs/2026-11-14-mirror-design.md)
- Build plan: [docs/superpowers/plans/2026-11-14-mirror.md](docs/superpowers/plans/2026-11-14-mirror.md)
- Earlier direction (a general chat client): [docs/archive/](docs/archive/)

## License

See [LICENSE](LICENSE).
