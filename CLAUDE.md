# Repartee Project Reference

Repartee is a mirror of its owner: it ingests one person's own material,
distils a readable self-model with evidence, and runs a terminal agent that
answers and drafts in that person's voice. Design:
`docs/superpowers/specs/2026-11-14-mirror-design.md`.

## Build Commands
- Install: `uv sync` (or `uv pip install -e . --group dev`)
- Tests: `.venv/bin/python -m pytest -q` (offline; `-m live` needs a key or the embedding model)
- Lint: `.venv/bin/ruff check src tests`
- Run: `repartee` (TUI), `repartee --help` for commands
- Try on sample data: copy `examples/persona` somewhere outside this repo and pass `--home`

## Hard Rule: No Personal Data In This Repository
- This repository is public. The owner's material and everything derived from
  it live in a separate private "home" (`repartee init PATH`).
- Never commit chat exports, journal or medical text, a `repartee.yaml`, or
  keys. `scripts/pre-commit` and `repartee.guard` enforce this; do not weaken them.
- All fixtures are fictional (`examples/persona` is an invented person).

## Project Structure
- `schema.py`, `config.py`, `home.py`: shared types, settings, home layout
- `ingest/`: WhatsApp, markdown, tastes, transcripts → items and chunks (no model)
- `store/`: SQLite index, keyword and embedding search
- `selfmodel/`: markdown pages of claims, owner-edit reconciliation, validated operations
- `distill/`: extract → route → consolidate → core → gaps (model calls)
- `voice/`: style statistics, style cards, sample picking, drafter
- `llm/`: the model protocol, Anthropic client, scripted fake, cache
- `mirror/`: the agent loop, tools, chat / draft / interview / review
- `evals/`: blind voice test, style distance, recall
- `runtime.py`, `sync.py`: wiring; `cli.py`, `tui/`: interfaces
- `scaffold.py`, `guard.py`, `gitops.py`: creating a home, keeping data out of this repo

## Code Style & Architecture
- Python 3.12+
- Dependencies managed with uv
- Object-oriented for clarity but minimalistic (avoid unnecessary abstraction)
- Follows PEP8 conventions; line length 100
- Open to integration with higher-level libraries when appropriate
- Willing to integrate other languages (bash, cython, rust) for performance or convenience
- Every model call goes through `llm/base.py`; callers name a tier and a purpose, never a model
- Tests use `FakeLLM` and `HashEmbedder`; nothing in the default test run touches the network

## Invariants Worth Knowing
- Item ids are stable: chat messages by content, markdown by location. Evidence links depend on it.
- The pipeline changes the self-model only through `ClaimOp`s; it can never alter
  a claim the owner confirmed, or anything the owner wrote.
- Loading and saving a self-model page with no operations is byte-identical.
- The mirror's own words are never evidence and never a voice sample.
- Holdout items are never used for style statistics, cards or samples.
- Nothing that changes per request goes in a system prompt (prompt caching).

## Configuration Approach
- API keys stored in environment variables only (never in config files)
- Owner identity, sources, model and embedding choices in `<home>/repartee.yaml`
- Follow XDG specification for file locations:
  - Config in XDG_CONFIG_HOME (`repartee/config.yaml`, only says where the home is)
  - Home defaults to XDG_DATA_HOME/repartee
  - Cache (index, model response cache, embedding model) in XDG_CACHE_HOME
  - No sensitive data in config files that will be in Git repo

## Model Preferences
- Anthropic only for now; every tier (`bulk`, `smart`, `chat`) defaults to `claude-opus-5-5`
- Tiers are set per home in `repartee.yaml`; a cheaper `bulk` model cuts extraction cost
- Before writing code that calls the Claude API, load the `claude-api` skill: request shapes change
- Future: other providers and local models behind the same protocol

## Memory System Design
- A compact core summary is always in context; everything else is fetched on demand through tools
- Avoid large system prompts at initialization
- Facts and events must come from a lookup; "I don't know" is a normal answer
- Support for temporal queries (date filters, timeline pages, claim validity dates)

## Later
- An MCP server exposing the mirror's tools to other agents
- Importers for exported assistant conversations and email
- A fine-tuned model for casual registers, if the blind test calls for it
