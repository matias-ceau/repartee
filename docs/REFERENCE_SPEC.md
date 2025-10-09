# Repartee Reference Specification

This document is the **source of truth** for the Repartee project.  
It consolidates current architecture, workflows, dependencies, and project-specific constraints.

---

## 1. Overview

**Repartee** is a personal AI assistant platform designed primarily for Arch Linux desktop use.  
It supports multiple interaction modes **(CLI, TUI, REPL)**, persistent & contextual memory systems, and connections to multiple AI model providers, with **OpenRouter** as the preferred default gateway.

The design prioritizes:
- **Local control** and **privacy-first options** (including "private mode" with anonymization and/or local model-only execution).
- **Human-readable modular architecture** where **memory/attention/persona** form a single conceptual entity.
- Flexible integration for **varied model providers** while only using essential Python libraries (e.g., `requests`, `asyncio`, `htmx` when applicable).
- Ability to spawn **specialized assistant swarms** for subprocess tasks.

---

## 2. Core Architecture

### 2.1 Runtime Modes
- **CLI mode** — for one-shot queries and interactive conversations (primary entrypoint through `src/repartee/main.py` → `CLI` class).
- **TUI modes**:
  - **fzf_tui** *(placeholder – empty file, to be developed)*.
  - **textual_tui** *(placeholder – empty file, to be developed)*.
- **REPL mode** — planned interactive shell.

### 2.2 Memory & Context System
- **Short-Term Memory (`ShortTermMemory`)**
  - Tracks immediate conversation context (traditional "chat history").
  - Token-aware trimming with `tiktoken`.
- **Episodic Memory (`EpisodicMemory`)**
  - Long-term historical record via SQLite + OpenAI embeddings. <// TODO: dont be too specific about which embeddings
  - Semantic search over past conversations.
- **Semantic Memory (`SemanticMemory` / `KnowledgeGraph`)**
  - Structured knowledge graph storage with concepts/relations. 
  - Supports importing from Obsidian vaults and semantic search.
- **Working Memory (`WorkingMemory`)**
  - Scratch space for active context requiring temporary retention. <// TODO: this is a bit vague, need to define better and also probably should be an MCP dynamic context

**Context Strategy**
- Memory modules can adapt their window size depending on:
  1. Model’s token limits.
  2. Task requirements (e.g., summarization vs detailed reasoning).
- Retrieval is dynamic based on **relevance heuristics**.

---

## 3. Providers & Model Layer

### 3.1 Current Integrations
- **OpenAI**
- **Anthropic**
- **Google Gemini**
- **(Placeholder)** Perplexity
- **(Planned)** OpenRouter as universal abstraction for routing calls.

### 3.2 Design Constraints
- Integration modules should:
  - Avoid heavy external SDK dependencies (prefer raw HTTP calls via `requests`/`httpx`/`asyncio`).
  - Support pluggable response streaming.
  - Allow capabilities introspection (context length, price, speed).
  - Provide **fallbacks** and **failover** where possible.
  
---

## 4. Configuration System
- Primary configuration is in `config/settings.yaml`, with layered overrides from:
  1. System config (`~/.config/repartee/settings.yaml`)
  2. Data directory config (`$XDG_DATA_HOME/repartee/settings.yaml`)
- API Keys come **only** from environment variables or `.env` (never committed).
- Default embeddings: OpenAI `text-embedding-3-small`. <// TODO: add local embeddings option for offline search, actually maybe
- Data location follows **XDG Base Directory spec**.

---

## 5. Logging, Error Handling, Testing

### 5.1 Logging
- Logging system **not yet implemented** (priority from `docs/ROADMAP.md`).
- Should provide:
  - Per-session logs in `$XDG_DATA_HOME/repartee/logs`.
  - Categories: info, warn, error, debug.

### 5.2 Error Handling
- Critical sections lacking retries/backoff (priority fix for model API calls).
- Roadmap calls for typed response objects and error case typing.

### 5.3 Testing
- Unit tests exist for some model integrations.
- Missing:
  - Integration tests.
  - Memory retrieval tests.
  - Edge case/error condition tests.

---

## 6. Missing or Placeholder Systems

1. **Daemon Mode**
   - Not implemented. Would be responsible for:
     - Background context retention.
     - Receiving tasks from multiple interfaces (CLI, HTTP, socket).
     - Event loop handling for async operations.
<// TODO: features such as a socket and IPC options such as localhost api, hooks (socket maybe), responding to text sent to fifo etc (file watch)  
2. **Private Mode / Local Model Routing**
   - Not implemented. Design target:
     - Option to run only local models (Ollama, Transformers, etc.).
     - Mixed-mode: anonymize sensitive data locally, send rest to remote LLM.
3. **Swarm Assistants**
   - Not implemented. Target: on-demand subprocess creation for specialized goals.
4. **FZF TUI / Textual UI**
   - Files exist but are empty.
5. **Model Registry/Factory**
   - Allows unified interface & provider switching.
6. **OpenRouter Integration**
   - Should unify provider calls and capability detection.

---

## 7. Project Constraints & Philosophy
- **Personal-use optimized**: Not built for general commercial deployment.
- **Arch Linux specific quirks** allowed and utilized.
- Keep **developer-facing layout intuitive** — minimal abstraction layers unless they serve clear modularity.
- The assistant **should feel like a single personality with modulated awareness** based on context and active tasks.

---

## 8. Next Steps (from analysis + ROADMAP alignment)
1. Implement logging system (incl. async support).
2. Implement retry/error handling for API integrations.
3. Add token counting to model integrations.
4. Create model registry & provider abstraction (start with OpenRouter).
5. Implement daemon mode with background session control.
6. Add "Private Mode" flag with local anonymization pipeline.
7. Flesh out missing TUIs.
8. Expand test coverage.

---