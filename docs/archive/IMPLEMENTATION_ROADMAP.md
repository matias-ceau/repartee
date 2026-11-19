# Repartee Implementation Roadmap

This document provides a detailed, actionable roadmap for developing Repartee into a production-ready personal knowledge-aware AI assistant.

## Overview

**Vision**: Repartee becomes the go-to CLI assistant for knowledge workers who want AI that understands their personal notes, documents, and workflows.

**Timeline**: 6 months (24 weeks)  
**Effort**: ~200-300 hours of development  
**Milestones**: 5 phases, each building on the previous

---

## Phase 1: Foundation (Weeks 1-4)

**Goal**: Create a stable, well-tested core that users can rely on

### Week 1: Infrastructure & Testing

**Logging System** [Priority: HIGH]
```
- [ ] Create repartee/logging.py module
- [ ] Implement structured logging with levels (DEBUG, INFO, WARN, ERROR)
- [ ] Log to XDG_DATA_HOME/repartee/logs/repartee.log
- [ ] Add log rotation (max 10MB per file, keep 5 files)
- [ ] Add --verbose and --debug flags to CLI
- [ ] Log all API calls with timing and token counts
```

**Test Infrastructure** [Priority: HIGH]
```
- [ ] Set up pytest framework
- [ ] Create tests/unit/ and tests/integration/ directories
- [ ] Add test fixtures for mocked API responses
- [ ] Write tests for config.py (100% coverage goal)
- [ ] Write tests for memory modules (short_term, episodic)
- [ ] Add test for CLI basic functionality
- [ ] Set up test data fixtures
```

### Week 2: Error Handling & Robustness

**API Error Handling** [Priority: HIGH]
```
- [ ] Add retry logic with exponential backoff to all model calls
- [ ] Handle rate limiting (429 errors) gracefully
- [ ] Handle network errors with user-friendly messages
- [ ] Add timeout handling for long-running requests
- [ ] Implement fallback models when primary fails
- [ ] Log all errors with full context
```

**Configuration Validation** [Priority: MEDIUM]
```
- [ ] Validate config file structure on load
- [ ] Check API keys on startup (optional --skip-key-check flag)
- [ ] Provide helpful error messages for missing config
- [ ] Add config file schema documentation
- [ ] Support config file hot-reloading
```

### Week 3: Python Version & Dependencies

**Python Version Relaxation** [Priority: HIGH]
```
- [ ] Audit code for Python 3.13-specific features
- [ ] Test compatibility with Python 3.11 and 3.12
- [ ] Update pyproject.toml: requires-python = ">=3.11"
- [ ] Update documentation with version requirements
- [ ] Add Python version check in __main__.py
```

**Dependency Management** [Priority: MEDIUM]
```
- [ ] Audit all dependencies for necessity
- [ ] Pin versions for stability
- [ ] Add dependency update strategy (monthly review)
- [ ] Document optional dependencies (e.g., Ollama support)
- [ ] Create requirements-dev.txt for development tools
```

### Week 4: CI/CD & Documentation

**Continuous Integration** [Priority: HIGH]
```
- [ ] Set up GitHub Actions workflow
- [ ] Run tests on push to main and PRs
- [ ] Test on Python 3.11, 3.12, 3.13
- [ ] Add code coverage reporting (codecov.io)
- [ ] Add linting (ruff or black + flake8)
- [ ] Add type checking (mypy)
```

**Installation & Setup Docs** [Priority: HIGH]
```
- [ ] Write detailed INSTALL.md
- [ ] Document common installation issues
- [ ] Create quick start guide
- [ ] Add troubleshooting section
- [ ] Document environment variables
- [ ] Add FAQ section
```

**Phase 1 Success Criteria**:
- ✅ 80%+ test coverage
- ✅ All tests passing in CI
- ✅ Zero known crashes in normal usage
- ✅ Installation works smoothly on Linux, macOS

---

## Phase 2: User Experience (Weeks 5-8)

**Goal**: Polish the interface so users want to use Repartee daily

### Week 5: REPL Mode

**Interactive REPL** [Priority: HIGH]
```
- [ ] Implement full REPL in ui/repl.py using prompt_toolkit
- [ ] Add syntax highlighting for commands
- [ ] Add auto-completion for commands and model names
- [ ] Implement multi-line input support
- [ ] Add history navigation (up/down arrows)
- [ ] Save command history to XDG_DATA_HOME
- [ ] Add /commands (e.g., /help, /clear, /save, /load)
```

**REPL Features** [Priority: MEDIUM]
```
- [ ] /switch <provider> - change model on the fly
- [ ] /memory - show current memory state
- [ ] /search <query> - search past conversations
- [ ] /export <format> - export conversation
- [ ] /config - show current configuration
- [ ] /clear - clear screen
- [ ] /exit or Ctrl+D to quit
```

### Week 6: Textual TUI

**Basic TUI Layout** [Priority: HIGH]
```
- [ ] Create textual_tui.py with Textual app
- [ ] Implement 3-column layout: conversations | chat | context
- [ ] Add conversation list sidebar (left)
- [ ] Add main chat area (center)
- [ ] Add context panel (right) - shows memory, sources
- [ ] Add status bar with model name, token count
```

**TUI Interactivity** [Priority: MEDIUM]
```
- [ ] Keyboard shortcuts (Ctrl+N new chat, Ctrl+S save, etc.)
- [ ] Mouse support for scrolling and clicking
- [ ] Search conversations with Ctrl+F
- [ ] Dark/light theme toggle
- [ ] Responsive layout for different terminal sizes
```

### Week 7: Conversation Management

**Persistence** [Priority: HIGH]
```
- [ ] Save conversations to SQLite database
- [ ] Load previous conversations on startup
- [ ] Implement conversation listing (--list-conversations)
- [ ] Add conversation deletion (--delete-conversation <id>)
- [ ] Add conversation export (--export-conversation <id>)
- [ ] Support multiple concurrent conversations
```

**Search & Filtering** [Priority: MEDIUM]
```
- [ ] Search conversations by content
- [ ] Filter by date range
- [ ] Filter by model used
- [ ] Filter by tags (if implemented)
- [ ] Sort by relevance, date, or token usage
```

### Week 8: Context Awareness

**Visual Context** [Priority: HIGH]
```
- [ ] Show which memories are being retrieved
- [ ] Display token usage in real-time
- [ ] Show sources for knowledge-based answers
- [ ] Highlight relevant context chunks
- [ ] Add "why was this retrieved?" explanations
```

**Export Functionality** [Priority: MEDIUM]
```
- [ ] Export to Markdown with metadata
- [ ] Export to JSON for programmatic use
- [ ] Export to HTML with styling
- [ ] Support batch export (all conversations)
- [ ] Include embeddings in export (optional)
```

**Phase 2 Success Criteria**:
- ✅ TUI is usable and responsive
- ✅ REPL mode works smoothly for daily tasks
- ✅ Users can find past conversations easily
- ✅ Context retrieval is visible and understandable

---

## Phase 3: Knowledge Integration (Weeks 9-12)

**Goal**: Make Repartee the best tool for leveraging personal knowledge

### Week 9: Obsidian Integration Polish

**Vault Importing** [Priority: HIGH]
```
- [ ] Implement full Obsidian vault import in semantic_memory.py
- [ ] Parse Markdown files with frontmatter
- [ ] Extract links [[like this]] and #tags
- [ ] Build knowledge graph from wikilinks
- [ ] Handle attachments (images, PDFs) by indexing metadata
- [ ] Incremental updates (only process changed files)
- [ ] Add --import-obsidian <path> CLI command
```

**Metadata Support** [Priority: MEDIUM]
```
- [ ] Parse YAML frontmatter (title, date, tags, etc.)
- [ ] Store metadata in knowledge graph
- [ ] Use metadata for filtering and ranking
- [ ] Support custom frontmatter fields
- [ ] Index creation and modification dates
```

### Week 10: Markdown Directory Support

**Generic Markdown Indexing** [Priority: HIGH]
```
- [ ] Support any directory of Markdown files (not just Obsidian)
- [ ] Implement file watcher for real-time updates
- [ ] Handle nested directory structures
- [ ] Support .md and .markdown extensions
- [ ] Optionally index code blocks separately
- [ ] Add --watch-directory <path> command
```

**Content Processing** [Priority: MEDIUM]
```
- [ ] Extract headings for hierarchical structure
- [ ] Chunk long documents intelligently (by section)
- [ ] Preserve code blocks as-is (don't embed)
- [ ] Extract links and create graph edges
- [ ] Support for embedded images (OCR optional, future)
```

### Week 11: Semantic Search

**Knowledge Base Search** [Priority: HIGH]
```
- [ ] Implement semantic search over all indexed knowledge
- [ ] Add /search command in REPL and TUI
- [ ] Return top-k most relevant documents
- [ ] Show relevance scores
- [ ] Support filtering by file type, date, tags
- [ ] Cache search results for performance
```

**Automatic Context Injection** [Priority: HIGH]
```
- [ ] Analyze user query for keywords
- [ ] Retrieve relevant notes automatically
- [ ] Inject into prompt as context
- [ ] Show user which notes were used
- [ ] Allow user to disable auto-retrieval if desired
- [ ] Rank by relevance (embeddings similarity)
```

### Week 12: Knowledge Graph Features

**Graph Visualization** [Priority: MEDIUM]
```
- [ ] Export graph to DOT format (Graphviz)
- [ ] Create simple text-based graph view in TUI
- [ ] Show connections between concepts
- [ ] Highlight most connected nodes
- [ ] Add --visualize-graph command
- [ ] (Future: Interactive web-based viz with vis.js)
```

**Citation & Source Tracking** [Priority: MEDIUM]
```
- [ ] Track which knowledge was used in responses
- [ ] Add citations to answers (e.g., "Source: note.md")
- [ ] Allow clicking/navigating to source file
- [ ] Export with bibliography
- [ ] Show provenance trail for claims
```

**Phase 3 Success Criteria**:
- ✅ Can import 1000+ note Obsidian vault in <1 minute
- ✅ Semantic search returns relevant results
- ✅ Answers clearly reference source knowledge
- ✅ Knowledge graph has accurate connections

---

## Phase 4: Privacy & Local Models (Weeks 13-16)

**Goal**: Make local-first operation practical and appealing

### Week 13: Local Embeddings

**Sentence Transformers Integration** [Priority: HIGH]
```
- [ ] Add sentence-transformers dependency (optional)
- [ ] Implement local embedding provider
- [ ] Support all-MiniLM-L6-v2 (small, fast)
- [ ] Support all-mpnet-base-v2 (better quality)
- [ ] Make embedding provider configurable in settings.yaml
- [ ] Add --embedding-provider flag (openai|local)
- [ ] Benchmark performance vs. OpenAI embeddings
```

**Embedding Model Management** [Priority: MEDIUM]
```
- [ ] Auto-download models on first use
- [ ] Cache models in XDG_CACHE_HOME
- [ ] Support model updates
- [ ] Add model selection CLI
- [ ] Document hardware requirements (RAM, etc.)
```

### Week 14: Ollama Integration

**Local LLM Support** [Priority: HIGH]
```
- [ ] Create models/ollama_models.py
- [ ] Detect if Ollama is running (localhost:11434)
- [ ] Support ollama list of available models
- [ ] Implement generate_text() for Ollama
- [ ] Handle streaming responses
- [ ] Add --provider ollama flag
- [ ] Document Ollama setup
```

**Model Routing** [Priority: MEDIUM]
```
- [ ] Create model router logic
- [ ] Route simple queries to local models
- [ ] Route complex queries to cloud models
- [ ] Make routing strategy configurable
- [ ] Add cost tracking (tokens * price)
- [ ] Show cost estimates before expensive calls
```

### Week 15: Private Mode

**Privacy Features** [Priority: HIGH]
```
- [ ] Implement --private flag for CLI
- [ ] When private: use only local models
- [ ] When private: use only local embeddings
- [ ] No data sent to external APIs
- [ ] Add privacy indicators in UI (lock icon, etc.)
- [ ] Document privacy guarantees
```

**PII Detection** [Priority: MEDIUM]
```
- [ ] Integrate basic PII detection (regex for emails, SSNs)
- [ ] Warn user if PII detected in query
- [ ] Optionally anonymize PII before sending to API
- [ ] Add --anonymize flag
- [ ] Allow user to whitelist certain patterns
```

### Week 16: Optimization & Documentation

**Performance Optimization** [Priority: MEDIUM]
```
- [ ] Profile embedding generation speed
- [ ] Batch embeddings for efficiency
- [ ] Cache frequently accessed embeddings
- [ ] Optimize SQLite queries (indexes, etc.)
- [ ] Lazy-load large knowledge graphs
```

**Privacy Documentation** [Priority: HIGH]
```
- [ ] Write PRIVACY.md explaining data flows
- [ ] Document what data is stored where
- [ ] Explain local vs. cloud model trade-offs
- [ ] Hardware requirements for local operation
- [ ] Add privacy FAQ
```

**Phase 4 Success Criteria**:
- ✅ Can run fully offline with Ollama + local embeddings
- ✅ Privacy mode never leaks data to external APIs
- ✅ Local models are practical for daily use (good speed)
- ✅ Clear documentation of privacy guarantees

---

## Phase 5: Ecosystem (Weeks 17-24)

**Goal**: Enable community contributions and long-term sustainability

### Week 17-18: Plugin System

**Plugin Architecture** [Priority: HIGH]
```
- [ ] Design plugin API (load, init, hooks)
- [ ] Create repartee/plugins/ directory
- [ ] Implement plugin discovery (scan directory)
- [ ] Support for tool plugins (e.g., search, calculate)
- [ ] Support for memory plugins (e.g., custom sources)
- [ ] Support for provider plugins (e.g., custom LLMs)
- [ ] Add plugin configuration in settings.yaml
```

**Example Plugins** [Priority: MEDIUM]
```
- [ ] GitHub plugin (search repos, issues)
- [ ] Todoist plugin (task management)
- [ ] Web search plugin (DuckDuckGo)
- [ ] Calculator plugin (for math queries)
- [ ] Weather plugin (example of external API)
- [ ] Document each plugin as template
```

### Week 19-20: Tool Calling Framework

**Function Calling** [Priority: HIGH]
```
- [ ] Implement tool calling for supported models (OpenAI, Anthropic)
- [ ] Define tool schema format
- [ ] Register tools from plugins
- [ ] Execute tools and handle responses
- [ ] Add safety checks (confirm before actions)
- [ ] Support multi-step tool usage
```

**Built-in Tools** [Priority: MEDIUM]
```
- [ ] File operations (read, write, list)
- [ ] Shell command execution (with confirmation)
- [ ] Knowledge search (query personal notes)
- [ ] Web scraping (if plugin installed)
- [ ] Date/time utilities
```

### Week 21: MCP Server Polish

**MCP Protocol** [Priority: MEDIUM]
```
- [ ] Complete mcp/host.py and mcp/client.py
- [ ] Implement all MCP protocol methods
- [ ] Add MCP server mode (--mcp-server)
- [ ] Document MCP integration
- [ ] Test with other MCP clients
- [ ] Add authentication if needed
```

**MCP Features** [Priority: LOW]
```
- [ ] Expose memory as MCP resources
- [ ] Expose tools as MCP tools
- [ ] Support notifications for updates
- [ ] Implement progress reporting
- [ ] Add logging for MCP calls
```

### Week 22: Developer Documentation

**API Documentation** [Priority: HIGH]
```
- [ ] Write docs/API.md with all public APIs
- [ ] Document plugin development
- [ ] Add code examples for common tasks
- [ ] Generate Sphinx docs from docstrings
- [ ] Add architecture diagrams
- [ ] Document memory system internals
```

**Contributing Guide** [Priority: HIGH]
```
- [ ] Write CONTRIBUTING.md
- [ ] Set up issue templates
- [ ] Define code style guide
- [ ] Add PR checklist template
- [ ] Document release process
- [ ] Create developer setup guide
```

### Week 23: Plugin Marketplace

**Plugin Registry** [Priority: LOW]
```
- [ ] Create plugins.yaml registry
- [ ] Add --list-plugins command
- [ ] Add --install-plugin <name> command
- [ ] Add --update-plugins command
- [ ] Support versioning
- [ ] Add plugin search
```

**Quality Assurance** [Priority: MEDIUM]
```
- [ ] Plugin review process (if community grows)
- [ ] Automated security scanning
- [ ] Plugin compatibility checking
- [ ] User ratings/reviews (optional)
```

### Week 24: Release & Launch

**1.0 Release Preparation** [Priority: HIGH]
```
- [ ] Complete all Phase 1-4 items (if not done)
- [ ] Write CHANGELOG.md
- [ ] Bump version to 1.0.0
- [ ] Create release notes
- [ ] Package for PyPI (pip install repartee)
- [ ] Create release branch
- [ ] Tag v1.0.0 in git
```

**Launch Activities** [Priority: MEDIUM]
```
- [ ] Post to Hacker News (Show HN)
- [ ] Post to /r/commandline and /r/selfhosted
- [ ] Write blog post explaining vision
- [ ] Create demo video (asciinema recording)
- [ ] Announce on Twitter/Mastodon
- [ ] Add to awesome-cli lists
```

**Phase 5 Success Criteria**:
- ✅ 3+ community-contributed plugins
- ✅ Developer documentation is comprehensive
- ✅ 1.0 release is stable and feature-complete
- ✅ Project has 100+ GitHub stars (awareness metric)

---

## Ongoing Tasks (Throughout All Phases)

### Community Management
```
- [ ] Respond to GitHub issues within 48 hours
- [ ] Review and merge PRs within 1 week
- [ ] Maintain roadmap and update progress
- [ ] Write monthly progress updates
- [ ] Engage with users on forums/Discord
```

### Performance Monitoring
```
- [ ] Track query latency (p50, p95, p99)
- [ ] Monitor memory usage
- [ ] Track API costs (if telemetry enabled)
- [ ] Identify bottlenecks
- [ ] Optimize hot paths
```

### Security
```
- [ ] Regularly update dependencies (weekly scan)
- [ ] Respond to security advisories ASAP
- [ ] Run security scanners (Bandit, Safety)
- [ ] Audit API key handling
- [ ] Review plugin security model
```

---

## Success Metrics

**Phase 1** (Foundation):
- 80%+ test coverage
- 0 known crashes
- CI passing on all PRs

**Phase 2** (User Experience):
- 5+ daily active users
- 90% retention after first week
- <5 minutes to first successful interaction

**Phase 3** (Knowledge Integration):
- Can handle 1000+ note vault
- Search returns relevant results 90%+ of the time
- Users report AI "understands" their notes

**Phase 4** (Privacy & Local Models):
- Full offline operation possible
- Local models respond in <3 seconds for simple queries
- Privacy mode never leaks data

**Phase 5** (Ecosystem):
- 3+ community plugins
- 10+ contributors
- 100+ GitHub stars

**Overall (6 months)**:
- 50+ active users
- 500+ GitHub stars
- Featured in at least 1 tech blog/newsletter
- Self-sustaining community forming

---

## Resource Requirements

**Developer Time**:
- Phase 1: 40-50 hours
- Phase 2: 40-50 hours
- Phase 3: 50-60 hours
- Phase 4: 40-50 hours
- Phase 5: 50-60 hours
- **Total**: 220-270 hours (~6 months at 10 hours/week)

**Infrastructure**:
- GitHub repository (free)
- CI/CD (GitHub Actions - free tier sufficient)
- PyPI hosting (free)
- Optional: Discord server for community (~$0-10/month for bot)

**API Costs** (for development/testing):
- OpenAI embeddings: ~$5-10/month
- Anthropic/OpenAI API calls: ~$20-30/month
- **Total**: ~$25-40/month

---

## Risk Mitigation

**Risk**: Scope creep leading to never finishing
**Mitigation**: Strict adherence to phase goals, say "no" to features outside roadmap

**Risk**: Competing tool emerges and obsoletes Repartee
**Mitigation**: Focus on unique niche (PKM integration), build community early

**Risk**: Developer loses interest/time
**Mitigation**: Build in public, find co-maintainers by Phase 3

**Risk**: Local models too slow/poor quality
**Mitigation**: Always support cloud models as fallback, document trade-offs

**Risk**: Obsidian/knowledge format changes
**Mitigation**: Use standard Markdown as base, make parsers pluggable

---

## Conclusion

This roadmap provides a clear path from the current state (~1,800 lines, partial implementation) to a production-ready 1.0 release. The key is maintaining focus on the core value proposition: **AI that understands your personal knowledge**.

By following this plan, Repartee can establish itself as the leading CLI tool for knowledge workers who want context-aware AI assistance integrated with their personal note-taking systems.

**Next Step**: Complete Phase 1, Week 1 (logging + tests) to build confidence in the foundation.
