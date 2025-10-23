# Repartee

> **AI that knows your knowledge**

Repartee is a command-line AI assistant that integrates with your personal knowledge base. Unlike generic AI tools that forget everything after each session, Repartee understands your notes, documents, and accumulated wisdom—making your knowledge available in every conversation.

**Status**: Active development | **Version**: 0.2.0 | **Target**: v1.0 in 6 months

---

## 🎯 Why Repartee?

**The Problem**: You have 100s or 1000s of notes in Obsidian, Logseq, or Markdown files. When you ask ChatGPT or Claude a question, they have no idea this knowledge exists.

**The Solution**: Repartee indexes your notes, builds a knowledge graph, and automatically retrieves relevant information during conversations. It's like having an AI assistant that's actually read all your notes.

**The Difference**: Most CLI agents are either coding-focused (Aider, Continue.dev) or generic chatbots (ChatGPT CLI). Repartee bridges the gap—it's an AI assistant designed for **knowledge workers** with extensive personal knowledge management systems.

## ✨ Key Features

### 🧠 Knowledge Integration (Unique to Repartee)
- **Obsidian Vault Integration**: Import and search your entire Obsidian vault
- **Knowledge Graph**: Automatically builds concept maps from your notes
- **Semantic Search**: Find relevant notes using AI embeddings
- **Auto-Context Injection**: Relevant notes are automatically added to conversations

### 💾 Advanced Memory Systems
- **Short-term Memory**: Current conversation context with intelligent truncation
- **Episodic Memory**: Searchable history of past conversations with vector embeddings
- **Semantic Memory**: Structured knowledge graph for concepts and relationships
- **Cross-Session Context**: AI remembers relevant information across conversations

### 🔌 Multi-Provider Support
- **OpenAI** (GPT-4, GPT-4o, etc.)
- **Anthropic** (Claude 3.7/3.5 Sonnet)
- **Google** (Gemini 1.5 Pro)
- **Perplexity** (Sonar models)
- **Local Models** (Ollama support - coming soon)

### 🖥️ Multiple Interfaces
- **CLI**: Quick one-shot queries and interactive conversations
- **REPL**: Persistent interactive shell (coming soon)
- **TUI**: Rich terminal UI with Textual (coming soon)

### 🔒 Privacy First
- **Local-First Design**: All data stored on your machine (XDG-compliant)
- **Local Model Support**: Run entirely offline with Ollama (coming soon)
- **Private Mode**: Zero data sent to external APIs (coming soon)
- **Transparent Storage**: Clear data handling, no hidden tracking

## 📦 Installation

**Requirements**: Python 3.13+ (we're working on supporting 3.11+)

```bash
# Clone the repository
git clone https://github.com/matias-ceau/repartee.git
cd repartee

# Install using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

**Note**: Repartee is in active development. Some features are still being implemented.

## Configuration

Repartee requires API keys to access language models. Set them as environment variables:

```bash
# OpenAI API key
export OPENAI_API_KEY=your-openai-key

# Anthropic API key
export ANTHROPIC_API_KEY=your-anthropic-key

# Google AI API key
export GOOGLE_API_KEY=your-google-key

# Perplexity API key
export PERPLEXITY_API_KEY=your-perplexity-key
```

Or create a `.env` file in your home directory with these keys.

## 🚀 Usage

### Basic Usage

```bash
# Interactive conversation with default model
repartee

# Use Anthropic's Claude
repartee -p anthropic

# Specify a particular model
repartee -p openai -m gpt-4-turbo

# One-shot query
repartee "What is the capital of France?"
```

### Working with Your Knowledge

```bash
# Import your Obsidian vault
repartee --import-obsidian ~/path/to/vault

# Search your notes
repartee "What did I write about machine learning?"

# Get answers using your knowledge
repartee "Summarize my notes on Python async programming"
```

### Advanced Features

```bash
# List recent conversations
repartee --list-conversations

# Continue a previous conversation
repartee --continue <conversation-id>

# Export conversation to Markdown
repartee --export <conversation-id>
```

## Memory Architecture

Repartee implements a multi-layered memory architecture:

1. **Short-Term Memory**: Manages the current conversation context with token-aware truncation to maintain context within model limits.

2. **Episodic Memory**: Stores conversation history with vector embeddings for semantic search, allowing the retrieval of relevant past exchanges.

3. **Semantic Memory**: Implements a knowledge graph that stores concepts and relationships, with special support for importing from Obsidian vaults.

## Project Structure

```
repartee/
├── config/             # Configuration files
├── src/
│   └── repartee/
│       ├── models/     # Model integrations (OpenAI, Anthropic, etc.)
│       ├── memory/     # Memory systems
│       │   ├── short_term_memory.py
│       │   ├── episodic_memory.py
│       │   └── semantic_memory.py
│       ├── ui/         # User interfaces
│       │   ├── cli.py
│       │   ├── repl.py
│       │   └── textual_tui.py
│       ├── tools/      # Tool integrations
│       └── main.py     # Entry point
└── pyproject.toml     # Package configuration
```

## Development

For development, uv makes dependency management simple:

```bash
# Add a new dependency
uv pip install some-package

# Update the lock file
uv pip freeze > requirements.txt
```

## 🗺️ Roadmap

Repartee is under active development. We've completed a comprehensive analysis of the project's viability and created a detailed roadmap. See [docs/ANALYSIS_RESULTS.md](docs/ANALYSIS_RESULTS.md) for the full analysis.

### Current Status (v0.2.0)
- [x] Core memory architecture (short-term, episodic, semantic)
- [x] Multi-provider model support (OpenAI, Anthropic, Google)
- [x] Basic CLI interface
- [x] Configuration system
- [x] Obsidian integration foundation

### In Progress (v0.3.0 - Foundation)
- [ ] Comprehensive logging system
- [ ] Test suite with 80%+ coverage
- [ ] CI/CD pipeline
- [ ] Error handling and retries
- [ ] Python 3.11+ support

### Coming Soon (v0.4.0 - User Experience)
- [ ] Complete REPL mode
- [ ] Rich TUI interface with Textual
- [ ] Conversation management (save, load, search)
- [ ] Export functionality

### Planned (v0.5.0+ - Knowledge & Privacy)
- [ ] Enhanced Obsidian integration
- [ ] Semantic search across knowledge base
- [ ] Local model support (Ollama)
- [ ] Local embeddings
- [ ] Privacy mode
- [ ] Plugin system

**Target**: v1.0 release in 6 months. See [docs/IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md) for the detailed plan.

## 🤝 Contributing

Repartee is actively seeking contributors! Whether you're interested in:
- Adding new model providers
- Improving the knowledge integration
- Building plugins
- Writing documentation
- Testing and reporting bugs

We'd love your help. See [docs/NEXT_ACTIONS.md](docs/NEXT_ACTIONS.md) for immediate opportunities.

### Development Setup

```bash
# Clone and install for development
git clone https://github.com/matias-ceau/repartee.git
cd repartee
uv pip install -e .

# Run tests (coming soon)
pytest

# Check code style
ruff check src/
```

## 📖 Documentation

- **[Analysis Results](docs/ANALYSIS_RESULTS.md)** - Project viability analysis and strategic positioning
- **[Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md)** - Detailed 6-month development plan
- **[Strategic Positioning](docs/STRATEGIC_POSITIONING.md)** - Market positioning and messaging
- **[Next Actions](docs/NEXT_ACTIONS.md)** - Immediate actionable next steps
- **[Reference Spec](docs/REFERENCE_SPEC.md)** - Technical architecture documentation

## 🙏 Acknowledgments

Repartee is built on the shoulders of giants:
- [Obsidian](https://obsidian.md) - Inspiration for knowledge management
- [OpenAI](https://openai.com), [Anthropic](https://anthropic.com), [Google](https://ai.google.dev) - AI model providers
- [Textual](https://textual.textualize.io) - Modern TUI framework
- [Rich](https://rich.readthedocs.io) - Beautiful terminal output

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Status**: Active development (v0.2.0)  
**Goal**: v1.0 in 6 months with full knowledge integration  
**Community**: Building - contributors welcome!  

⭐ Star this repo if you're interested in AI that understands your personal knowledge!
