# Repartee

Repartee is a powerful command-line interface (CLI) for interacting with AI models, featuring sophisticated memory systems that enable context-aware conversations. It's designed to provide a seamless experience across different AI providers while maintaining context and user preferences.

## Features

- **Multiple Model Support**: Connect to models from OpenAI, Anthropic, Google, and Perplexity
- **Advanced Memory Systems**:
  - Short-term memory for current conversation context
  - Episodic memory with vector embeddings for semantic search of past conversations
  - Semantic memory with knowledge graph for conceptual knowledge storage
- **Obsidian Integration**: Import knowledge graphs from Obsidian vaults to leverage existing knowledge
- **Multiple Interfaces**: Command-line, TUI, and REPL interfaces
- **Flexible Configuration**: Easy API key management and customizable settings

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/repartee.git
cd repartee

# Create a virtual environment using uv
uv venv .

# Install the package (uv automatically detects the pyproject.toml)
uv pip install -e .
```

When working in the project directory, uv automatically detects and uses the local virtual environment without explicit activation.

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

## Usage

### Basic Usage

```bash
# Interactive conversation with OpenAI's default model
repartee

# Interactive conversation with Anthropic's Claude
repartee -p anthropic

# Specify a particular model
repartee -p openai -m gpt-4-turbo

# One-shot query
repartee "What is the capital of France?"
```

### Advanced Features

```bash
# Import knowledge from an Obsidian vault
repartee --import-obsidian ~/path/to/obsidian/vault

# List recent conversations
repartee --list-conversations
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

## Roadmap

- [x] Implement advanced memory architecture
- [x] Build OpenAI model integration
- [x] Build Anthropic model integration
- [x] Implement CLI interface
- [x] Add Obsidian vault integration
- [ ] Make embedding module configurable (default to OpenAI v3 small)
- [ ] Set LLM data location configurable (default to $XDG_DATA_HOME/.repartee)
- [ ] Support metadata in YAML frontmatter for knowledge markdown files
- [ ] Implement knowledge graph visualization with vis.js or similar tools
- [ ] Add JSON-LD/RDF support for enhanced knowledge representation
- [ ] Implement tool calling framework
- [ ] Add TUI interfaces
- [ ] Implement advanced conversation retrieval
- [ ] Add multi-agent architecture
- [ ] Implement collaborative agent workflows

## Contributing

Contributions are welcome! Feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
