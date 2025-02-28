# Repartee Project Reference

## Build Commands
- Install dependencies: `uv pip install -e .`
- Update lock file: `uv pip freeze > requirements.txt`
- Run application: `repartee`
- Run with specific model: `repartee -p anthropic` or `repartee -p openai -m gpt-4-turbo`

## Project Structure
- CLI assistant with advanced memory architecture (short-term, episodic, semantic)
- Multiple model integrations (OpenAI, Anthropic, Google, Perplexity)
- Multi-interface support (CLI, TUI, REPL)
- Obsidian knowledge graph integration

## Code Style & Architecture
- Python 3.12+
- Dependencies managed with uv
- Object-oriented for clarity but minimalistic (avoid unnecessary abstraction)
- Follows PEP8 conventions
- Open to integration with higher-level libraries (langchain, llamaindex, etc.) when appropriate
- Willing to integrate other languages (bash, cython, rust) for performance or convenience
- Memory architecture is layered (short-term, episodic, semantic)

## Configuration Approach
- API keys stored in environment variables only (never in config files)
- Model preferences, paths, non-confidential settings in YAML config
- Follow XDG specification for file locations:
  - Config in XDG_CONFIG_HOME
  - Data (SQLite DBs, conversation logs) in XDG_DATA_HOME
  - Cache in XDG_CACHE_HOME
  - No sensitive data in config files that will be in Git repo

## Model Preferences
- Default one-shot queries: Claude 3.7 (formerly Claude 3.5)
- Default REPL: GPT-4o
- Specialized tasks: o3-mini-high, Perplexity
- Future: Local models (ollama, transformers) as router models

## Memory System Design
- Session-based fresh memory by default
- Compressed summary of relevant information
- On-demand retrieval of semantic/episodic memories as context
- Low threshold for memory retrieval based on contextual cues
- Avoid large system prompts at initialization
- Dynamic context loading based on conversation analysis
- Support for temporal queries and project-specific context retrieval

## Future Integration
- Testing framework TBD
- Potential integrations with ChromaDB
- Multi-agent architecture
- Local model routing

## Key Files
- `src/repartee/main.py`: Entry point
- `src/repartee/config.py`: Configuration management
- `src/repartee/memory/`: Memory systems implementation
- `src/repartee/models/`: Model provider integrations
- `src/repartee/ui/`: User interfaces