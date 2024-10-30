# Repartee

Repartee is a command-line interface (CLI) tool designed to interface with multiple AI APIs to handle various functionalities, including sending prompts and receiving responses from AI models. It tracks spending based on token usage and offers a flexible configuration system.

## Core Functionality
- Interface with multiple AI APIs (GPT-4o, GPT-4-turbo, GPT-3.5-turbo from OpenAI, Claude 3, Gemini 1.5, Perplexity).
- Retrieve API keys from environment variables.
- Configure API parameters such as max tokens and temperature.
- Support YAML configuration files for settings.
- Command-line argument parsing for API selection and prompt input.
- Output formatting options (standard output, file, REPL).
- Interactive REPL prompt.
- Input from files and shell command piping.
- Advanced features like error handling, retry mechanisms, and spending tracking.

## Project Structure
```
repartee
├── LICENSE
├── README.md
├── config
│   └── settings.yaml
├── dev
│   ├── api.py
│   ├── notes
│   │   ├── Project canvas.canvas
│   │   ├── embedding.md
│   │   └── model names and prices.md
│   ├── prompts
│   │   ├── pp_2024-05-27.md
│   │   ├── pp_2024-06-05.md
│   │   └── project_file_o1.md
│   ├── terminal-emulator-in-textual.py
│   ├── terminal_gui.py
│   ├── tui.py
│   ├── tui2.py
│   └── utils
│       └── scrape.py
├── src
│   └── repartee
│       ├── __init__.py
│       ├── config.py
│       ├── main.py
│       ├── memory
│       │   ├── __init__.py
│       │   └── rag.py
│       ├── models
│       │   ├── __init__.py
│       │   ├── anthropic_models.py
│       │   ├── google_models.py
│       │   ├── openai_models.py
│       │   └── perplexity_models.py
│       ├── prompts
│       │   └── __init__.py
│       ├── rag.py
│       ├── tools
│       │   └── __init__.py
│       └── ui
│           ├── __init__.py
│           ├── cli.py
│           ├── fzf_tui.py
│           ├── repl.py
│           └── textual_tui.py
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## Key Components

1. **cli.py**: Handles command-line argument parsing and routes the inputs to the appropriate API client.
2. **config.py**: Manages loading and accessing configuration settings from YAML files.
3. **api_clients/openai_api.py**: Contains functions to interact with OpenAI's GPT models, including checking API key availability and making API requests.
4. **output_formatter.py**: Formats the AI responses into markdown format suitable for Obsidian, with specific callout blocks for system and user messages.
5. **spending_tracker.py**: Tracks token usage and calculates spending based on the model's token cost.
6. **repl.py**: Provides an interactive Read-Eval-Print Loop (REPL) for live prompt-response sessions.

## Roadmap

- [x] Set up the initial project structure
- [x] Implement CLI argument parsing
- [x] Implement configuration management
- [x] Implement OpenAI API client
- [ ] Implement Claude 3 API client
- [ ] Implement Gemini 1.5 API client
- [ ] Implement Perplexity API client
- [x] Implement basic output formatting
- [ ] Implement spending tracker
- [ ] Implement REPL interface
- [ ] Write tests for CLI
- [ ] Write tests for configuration management
- [ ] Write tests for API clients
- [ ] Write tests for prompt handler
- [ ] Write tests for output formatter
- [ ] Write tests for REPL
- [ ] Write tests for spending tracker
 
