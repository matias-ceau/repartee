# Repartee

## Overview

Repartee is a versatile command-line interface (CLI) tool designed to interact with multiple AI APIs, including GPT-4o, Claude 3, Gemini 1.5, and Perplexity. It offers extensive features for managing and utilizing these AI resources effectively, providing users with the ability to send prompts, receive responses, track usage, and customize their experience.

### Definition of Repartee

*Repartee (noun)*: 
**A quick and witty reply or conversation characterized by such replies.**
Example: 
- "His repartee made everyone laugh during the meeting."
- "Her sharp repartee often left her colleagues speechless."

**In the context of this AI tool, "Repartee" symbolizes the swift and clever interactions users can have with various AI models, facilitating quick and intelligent exchanges to suit their needs.**

## Features

### Core Functionality
- Interface with multiple AI APIs (GPT-4o, Claude 3, Gemini 1.5, Perplexity).
- Send prompts to selected APIs and receive responses.
- Track spending by calculating the number of tokens used.

### Configuration
- Securely store API keys.
- Configure API parameters (e.g., max tokens, temperature).
- Support for a YAML configuration file.

### User Interface
- Command-line argument parsing for selecting APIs and prompts.
- Provide prompts via file or standard input.
- Output formatting options (standard output, file, REPL).
- Interactive REPL prompt.
- Input files and pipe shell commands support.
- Support for bash and xonsh commands.

### Advanced Features
- Support for multiple prompts in a single run.
- Chain responses from different APIs.
- Error handling and retry mechanisms for API calls.
- Logging for tracking API usage and responses.
- Display awesome prompts.
- Track spending and updated costs per token.

### Customization
- Easily add new APIs.
- Customizable prompt templates.
- Create custom 'roles' with custom initial prompts.

### Enhancements
- Caching responses to avoid redundant API calls.
- Rate limiting to handle API usage limits.
- Interactive mode for live prompt and response sessions.
- Easily searchable chats (using fzf or a Python library).
- Automatically save chats as markdown files.
- Internet search capability (for ChatGPT and Perplexity).

## Project Structure

```
repartee/
├── config/
│   ├── api_keys.yaml       # Store API keys securely
│   └── settings.yaml       # General settings and parameters
├── src/
│   ├── __init__.py
│   ├── main.py             # Entry point for the CLI tool
│   ├── api_caller.py       # Module for handling API calls
│   ├── config_manager.py   # Module for managing configuration
│   ├── prompt_handler.py   # Module for handling prompts and responses
│   ├── output_formatter.py # Module for formatting output
│   ├── repl.py             # Module for REPL interface
│   ├── spending_tracker.py # Module for tracking token spending
│   ├── search_util.py      # Module for searchable chats
│   ├── shell_commands.py   # Module for handling shell commands
│   ├── utils.py            # Utility functions
├── tests/
│   ├── test_api_caller.py
│   ├── test_config_manager.py
│   ├── test_prompt_handler.py
│   ├── test_output_formatter.py
│   ├── test_repl.py
│   ├── test_spending_tracker.py
│   ├── test_search_util.py
│   └── test_shell_commands.py
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

## To Do

- [ ] Plan the Configuration System
  - [ ] Define the structure of `api_keys.yaml` and `settings.yaml`.
  - [ ] Implement `config_manager.py` to handle reading/writing configurations using YAML.
- [ ] Develop Core Modules
  - [ ] Start with `api_caller.py` to handle API interactions.
  - [ ] Create `spending_tracker.py` to track token usage and calculate costs.
- [ ] Implement User Interface
  - [ ] Use `argparse` in `main.py` for command-line argument parsing.
  - [ ] Integrate `output_formatter.py` for formatting outputs.
  - [ ] Develop `repl.py` for an interactive REPL interface.
- [ ] Add Advanced Features
  - [ ] Implement `prompt_handler.py` to manage prompts, custom roles, and input from files or standard input.
  - [ ] Add support for chaining API responses and handling multiple prompts.
  - [ ] Include error handling, logging, and caching in `utils.py`.
- [ ] Enhancements
  - [ ] Develop `search_util.py` for searchable chats and saving chats as markdown files.
  - [ ] Implement `shell_commands.py` to handle bash and xonsh commands within the tool.
- [ ] Testing
  - [ ] Write unit tests for each module to ensure reliability and correctness.
  - [ ] Use the `tests/` directory for organizing test cases.
