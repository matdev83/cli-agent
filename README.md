# CLI Software Development Agent

## Brief Description

An MVP (Minimum Viable Product) of a command-line software development agent designed to assist with various coding and development tasks.

## Purpose

This project aims to re-implement core features of the Cline VSCode extension as a standalone Command Line Interface (CLI) tool. It focuses on reusing Cline's successful agent loop, prompting strategies, and tool handling methodologies to create a versatile and extensible software development assistant that operates outside of an IDE.

## Core Features

*   **LLM Interaction**: Communicates with Large Language Models to understand tasks and generate solutions.
    *   Supports a `MockLLM` for deterministic testing using pre-defined responses.
    *   Integrates with `OpenRouterLLM` to connect to various live language models.
*   **Response Parsing**: Intelligently parses LLM responses to identify and extract tool usage requests (formatted as XML-like blocks).
*   **Tool Execution**: A suite of tools to interact with the development environment:
    *   **File Operations**: `read_file`, `write_to_file`, `replace_in_file` (using specific SEARCH/REPLACE blocks), `list_files`, `search_files` (regex search in files).
    *   **Command Execution**: `execute_command` to run shell commands, with an optional approval mechanism.
    *   **Code Analysis**: `list_code_definition_names` to extract top-level definitions from source code (currently Python-focused).
*   **System Prompt**: Utilizes a detailed system prompt inspired by Cline, guiding the LLM's behavior and tool usage.
*   **Stubbed Advanced Tools**: Includes stubs for future implementation of `browser_action`, `use_mcp_tool`, and `access_mcp_resource`.

## Setup Instructions

1.  **Python Version**: Python 3.8+ is recommended.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Assuming a `requirements.txt` would be standard. If not present, necessary packages like `openai` would need to be installed manually).*
3.  **Set API Key**: For using live models via OpenRouter, set the `OPENROUTER_API_KEY` environment variable:
    ```bash
    export OPENROUTER_API_KEY="your_api_key_here"
    ```

## How to Run the Agent

The agent is run via the main CLI script.

**Basic Usage:**

```bash
python -m src.cli "Your task description here"
```

**Key Arguments:**

*   `task`: (Required) A string describing the task for the agent to perform.
*   `--model <model_name>`: Specify the LLM to use.
    *   `mock`: Uses the `MockLLM`. Requires `--responses-file`.
    *   An OpenRouter model name (e.g., `anthropic/claude-3-opus`): Uses `OpenRouterLLM`. Requires `OPENROUTER_API_KEY`.
    *   Default: `mock`.
*   `--responses-file <path>`: Path to a JSON file containing a list of mock responses for the `MockLLM`. Required if `model` is `mock`.
*   `--auto-approve`: If set, automatically approves commands that would normally require user confirmation (e.g., `execute_command` with `requires_approval=true`).
*   `--cwd <path>`: Sets the current working directory for the agent. Defaults to the directory where the script is run.
*   `--llm-timeout <seconds>`: Sets the timeout in seconds for LLM API calls. Defaults to `120.0`. This applies to calls made via `OpenRouterLLM`.

**Example with Mock LLM:**

```bash
python -m src.cli "Create a Python file that prints hello world" --model mock --responses-file path/to/mock_responses.json
```

**Example with OpenRouter LLM:**

```bash
python -m src.cli "Refactor the main function in utils.py to be more efficient" --model anthropic/claude-3-opus --auto-approve
```

## How to Run Tests

The project uses `pytest` for running tests. Tests are located in the `tests/` directory.

1.  **Install development dependencies** (including pytest and other testing tools):
    ```bash
    poetry install -E dev
    ```
    (If you haven't installed Poetry yet, please follow standard Poetry installation instructions.)

2.  **Run all tests:**
    ```bash
    poetry run pytest
    ```

3.  **Run specific test files:**
    ```bash
    poetry run pytest tests/test_agent.py
    poetry run pytest tests/test_llm.py
    ```

The `MockLLM` is used extensively in tests to provide deterministic LLM responses, facilitating testing of agent logic and tool interactions.

## Project Structure Overview

```
├── docs/
│   ├── CODEX_PLAN.md       # Original project plan and architecture
│   ├── MVP-TASKS.md        # MVP task breakdown and status
│   └── CODE_REVIEW.md      # Detailed code review and recommendations
├── src/
│   ├── __init__.py
│   ├── agent.py            # DeveloperAgent: core agent logic, tool dispatch
│   ├── assistant_message.py # Parses LLM responses for tool calls
│   ├── cli.py              # CLI entry point, argument parsing
│   ├── llm.py              # LLM interaction (MockLLM, OpenRouterLLM)
│   ├── memory.py           # Conversation history management
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── commands.py     # Prompts for special commands (e.g., new_task)
│   │   └── system.py       # Main system prompt definition
│   └── tools/
│       ├── __init__.py
│       ├── code.py         # Tools for code analysis (list_code_definitions, stubs for browser/MCP)
│       ├── command.py      # execute_command tool
│       └── file.py         # File operation tools
└── vendor/
    ├── cline/              # Reference Cline VSCode extension code
    └── crewAI/             # Reference CrewAI framework code (evaluated but not used)
```

## Further Information

*   **`docs/CODEX_PLAN.md`**: For insights into the original design philosophy, architecture, and planned features.
*   **`docs/MVP-TASKS.md`**: To understand the scope and checklist for the initial Minimum Viable Product.
*   **`docs/CODE_REVIEW.md`**: For a detailed analysis of the current codebase, identified issues, and recommendations for improvement.
*   **`AGENTS.md`**: (To be created) This document will provide more details on agent capabilities, advanced usage, and future development plans.

---

This README provides a basic overview. For more in-depth information, please consult the documents in the `docs/` directory.
