# MVP Task Breakdown

This document enumerates the concrete tasks required to deliver a minimal but functional CLI-based software development agent. Tasks are ordered sequentially so that foundations are built before layering extra features.

## 1. Project initialization
- **1.1 Inspect vendor code** – review `vendor/cline` for prompts and tool semantics. Optionally consult `vendor/crewAI` for inspiration, but the MVP will not depend on it.
- **1.2 Set up Python package** – create a `src` package structure and `pyproject.toml` (or `setup.py`). Include basic lint and test configuration (e.g. `ruff`, `pytest`).

## 2. Port Cline prompts
- **2.1 Extract system prompt** – translate `vendor/cline/src/core/prompts/system.ts` into a Python multiline string (maintain tool descriptions verbatim). Store this in `src/prompts/system.py`.
- **2.2 Extract command prompts** – convert prompts like `new_task` and `condense` from `commands.ts` to Python templates.
- **2.3 Provide helper functions** – write small utilities that deliver the appropriate prompt text given runtime options (working directory, browser support, etc.).

## 3. Assistant message parsing
- **3.1 Port parser logic** – implement a Python version of Cline’s `parseAssistantMessageV2`. It should detect `<tool>` XML blocks in the LLM response and return a structured representation (`tool_name`, parameters, and free text segments).
- **3.2 Unit tests** – create tests ensuring the parser correctly interprets typical responses and edge cases.

## 4. Core tool implementations
Implement Python tool classes matching Cline's specification. Each tool exposes a `run()` method.

- **4.1 File tools** – `read_file`, `write_to_file`, `replace_in_file`, `list_files`, and `search_files`.
- **4.2 Command execution** – `execute_command` with a flag requiring user approval for dangerous commands. Use Python’s `subprocess` module and handle timeouts or errors gracefully.
- **4.3 Code navigation** – `list_code_definition_names` or similar features to help the agent explore the project. Stub advanced tools (`browser_action`, `use_mcp_tool`, `access_mcp_resource`) but leave hooks for future work.
- **4.4 Tests for each tool** – verify that file modifications and command execution behave as expected.

## 5. Agent and task loop
- **5.1 Configure the agent** – build a `DeveloperAgent` class that loads the system prompt, registers the core tools, and defines how messages are sent to the LLM.
- **5.2 Conversation loop** – replicate Cline’s loop by repeatedly sending user input + tool results to the LLM, parsing its reply, executing the indicated tool, and appending the outcome to the conversation history until `attempt_completion` is triggered or a limit is hit.
- **5.3 Memory** – maintain short term conversation history and optional file context using our own data structures.

## 6. CLI entry point
- **6.1 Create `src/cli.py`** – handle command-line arguments (task description, model name, auto-approve flag, etc.).
- **6.2 Run the agent** – boot the `DeveloperAgent` with the given options and invoke the conversation loop. Print tool outputs in a user-friendly manner.
- **6.3 Error handling** – catch exceptions, offer to continue or abort on failure, and log actions for later review.

## 7. Documentation and examples
- **7.1 Update README** – provide instructions on installation, running the CLI and contributing.
- **7.2 Example workflows** – add a few step-by-step examples demonstrating typical usage (e.g. create a new file, run tests, fix a lint error).

## 8. Further enhancements (post-MVP)
These tasks are optional for the initial deliverable but should be tracked for future work.

- Integrate `browser_action`, `use_mcp_tool` and `access_mcp_resource` once a backend is available.
- Consider support for multiple agents collaborating in the future.
- Add advanced summarization and context condensation using prompts from Cline.

