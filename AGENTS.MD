# Agent Capabilities and Internals

This document provides more detailed information about the `DeveloperAgent`, its interaction model, and advanced usage.

## Agent Overview

The `DeveloperAgent` is the core component responsible for understanding user tasks, interacting with Large Language Models (LLMs), and utilizing a suite of tools to perform actions on behalf of the user. It aims to replicate and extend the capabilities found in IDE-based coding assistants but within a command-line interface.

## Agent Loop and User Interaction

The agent operates in a loop, typically initiated by a user's task request:

1.  **Task Input**: The user provides an initial task description.
2.  **At-Mention Processing**:
    *   Before sending the task to the LLM, the user's input string is scanned for any file path mentions (e.g., `@src/main.py`, `@./docs/requirements.txt`).
    *   For each unique and valid file path mentioned:
        *   The agent attempts to read the content of the file from the workspace.
        *   Successfully read file contents are then formatted (e.g., using XML-like tags like `<file_content path="REL_PATH">...</file_content>`) and prepended to the original user task.
    *   This mechanism allows users to easily inject relevant file context directly into the prompt for the LLM.
    *   A file caching system is used to keep track of available files in the project directory. This cache supports tab-based autocompletion for file paths when typing an at-mention (e.g., `@src/cli<TAB>`). The cache can be updated using the `/refresh` command.
3.  **LLM Interaction**: The (potentially modified) user task, along with conversation history and a system prompt, is sent to the configured LLM.
4.  **Response Parsing**: The LLM's response is parsed. This involves:
    *   Extracting plain text portions of the response.
    *   Identifying any tool usage requests (e.g., to read a file, execute a command). These are typically encoded in an XML-like format by the LLM based on the system prompt's instructions.
5.  **Tool Execution**:
    *   If a tool use is identified, the agent validates the tool name and its parameters.
    *   Permissions are checked: some tools or specific parameters (like dangerous commands) may require user confirmation via a (y/n) prompt unless an auto-approval flag is set (e.g., `--auto-approve` or more granular flags like `--allow-execute-all-commands`).
    *   If approved, the tool is executed. The result of the tool execution (e.g., file content, command output, error message) is then typically fed back into the LLM in the next turn as user input to inform subsequent steps.
6.  **Continuation or Completion**:
    *   If the LLM used a tool, the agent usually loops back to step 3, sending the tool's result to the LLM for further processing or to generate the next action.
    *   If the LLM responds with plain text and indicates the task is complete (e.g., by using a special `attempt_completion` tool or by its textual response), the agent concludes the current task and presents the final result to the user.
    *   The loop also terminates if a maximum number of steps is reached.

## Key Features and Components

*   **Memory**: Maintains conversation history to provide context to the LLM.
*   **Tools**: A collection of functions the agent can use (see `README.md` for a list).
*   **System Prompt**: A detailed initial prompt that guides the LLM on its role, available tools, expected response format, and constraints.
*   **Confirmation Mechanism**: Ensures user control over potentially sensitive actions.
*   **Error Handling**: The agent includes logic to handle malformed LLM responses or tool execution errors, providing feedback to the LLM to encourage self-correction. This includes an "admonishment" message if the LLM makes several consecutive errors.
*   **File Mention Context Injection**: As described above, automatically includes content of mentioned files in prompts.
*   **File Path Autocompletion**: Aids users in specifying correct file paths for at-mentions.

## Slash Commands

The CLI provides several slash commands for controlling agent behavior and accessing features interactively:

*   **/model `<model_name>`**:
    *   **Description**: Changes the active Large Language Model.
    *   **Usage**: `/model anthropic/claude-3-opus` or `/model mock`

*   **/set-timeout `<seconds>`**:
    *   **Description**: Adjusts the timeout for LLM API calls (in seconds).
    *   **Usage**: `/set-timeout 180.5`

*   **/plan**:
    *   **Description**: Switches the agent to "PLAN MODE", where it focuses on generating a plan to accomplish a task rather than immediate execution.
    *   **Usage**: `/plan`

*   **/act**:
    *   **Description**: Switches the agent to "ACT MODE" (default), where it focuses on direct execution of steps to accomplish a task.
    *   **Usage**: `/act`

*   **/refresh**:
    *   **Description**: Re-scans the project directory and updates the internal cache of file paths used for at-mention autocompletion. Useful if new files are added or removed during an agent session.
    *   **Usage**: `/refresh`

*   **/undo**:
    *   **Description**: Reverts file changes based on auto-commits made by the agent during the current session. If no commit ID is provided, it reverts the last auto-commit. If a commit ID (or its prefix) is provided, it attempts to revert to the state *before* that specific commit.
    *   **Usage**: `/undo` or `/undo <commit_id>`

*   **/undo-all**:
    *   **Description**: Reverts all auto-committed changes made by the agent during the current CLI session back to the state of the repository when the session started.
    *   **Usage**: `/undo-all`

*   **/help**:
    *   **Description**: Displays detailed help information for all available slash commands.
    *   **Usage**: `/help`

*   **/stop**:
    *   **Description**: Signals the currently active agent task to attempt a graceful stop.
    *   **Usage**: `/stop`

*   **/exit**:
    *   **Description**: Exits the CLI application.
    *   **Usage**: `/exit`

## Future Development

Potential areas for future development include:
*   More sophisticated planning capabilities (PLAN MODE).
*   Enhanced toolset, including more complex code manipulation and browser interaction.
*   Deeper integration with version control systems beyond basic auto-commits.
*   Improved error handling and self-correction by the LLM.
*   Support for multi-capability plugins (MCPs).
