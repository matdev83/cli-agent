# MVP Task Breakdown: Roadmap for CLI Agent Enhancement

This document outlines the remaining and new tasks required to enhance the MVP of the CLI-based software development agent. It incorporates findings from the initial code review and aims to achieve closer feature parity with the reference Cline VSCode extension, while adhering to TDD, SOLID, KISS, and DRY principles.

## I. Core Agent Refactoring & Design Improvements

These tasks focus on improving the agent's architecture, maintainability, and robustness based on the initial code review.

    *   **Description:** Replace the `if/elif/else` structure in `DeveloperAgent._run_tool` with a more extensible mechanism like dictionary-based dispatch or the Strategy pattern.
    *   **Reasoning:** Addresses code review point AC1; improves adherence to Open/Closed Principle (SOLID).
    *   **Status:** Done

    *   **Description:** Define a `Tool` ABC or Protocol that all tools must implement. This interface should define methods like `execute(params: Dict) -> str` and properties for `name`, `description`, and `parameters_schema`.
    *   **Reasoning:** Addresses AC2; improves SOLID (Liskov Substitution, Interface Segregation), makes tool addition cleaner, and facilitates dynamic prompt generation.
    *   **Status:** Done

    *   **Description:** Replace the current string-searching logic in `src/assistant_message.py` with a robust XML parser (e.g., Python's built-in `xml.etree.ElementTree`).
    *   **Reasoning:** Addresses SF6; significantly improves reliability of parsing LLM responses, handling XML entities and format variations.
    *   **Status:** Done

    *   **Description:** Use a templating engine (e.g., Jinja2) for constructing the system prompt in `src/prompts/system.py`.
    *   **Reasoning:** Addresses AC3, SF14; makes managing complex prompt logic (conditionals, variable substitution) cleaner and less error-prone.
    *   **Status:** Done

    *   **Description:** Define an `LLMWrapper` ABC or Protocol with methods like `send_message` to ensure a consistent contract for different LLM backends.
    *   **Reasoning:** Addresses AC4; improves extensibility for adding new LLM providers.
    *   **Status:** Done

*   **1.6. Clarify and Implement "Command Prompts" Workflow:**
    *   **Description:** Define and implement a clear workflow for how "command prompts" (from `src/prompts/commands.py`) and associated meta-tools (e.g., `new_task`, `condense`, `report_bug`, `new_rule` stubs in `meta_tools.py`) are invoked and processed by the agent.
    *   **Reasoning:** Addresses SF15; ensures these Cline-inspired interaction patterns are effectively integrated.
    *   **Status:** To Do

    *   **Description:** Implement a consistent mechanism for tools to report execution failures. The agent should catch these errors and relay them to the LLM in a structured format.
    *   **Reasoning:** Improves agent robustness and allows LLM to react to tool issues.
    *   **Status:** Done

    *   **Description:** Make the `_to_bool` utility (or equivalent logic for parsing boolean tool parameters from LLM) stricter (e.g., only 'true'/'false') or more robust with clear documentation for the LLM.
    *   **Reasoning:** Addresses SF1; reduces ambiguity in LLM communication.
    *   **Status:** Done

## II. Tool Implementation & Enhancement

These tasks focus on implementing missing tools from Cline, completing stubbed tools, and enhancing existing ones.

    *   **Description:** Complete the implementation of the `browser_action` tool, integrating a browser control library (e.g., Playwright or Selenium) to provide capabilities like launching, clicking, typing, scrolling, and closing the browser, as specified in Cline's system prompt.
    *   **Reasoning:** Currently stubbed; essential for web interaction tasks.
    *   **Status:** Done

*   **2.2. Fully Implement `use_mcp_tool`:**
    *   **Description:** Complete the implementation of the `use_mcp_tool` for interacting with external Model Context Protocol (MCP) servers.
    *   **Reasoning:** Currently stubbed; key for extending agent capabilities via MCP.
    *   **Status:** To Do

*   **2.3. Fully Implement `access_mcp_resource`:**
    *   **Description:** Complete the implementation of the `access_mcp_resource` tool for accessing data resources exposed by MCP servers.
    *   **Reasoning:** Currently stubbed; complements `use_mcp_tool`.
    *   **Status:** To Do

    *   **Description:** Add and implement the `load_mcp_documentation` tool as described in Cline's system prompt, allowing the agent to fetch documentation for creating MCP servers.
    *   **Reasoning:** Missing from current Python agent; supports MCP development.
    *   **Status:** Done

    *   **Description:** Extend the `list_code_definition_names` tool to support more programming languages beyond Python, potentially using generic parsing libraries (e.g., tree-sitter).
    *   **Reasoning:** Addresses SF16; increases tool utility across different project types.
    *   **Status:** Done

*   **2.6. Verify and Align All Tools with Cline Specifications:**
    *   **Description:** Review all existing Python tools (`execute_command`, `read_file`, `write_to_file`, `replace_in_file`, `search_files`, `list_files`, `ask_followup_question`, `attempt_completion`) against their detailed descriptions and parameters in `vendor/cline/src/core/prompts/system.ts`. Ensure full compliance.
    *   **Reasoning:** Ensures faithful porting of Cline's interaction model.
    *   **Status:** To Do

## III. Advanced Agent Features

Implementing more sophisticated agent behaviors based on Cline's design.

*   **3.1. Implement ACT MODE vs. PLAN MODE Logic:**
    *   **Description:** Introduce distinct "ACT MODE" and "PLAN MODE" operational modes for the agent. In PLAN MODE, the agent should use the `plan_mode_respond` tool for interactions. This includes managing state and tool availability based on the current mode.
    *   **Reasoning:** Core feature of Cline's interaction model for planning and execution.
    *   **Status:** To Do

*   **3.2. Implement `plan_mode_respond` Tool:**
    *   **Description:** Add and integrate the `plan_mode_respond` tool, enabling the agent to communicate and present plans during PLAN MODE.
    *   **Reasoning:** Essential for the PLAN MODE functionality.
    *   **Status:** To Do

*   **3.3. Develop Sophisticated Memory Management:**
    *   **Description:** Design and implement improved memory management, such as context window summarization for long conversations and potentially integrating vector stores for semantic search of history or external documents.
    *   **Reasoning:** Enhances agent's ability to handle complex, multi-turn tasks and large contexts.
    *   **Status:** To Do

## IV. Testing & Quality Assurance (TDD)

Ensuring the agent is reliable and robust through comprehensive testing.

*   **4.1. Develop Comprehensive Unit Test Suite:**
    *   **Description:** Create unit tests for all modules and critical functions, particularly:
        *   `src/assistant_message.py` (parser logic with various inputs).
        *   `src/agent.py` (`DeveloperAgent.run_task` using `MockLLM` and mocked tools).
        *   Each individual tool in `src/tools/` (mocking external interactions).
        *   Prompt generation logic in `src/prompts/`.
    *   **Reasoning:** Core TDD practice; ensures component reliability (addresses Section 5 of code review).
    *   **Status:** To Do

*   **4.2. Implement Integration Tests for Agent Loop:**
    *   **Description:** Develop integration tests that verify the end-to-end agent loop, using `MockLLM` to simulate interactions and tool execution sequences.
    *   **Reasoning:** Validates the orchestration of different components.
    *   **Status:** To Do

## V. Documentation

Maintaining clear and up-to-date documentation.

*   **5.1. Update `AGENTS.md`:**
    *   **Description:** As features from sections I-III are implemented, update `AGENTS.md` to accurately reflect the agent's capabilities, toolset, operational modes, and memory management.
    *   **Reasoning:** Keeps design and capability documentation current.
    *   **Status:** Ongoing (as features are completed)

*   **5.2. Ensure README Accuracy:**
    *   **Description:** Regularly review and update `README.md` to provide correct setup, usage instructions, and command-line arguments, especially as new tools or models are supported.
    *   **Reasoning:** Essential for user understanding and onboarding.
    *   **Status:** Ongoing
