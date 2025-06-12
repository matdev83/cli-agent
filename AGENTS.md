# Agent Capabilities and Design

This document outlines the capabilities, design, and future development aspects of the CLI Software Development Agent.

## 1. The DeveloperAgent

*   **Role**: The `DeveloperAgent` is the core component that orchestrates interactions between the Large Language Model (LLM) and the suite of available tools to accomplish user-defined software development tasks.
*   **Core Logic**: It manages the conversation history with the LLM, invokes the LLM with the current context, parses the LLM's responses to identify requests for tool usage, and executes those tools, feeding the results back into the conversation.

## 2. Agent Loop

The agent operates on a cyclical process:

1.  **Initialization**: The agent is initialized with a system prompt (see `src/prompts/system.py`) that defines its persona, available tools, and operational guidelines. The current working directory and other environment details are also established.
2.  **User Task**: The loop begins with an initial task provided by the user via the CLI.
3.  **LLM Invocation**: The agent sends the entire conversation history (which includes the system prompt, user task, previous tool calls, and their results) to the configured LLM.
4.  **Response Parsing**: The LLM's response is parsed by `assistant_message.parse_assistant_message`:
    *   It distinguishes between plain text content and tool usage blocks (formatted in an XML-like structure).
    *   If the `attempt_completion` tool is identified, the agent considers the task complete and presents the final result provided within the tool's parameters.
    *   If other tool usage blocks are found, the agent prepares to execute the specified tool.
    *   If no tool blocks are found, the LLM's textual response is typically presented to the user (though the current loop primarily focuses on tool execution or completion).
5.  **Tool Execution**:
    *   The agent maps the tool name extracted from the LLM's response to the corresponding implemented tool function.
    *   Parameters for the tool, also extracted from the response, are passed to the tool function.
    *   The tool is executed. This might involve file system operations, running shell commands, or analyzing code.
    *   Certain tools, like `execute_command`, may require explicit user approval if their `requires_approval` parameter is set to `true` and the agent is not in `auto-approve` mode.
6.  **History Update**: The result of the tool execution (e.g., file content, command output, success/error message) is formatted as a user message and appended to the conversation history. This informs the LLM of the outcome of its requested action.
7.  **Iteration**: The loop repeats from step 3, sending the updated history to the LLM, until either the `attempt_completion` tool is called or a predefined maximum number of steps is reached (to prevent infinite loops).

## 3. Prompting Strategy

Effective communication with the LLM is key to the agent's performance:

*   **System Prompt**: A comprehensive system prompt, located in `src/prompts/system.py` and adapted from Cline's VSCode extension, is crucial. It meticulously details:
    *   The agent's persona (Cline, a skilled software engineer).
    *   The list of available tools.
    *   The precise XML-like format for requesting tool usage, including tool names and parameter names/tags.
    *   Guidelines for behavior, problem-solving, and interaction flow.
*   **Tool Communication**: The LLM requests the use of a tool by embedding an XML-like structure in its response. For example:
    ```xml
    <read_file>
    <path>src/main.py</path>
    </read_file>
    ```
*   **Tool Results**: After a tool is executed, its output (or any error message) is sent back to the LLM in the subsequent turn, framed as a user message. A typical format is:
    ```
    Result of read_file:
    # Content of src/main.py...
    ```
    This allows the LLM to process the outcome of its previous action and decide on the next step.

## 4. Current Toolset

The agent is equipped with the following tools, primarily defined in `src/tools/`:

*   `execute_command`: Executes a shell command. It can require user approval for commands deemed potentially risky if the `requires_approval` parameter is true.
*   `read_file`: Reads the entire content of a specified file and returns it as a string.
*   `write_to_file`: Writes provided content to a specified file. It will create the file if it doesn't exist or overwrite it if it does. It also creates necessary parent directories.
*   `replace_in_file`: Applies targeted modifications to an existing file using one or more `SEARCH/REPLACE` blocks defined in a diff-like format.
*   `search_files`: Performs a regular expression search across files within a specified directory, returning matching lines with their context.
*   `list_files`: Lists files and directories within a given path, with an option for recursive listing.
*   `list_code_definition_names`: Lists top-level code definitions (functions, classes, etc.) for supported source files (currently Python `.py` files only) in a directory.
*   `browser_action` (Stubbed): Intended for future implementation of browser interaction capabilities (e.g., navigation, scraping, form filling via Puppeteer).
*   `use_mcp_tool` (Stubbed): Designed for future integration with Model Context Protocol (MCP) servers, allowing the agent to use tools provided by these external servers.
*   `access_mcp_resource` (Stubbed): Planned for accessing data resources exposed by MCP servers.

The LLM can also invoke other "meta" tools defined in the system prompt, such as:
*   `attempt_completion`: Used by the LLM to signal that it believes the task is complete and to provide the final result.
*   `ask_followup_question`: Allows the LLM to ask clarifying questions to the user.
*   `new_task`, `condense`, `report_bug`, `new_rule`, `plan_mode_respond`, `load_mcp_documentation`: These are also defined in the system prompt and `assistant_message.py`'s `TOOL_USE_NAMES`, implying the LLM can request their use. However, their full integration into the agent's control flow (beyond generic tool dispatch) is a subject for further development.

## 5. Extending with New Tools

Adding new capabilities to the agent involves integrating new tools.

**Current Method:**

1.  **Define Tool Function**: Create a Python function for the new tool within a module in the `src/tools/` directory (e.g., `src/tools/my_new_tool.py`). This function should encapsulate the tool's logic.
2.  **Expose in `__init__`**: Add the tool function to the `__all__` list in `src/tools/__init__.py` to make it easily importable.
3.  **Register in Agent**: Add the tool function to the `self.tools` dictionary in `DeveloperAgent.__init__`, mapping the tool's string name (as used by the LLM) to the Python function.
    ```python
    self.tools: Dict[str, Callable[..., object]] = {
        # ... existing tools
        "my_new_tool_name": tools.my_new_tool_function,
    }
    ```
4.  **Update System Prompt**: Modify `src/prompts/system.py` to include the new tool. This involves:
    *   Adding a new section describing the tool: its purpose, parameters, and exact XML usage format.
    *   Ensuring the LLM knows when and how to use this new tool.
5.  **Update Parser Definitions**: Add the new tool's name to `TOOL_USE_NAMES` in `src/assistant_message.py`. Add any new parameter names to `TOOL_PARAM_NAMES`. This step is crucial for the parser to recognize and extract the tool call and its arguments.

**Recommended Future Approach (based on `docs/CODE_REVIEW.md`):**

To improve modularity, maintainability, and reduce boilerplate when adding tools:

1.  **Tool Base Class**: Define a common `Tool` Abstract Base Class (ABC) or Protocol. This class would require methods/properties like:
    *   `name() -> str`: The official name of the tool for LLM interaction.
    *   `description() -> str`: A brief description of what the tool does.
    *   `parameters_schema() -> Dict`: A schema defining the tool's parameters (e.g., using JSON Schema).
    *   `execute(params: Dict) -> str`: The method that performs the tool's action.
2.  **Implement Tool Class**: For each new tool, create a class that inherits from the `Tool` ABC and implements the required methods.
3.  **Agent Registration**: The `DeveloperAgent` would maintain a list or dictionary of these `Tool` objects. Registration could be manual or semi-automated (e.g., discovering tool classes in a specific module).
4.  **Dynamic Prompt Generation**: The agent could dynamically generate the tool documentation section of the system prompt based on the metadata (`name`, `description`, `parameters_schema`) provided by each registered tool. This ensures the prompt always reflects the actual available tools.
5.  **Simplified Dispatch**: The agent's `_run_tool` method would simply find the `Tool` object by name and call its `execute` method, passing the parsed parameters. Parameter validation could also be handled by the `Tool` object itself based on its schema.
6.  **Parser Update**: The parser might only need `TOOL_USE_NAMES` to identify tool blocks, or this could also be derived from registered tools. Parameter extraction would still occur, but validation would be delegated.

## 6. Future Enhancements

The current MVP provides a solid foundation. Potential areas for future development include:

*   **Full Tool Implementation**: Complete the implementation of currently stubbed tools: `browser_action` (requiring a browser control library like Playwright or Selenium) and MCP tools (`use_mcp_tool`, `access_mcp_resource`, requiring an MCP client/server setup).
*   **Expanded Language Support**: Enhance `list_code_definition_names` to support more programming languages beyond Python, potentially using universal code parsing libraries (e.g., tree-sitter).
*   **Sophisticated Memory Management**:
    *   Implement context window summarization techniques to handle long conversations.
    *   Explore using vector stores for semantic searching of conversation history or external documentation, providing more relevant context to the LLM.
*   **Dynamic Tool Loading**: Allow tools to be loaded dynamically (e.g., as plugins), making the agent more extensible without modifying core code.
*   **Command Prompt Integration**: Fully integrate the "command prompts" defined in `src/prompts/commands.py` (like `new_task`, `condense`) into the agent's workflow, allowing for more nuanced interactions beyond simple tool calls.
*   **Improved Error Handling**: Refine error handling mechanisms between tools, the agent, and the LLM, providing clearer feedback and potentially enabling automated recovery strategies.
*   **User Interface**: While currently CLI-based, future iterations could explore richer terminal interfaces (e.g., using libraries like `Textualize`) or even web UIs.
*   **Testing Framework**: Establish a comprehensive automated testing suite (`pytest`) covering unit, integration, and potentially end-to-end tests.
