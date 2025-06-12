# Code Review for CLI Software Development Agent MVP

## 1. General Observations

The project successfully implements the core requirements for an MVP CLI software development agent, largely following the `CODEX_PLAN.md`. It reuses Cline's prompting style and tool interaction paradigm. The codebase is relatively small and generally understandable. Key functionalities like agent loop, LLM interaction (mock and OpenRouter), and basic tool execution (file operations, command execution) are in place.

However, as an MVP with tasks marked "done", there are several areas that need attention to mature the codebase, improve robustness, maintainability, and adherence to software engineering best practices as outlined in the project description.

## 2. Architectural Concerns

### AC1: Agent Tool Dispatch
- **Observation:** `DeveloperAgent._run_tool` uses a long series of `if/elif/else` statements to dispatch tool calls. This is a common anti-pattern that violates the Open/Closed Principle (SOLID).
- **Recommendation:** Refactor to a more extensible dispatch mechanism. Options:
    - **Dictionary-based dispatch:** Map tool names to handler methods.
    - **Strategy Pattern:** Define a `Tool` interface and have concrete tool classes implement it. The agent would then call `tool.run(params)`. This is a more robust and SOLID-aligned approach.

### AC2: Tool Abstraction
- **Observation:** Tools are currently implemented as standalone functions. While `src/tools/__init__.py` groups them, there's no common interface or base class for tools. This makes it harder to enforce a consistent contract for tool execution, parameter handling, or metadata (like descriptions, parameter definitions for dynamic help generation).
- **Recommendation:** Introduce a `Tool` abstract base class (ABC) or a protocol that all tools must implement. This class could define methods like `execute(params: Dict) -> str` and properties like `name`, `description`, `parameters_schema`. The `DeveloperAgent` would then interact with these `Tool` objects.

### AC3: Prompts Management
- **Observation:** System prompt construction in `src/prompts/system.py` uses string replacements and regex for conditional sections (browser, MCP). While functional, this can become unwieldy and error-prone as complexity grows. `prompts/commands.py` contains large multi-line strings for specific command responses, which is acceptable but could be managed more systematically if more commands are added.
- **Recommendation:**
    - For `system.py`: Consider using a proper templating engine (e.g., Jinja2) for managing the system prompt. This would make conditional logic and variable substitution cleaner.
    - For `commands.py`: For now, it's acceptable. If the number of such command prompts grows, consider loading them from separate files or a more structured format.

### AC4: LLM Abstraction
- **Observation:** `MockLLM` and `OpenRouterLLM` both implement `send_message`. This is good. However, there's no formal interface or ABC ensuring this contract.
- **Recommendation:** Define an `LLMWrapper` ABC or Protocol with the `send_message` method to formalize this. This helps if more LLM backends are added in the future.

## 3. Module-specific Feedback

### `src/agent.py` (DeveloperAgent)
- **SF1: `_to_bool` utility:**
    - **Observation:** `_to_bool` converts strings like "true", "1", "yes" to `True`. This is fragile as LLM outputs for booleans might vary. The system prompt for `execute_command` specifies 'true' or 'false'.
    - **Recommendation:** Strictly expect 'true' or 'false' as per the prompt, or make the parsing more robust if flexibility is desired (though strictness is often better with LLMs). Consider moving this to a common utilities module if used elsewhere.
- **SF2: Hardcoded tool knowledge in `_run_tool`:**
    - **Observation:** Besides dispatch, `_run_tool` has specific logic for parameter extraction and pre/post-processing for some tools (e.g., `read_file` updating memory, `replace_in_file` and `write_to_file` not returning content directly to LLM).
    - **Recommendation:** If a `Tool` ABC is introduced (AC2), this logic should reside within the respective tool's implementation. The tool's `execute` method would handle its specific parameters and side effects (like updating memory, which could be passed as a dependency to tools that need it).
- **SF3: `auto_approve` handling:**
    - **Observation:** `auto_approve` is passed down to `execute_command`. This is fine.
- **SF4: `supports_browser_use`:**
    - **Observation:** This flag is used to modify the system prompt. The stubs for browser tools exist. This seems correctly handled for an MVP.
- **SF5: Error Handling in `_run_tool`:**
    - **Observation:** If a tool is unknown, it raises `ValueError`. If a tool itself fails (e.g., `replace_in_file` if search block not found), that exception propagates.
    - **Recommendation:** Ensure consistent error reporting back to the LLM. Tool execution failures should be caught and formatted as a user message indicating the tool failed and why.

### `src/assistant_message.py`
- **SF6: XML Parsing:**
    - **Observation:** `parse_assistant_message` uses string searching (`find`, `find_all`) to parse the XML-like tool calls. This is highly susceptible to errors if the LLM output deviates even slightly from the expected format (e.g., extra spaces, different quoting for attributes if they were used). It also doesn't handle XML entities (e.g., `&lt;`, `&amp;`) within parameter values.
    - **Recommendation:** Use a lightweight, robust XML parser (e.g., Python's built-in `xml.etree.ElementTree`). This would handle variations in whitespace, ensure proper tag matching, and decode entities correctly. The current prompt format is simple enough that a full XML parser might seem like overkill, but it significantly improves robustness.
- **SF7: `TOOL_USE_NAMES` and `TOOL_PARAM_NAMES`:**
    - **Observation:** These are global lists. If tools become more dynamic (AC2), these lists would need to be generated or managed differently.
    - **Recommendation:** With a `Tool` ABC, each tool could declare its name and parameters. The parser could then validate against known tools.

### `src/cli.py`
- **SF8: Model Selection:**
    - **Observation:** `--model` argument defaults to "mock" and the help string says "only 'mock' supported", but the code implements `OpenRouterLLM`.
    - **Recommendation:** Update help string to reflect available models. Consider making model selection more flexible (e.g., using a factory or configuration).
- **SF9: `OPENROUTER_API_KEY` Handling:**
    - **Observation:** API key is read from `os.environ`. This is standard. Error is raised if not set when `OpenRouterLLM` is used.
    - **Recommendation:** Good practice.
- **SF10: Error Handling in `main`:**
    - **Observation:** A generic `except Exception` catches errors from `run_agent`.
    - **Recommendation:** While okay for an MVP, more specific error handling could provide better user feedback for common issues (e.g., invalid mock file path, API connection errors).

### `src/llm.py`
- **SF11: `MockLLM` responses:**
    - **Observation:** If `_index` goes beyond `_responses`, it returns `""`.
    - **Recommendation:** This might be unexpected. Consider raising an error or returning a specific "no more mock responses" message if this state is reached, as it usually indicates a test setup issue or a longer-than-expected conversation.
- **SF12: OpenRouterLLM Headers:**
    - **Observation:** `HTTP-Referer` and `X-Title` are hardcoded.
    - **Recommendation:** If these need to be configurable, move them to constants or configuration. For now, it's fine.

### `src/memory.py`
- **SF13: Simplicity:**
    - **Observation:** `Memory` is a straightforward dataclass. It serves its purpose for the MVP.
    - **Recommendation:** No immediate changes needed. Future enhancements might involve persistence, summarization, or more complex context management, which would require significant changes here.

### `src/prompts/system.py` (also see AC3)
- **SF14: Prompt Brittleness:**
    - **Observation:** The prompt relies on very specific phrasing and formatting (e.g., `${cwd.toPosix()}`). Small deviations in the template processing could break the prompt.
    - **Recommendation:** Using a templating engine (AC3) would mitigate this. Thorough testing of prompt generation is key.

### `src/prompts/commands.py`
- **SF15: Missing Implementation Link:**
    - **Observation:** This file defines prompt templates for commands like `new_task`, `condense`, `new_rule`, `report_bug`. However, the `DeveloperAgent` and `assistant_message.py` do not seem to be set up to specifically recognize or trigger these "commands" based on user input. The `TOOL_USE_NAMES` list includes them, implying the LLM might emit them as tool calls. The `CODEX_PLAN.md` mentions "prompt templates for special commands (`new_task`, `condense`, ...)" as part of porting Cline. It's unclear how a user would invoke these or how the agent decides the LLM should use `new_task` vs. `attempt_completion`.
    - **Recommendation:** Clarify the workflow for these "command prompts". Are they invoked by specific user inputs? Does the agent preprocess user input to switch to these modes? If the LLM is expected to call `<new_task>`, how is that different from any other tool? The current agent loop seems to treat all LLM responses with tags as standard tool calls.

### `src/tools/*`
- **SF16: `tools/code.py` - `list_code_definition_names`:**
    - **Observation:** Only supports Python (`.py`). The `CODEX_PLAN.md` mentions this tool generally.
    - **Recommendation:** Document the current Python-only limitation. For future enhancements, consider using more generic parsing libraries or ctags-like functionality if broader language support is needed. The output format is custom; ensure the LLM understands it from the prompt.
- **SF17: `tools/code.py` - Stubs:**
    - **Observation:** `browser_action`, `use_mcp_tool`, `access_mcp_resource` correctly raise `NotImplementedError`.
    - **Recommendation:** This is good for MVP. Ensure system prompt accurately reflects that these might not be fully functional if `supports_browser_use` is true but the tool is still just a stub.
- **SF18: `tools/file.py` - `replace_in_file`:**
    - **Observation:** The custom diff block parsing is specific. If `search` block isn't found, it raises `ValueError`.
    - **Recommendation:** This error should be caught by the agent and relayed to the LLM as a tool execution failure message. The LLM needs to be very precise with the `SEARCH` block, as per the prompt.
- **SF19: `tools/command.py` - `execute_command`:**
    - **Observation:** Handles approval well. Captures stdout/stderr.
    - **Recommendation:** Good.

## 4. Adherence to Project Standards

### SOLID:
- **Violated:** Open/Closed Principle in `DeveloperAgent._run_tool` (see AC1).
- **Opportunity:** Introducing a `Tool` ABC (AC2) and `LLMWrapper` ABC (AC4) would improve adherence to Liskov Substitution and Interface Segregation. Single Responsibility Principle is mostly okay, but `DeveloperAgent` does a bit too much regarding specific tool logic.

### DRY (Don't Repeat Yourself):
- **Generally Good:** Not much obvious repetition of code.
- **Minor:** The XML tag parsing logic in `assistant_message.py` involves similar start/end tag finding; a generalized XML helper could be slightly DRYer but a full XML parser is better (SF6).

### KISS (Keep It Simple, Stupid):
- **Mostly Good:** The components are individually quite simple.
- **Risk:** `DeveloperAgent._run_tool` could grow complex. The string-based parsing in `assistant_message.py` and prompt generation in `system.py` are simple now but risk becoming complicated with more features.

### Pythonic Conventions:
- **Generally Good:** Code uses Python features appropriately. Naming is generally Pythonic.
- **Minor:** `_to_bool` could be more Pythonic by directly checking `value.lower() == 'true'`.

## 5. Testability (TDD aspects)

- **Observation:** No test files (`tests/`) are visible in the provided `ls` output. The project description explicitly mentions TDD.
- **MockLLM:** The presence of `MockLLM` is excellent for TDD as it allows deterministic testing of the agent loop and tool interactions without actual LLM calls.
- **Tool Functions:** Individual tool functions in `tools/` are generally testable as they are pure functions or interact with the filesystem/subprocess in predictable ways.
- **`DeveloperAgent`:** Testing `DeveloperAgent` would heavily rely on `MockLLM` and mocking filesystem/subprocess calls for tools.
- **`assistant_message.py`:** `parse_assistant_message` is a pure function and should be highly testable with various input strings.
- **Recommendation:** Implement unit tests for all components, especially:
    - `assistant_message.parse_assistant_message` with various valid and malformed inputs.
    - `DeveloperAgent.run_task` using `MockLLM` and mocked tools to verify conversation flow and tool dispatch.
    - Each tool function with appropriate mocking for external interactions.
    - Prompt generation logic.

## 6. Specific Recommendations Summary

1.  **Refactor `DeveloperAgent._run_tool`:** Implement a dictionary-based dispatch or a Strategy pattern with a `Tool` ABC (AC1, AC2).
2.  **Introduce `Tool` ABC/Protocol:** Define a common interface for all tools (AC2).
3.  **Adopt XML Parser:** Replace string-based parsing in `assistant_message.py` with `xml.etree.ElementTree` or similar (SF6).
4.  **Use Templating Engine for System Prompt:** Manage `system.py` prompt complexity with Jinja2 or similar (AC3, SF14).
5.  **Formalize LLM Interface:** Create an `LLMWrapper` ABC/Protocol (AC4).
6.  **Clarify "Command Prompts" Workflow:** Define how prompts from `prompts/commands.py` are invoked and handled by the agent (SF15).
7.  **Implement Unit Tests:** Add comprehensive unit tests for all modules, leveraging `MockLLM` and mocking (Section 5).
8.  **Improve Error Reporting to LLM:** Ensure tool execution failures are consistently caught and reported back to the LLM in a structured way.
9.  **Review `_to_bool`:** Make it stricter or more robust based on expected LLM behavior (SF1).
10. **Update `cli.py` Help:** Reflect actual model support (SF8).

This review provides a roadmap for refactoring and hardening the MVP codebase. Addressing these points will significantly improve the agent's robustness, maintainability, and extensibility.
