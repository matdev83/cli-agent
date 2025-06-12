# Plan for CLI-based Software Development Agent

This repository contains references to two projects under `vendor/`:

- **Cline** – a VSCode extension implementing an agentic coding assistant. It includes
  complex prompts, tool definitions and the agent loop for interacting with the
  user via VSCode. Key files:
  - `vendor/cline/src/core/prompts/system.ts` – defines the system prompt and
    detailed documentation for all tools such as `execute_command`, `read_file`,
    `write_to_file`, `replace_in_file`, `search_files`, `browser_action`, etc.
  - `vendor/cline/src/core/prompts/commands.ts` – prompt templates for special
    commands (`new_task`, `condense`, ...).
  - `vendor/cline/src/core/assistant-message` – logic to parse the assistant
    output into text blocks and structured *tool_use* blocks.
  - `vendor/cline/src/core/task/index.ts` – implements the full task loop inside
    VSCode (sending user input to the LLM, executing tool calls, tracking
    checkpoints, etc.).

After evaluating the CrewAI framework found in `vendor/crewAI`, we concluded that
its abstractions are not required for a minimal command-line agent. The project
will instead implement its own lightweight orchestration layer in Python.

## Goal
Create a minimal viable product of a CLI development agent that works outside of
VSCode. The new agent should reuse Cline's prompting approach and tool semantics
while implementing its own conversation loop for LLM interaction.

## High level architecture
1. **CLI entry point** – a Python script (e.g. `src/cli.py`) using `click` or
   `argparse` to accept the user request. It bootstraps a `DeveloperAgent`
   configured with our prompts and tools, then enters a loop until the task is
   complete.
2. **Agent definition** – implement a simple `DeveloperAgent` class that loads
   Cline's `SYSTEM_PROMPT` from `vendor/cline/src/core/prompts/system.ts` and
   coordinates tool usage. The agent operates in a sequential tool-use loop.
3. **Tool implementations** – Python classes deriving from a lightweight `Tool`
   base implementing the operations described in the prompts:
   - `execute_command` – run shell commands with optional approval.
   - File operations: `read_file`, `write_to_file`, `replace_in_file`.
   - Project exploration: `search_files`, `list_files`, `list_code_definition_names`.
   - Additional tools such as `browser_action`, `use_mcp_tool` and
     `access_mcp_resource` can be stubbed initially.
4. **Assistant message parser** – port Cline's TypeScript logic from
   `parse-assistant-message.ts` to Python in order to detect `<tool>` blocks in
   the model response. This lets the loop know which tool to execute and what
   parameters were supplied.
5. **Conversation loop** – similar to Cline's `initiateTaskLoop` in the Task
   class. The loop will:
   - Send the user request (plus context) to the LLM.
   - Parse the assistant reply for tool uses.
   - Execute the tool and append the result to the conversation history.
   - Continue until the assistant calls `attempt_completion` or a max iteration
     limit is reached.
6. **Context & Memory** – maintain conversation state in memory (e.g. a JSON log).
   File exploration and search results can be stored as context blocks similar to
   how Cline does.
7. **MCP integration** – when `use_mcp_tool` or `access_mcp_resource` is
   requested, communicate with an MCP server. Documentation for building such
   servers is loaded via the `load_mcp_documentation` tool from Cline.
8. **User approvals & safety** – replicate Cline's human‑in‑the‑loop approach by
   prompting the user before running risky commands (`requires_approval=true`).
9. **Testing & examples** – provide example workflows and basic unit tests for
   the parser and tools. Include a sample `Makefile` or `tox` config to run
   tests.

## Suggested implementation steps
1. **Set up project structure** in `src/` for Python modules and CLI entry.
2. **Port prompts** – convert `system.ts` and command prompts to Python multiline
   strings. Keep the tool descriptions intact.
3. **Implement tool classes** in Python matching the XML interface defined in the
   prompts. Initially implement core file and command tools; browser and MCP
   tools can be added later.
4. **Write assistant message parser** based on Cline's `parseAssistantMessageV2`
   to extract tool calls from the model output.
5. **Create the developer agent** configured with the system prompt and tools.
   Repeatedly send/receive messages until completion using the custom loop.
6. **Build CLI script** to accept a user task, run the agent loop and print tool
   execution results along the way. Provide options to auto‑approve commands or
   require confirmation.
7. **Iterate** – add features such as MCP integration, browser actions and
   context summarization as needed.

## Next steps
The plan above outlines the major tasks required to adapt the Cline extension
into a standalone CLI agent. Start by porting the prompts and parsing logic,
then gradually implement tools and the conversation loop in Python. With a
minimal orchestration layer we can focus on mapping Cline's capabilities into
this new environment.
