from __future__ import annotations

import os
import json
import traceback
from typing import Callable, Dict, List, Any, Optional # Added Optional
from pathlib import Path

from .memory import Memory
from .assistant_message import parse_assistant_message, ToolUse, TextContent
from .prompts.system import get_system_prompt

from src.tools import (
    Tool,
    ReadFileTool,
    WriteToFileTool,
    ReplaceInFileTool,
    ListFilesTool,
    SearchFilesTool,
    ExecuteCommandTool,
    ListCodeDefinitionsTool,
    BrowserActionTool,
    UseMCPTool,
    AccessMCPResourceTool,
    NewTaskTool,
    CondenseTool,
    ReportBugTool,
    NewRuleTool
)

class DeveloperAgent:
    """Simple developer agent coordinating tools and LLM messages."""

    def __init__(
        self,
        # Updated to reflect that send_message can return Optional[str] due to LLMWrapper protocol
        send_message: Callable[[List[Dict[str, str]]], Optional[str]],
        *,
        cwd: str = ".",
        auto_approve: bool = False,
        supports_browser_use: bool = False,
        browser_settings: dict | None = None,
        mcp_servers_documentation: str = "(No MCP servers currently connected)"
    ) -> None:
        self.send_message = send_message
        self.cwd: str = os.path.abspath(cwd)
        self.auto_approve: bool = auto_approve
        self.supports_browser_use: bool = supports_browser_use

        self.memory = Memory()
        self.history: List[Dict[str, str]] = self.memory.history

        tool_instances: List[Tool] = [
            ReadFileTool(), WriteToFileTool(), ReplaceInFileTool(), ListFilesTool(),
            SearchFilesTool(), ExecuteCommandTool(), ListCodeDefinitionsTool(),
            BrowserActionTool(), UseMCPTool(), AccessMCPResourceTool(), NewTaskTool(),
            CondenseTool(), ReportBugTool(), NewRuleTool()
        ]
        self.tools_map: Dict[str, Tool] = {tool.name: tool for tool in tool_instances}

        os.chdir(self.cwd)

        system_prompt_str = get_system_prompt(
            tools=self.tools_map.values(),
            cwd=self.cwd,
            supports_browser_use=self.supports_browser_use,
            browser_settings=browser_settings,
            mcp_servers_documentation=mcp_servers_documentation
        )
        self.memory.add_message("system", system_prompt_str)

    def _run_tool(self, tool_use: ToolUse) -> str:
        tool_name = tool_use.name
        tool_params = tool_use.params

        if tool_name == "attempt_completion":
            return tool_params.get("result", "Task considered complete by the assistant.")

        tool_to_execute = self.tools_map.get(tool_name)

        if tool_to_execute is None:
            return f"Error: Unknown tool '{tool_name}'. Please choose from the available tools."

        try:
            result_str = tool_to_execute.execute(tool_params, agent_memory=self)

            if tool_name == "read_file" and not result_str.startswith("Error:"):
                file_path_param = tool_params.get("path")
                if file_path_param:
                    abs_path_for_memory = Path(file_path_param)
                    if not abs_path_for_memory.is_absolute():
                        abs_path_for_memory = Path(self.cwd) / abs_path_for_memory
                    self.memory.add_file_context(str(abs_path_for_memory.resolve()), result_str)
            return result_str

        except NotImplementedError as e:
            print(f"Tool '{tool_name}' is not fully implemented: {e}")
            return f"Note: Tool '{tool_name}' is recognized but not fully implemented. {str(e)}"
        except ValueError as e:
            print(f"ValueError during execution of tool '{tool_name}': {e}\n{traceback.format_exc()}")
            return f"Error: Tool '{tool_name}' encountered a value error. Reason: {str(e)}"
        except Exception as e:
            print(f"Unexpected error during execution of tool '{tool_name}': {e}\n{traceback.format_exc()}")
            return f"Error: Tool '{tool_name}' failed to execute. Reason: {str(e)}"


    def run_task(self, user_input: str, max_steps: int = 20) -> str:
        """Run the agent loop until attempt_completion or step limit reached."""
        self.memory.add_message("user", user_input)
        for _i in range(max_steps):
            assistant_reply = self.send_message(self.history)

            if assistant_reply is None:
                no_reply_message = "LLM did not provide a response. Ending task."
                # Add a system message to history for context, or perhaps 'assistant' role
                # to indicate the assistant (LLM) failed to respond.
                self.memory.add_message("system", no_reply_message)
                return no_reply_message

            self.memory.add_message("assistant", assistant_reply)

            # parse_assistant_message expects a string. If assistant_reply could be None,
            # this was a potential point of failure. Now guarded by the None check above.
            parsed_responses = parse_assistant_message(assistant_reply)

            text_content_parts = [p.content for p in parsed_responses if isinstance(p, TextContent)]
            final_text_response = "\n".join(text_content_parts).strip()

            tool_uses = [p for p in parsed_responses if isinstance(p, ToolUse)]

            if not tool_uses:
                return final_text_response if final_text_response else "No further action taken."

            tool_to_run = tool_uses[0]

            if tool_to_run.name == "attempt_completion":
                completion_result = tool_to_run.params.get("result", "")
                # If there was text before the attempt_completion, prepend it.
                if final_text_response:
                    return f"{final_text_response}\n{completion_result}".strip()
                return completion_result

            tool_result_text = self._run_tool(tool_to_run)

            self.memory.add_message("user", f"Result of {tool_to_run.name}:\n{tool_result_text}")

        return "Max steps reached without completion."

# Example of how DeveloperAgent might be run (conceptual)
if __name__ == '__main__':
    from src.llm import MockLLM # For the example

    def mock_send_message_for_agent_constructor(history: List[Dict[str, str]]) -> Optional[str]:
        # This is the function that will be wrapped by MockLLM instance's send_message
        # For the DeveloperAgent constructor, we pass the method of an LLM instance.
        # This __main__ block is for conceptual testing of DeveloperAgent.
        # The actual MockLLM().send_message will handle response exhaustion.
        print("\n--- Mock LLM Call (via agent's send_message) ---")
        for msg in history[-2:]:
            print(f"  {msg['role'].upper()}: {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}")

        # This basic mock doesn't use a response list like MockLLM.
        # It's just to satisfy the callable type for the example.
        # Actual tests use MockLLM instance.
        last_user_message_content = history[-1]['content'] if history and history[-1]['role'] == 'user' else ""
        if "list files in src" in last_user_message_content:
            return "<tool_use><tool_name>list_files</tool_name><params><path>src</path></params></tool_use>"
        return "<text_content>Default mock response for __main__.</text_content>"

    # This example needs a proper MockLLM instance for send_message
    # to test exhaustion.
    # Let's refine the example to use MockLLM correctly.

    Path("src").mkdir(exist_ok=True)
    # Create a dummy agent.py for read_file test in the mock, if other tests need it
    (Path("src") / "agent.py").write_text("class DeveloperAgent:\n  pass # Dummy for tests")

    print("--- Testing Agent with Response Exhaustion ---")
    exhausted_responses = ["<tool_use><tool_name>read_file</tool_name><params><path>src/agent.py</path></params></tool_use>"]
    exhausted_llm = MockLLM(exhausted_responses)
    agent_exhaustion_test = DeveloperAgent(send_message=exhausted_llm.send_message, cwd=".")

    try:
        # First call consumes the only response.
        result1 = agent_exhaustion_test.run_task("Read agent.py", max_steps=2)
        print(f"Result 1 (exhaustion test): {result1}")
        # Second call to run_task or if max_steps allowed further loop, send_message would return None.
        # The current run_task structure with max_steps=2:
        # 1. User: "Read agent.py"
        # 2. Assistant: (tool_call read_file) - uses the response
        # 3. User: (tool_result)
        # (Loop 2 of 2)
        # 4. Assistant: send_message() -> MockLLM returns None
        #    run_task should catch this and return "LLM did not provide a response..."
        assert result1 == "LLM did not provide a response. Ending task."
        assert agent_exhaustion_test.history[-1]["content"] == "LLM did not provide a response. Ending task."
        assert agent_exhaustion_test.history[-1]["role"] == "system"

    except Exception as e:
        print(f"Error during agent exhaustion test: {e}\n{traceback.format_exc()}")
    finally:
        if (Path("src") / "agent.py").exists():
             (Path("src") / "agent.py").unlink()
        try: Path("src").rmdir()
        except OSError: print("Note: src directory may not be empty or other tests use it.")
        pass
