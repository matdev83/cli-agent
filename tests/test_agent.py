import pytest
import json
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from typing import List, Dict, Any

from src.agent import DeveloperAgent
from src.llm import MockLLM
from src.tools.tool_protocol import Tool
from src.assistant_message import ToolUse

class MockTool(Tool):
    def __init__(self, name="mock_tool", execute_return_value="Result from mock_tool"):
        self._name = name
        self._description = f"A mock tool named {name}."
        self._parameters = [{"name": "param1", "description": "A parameter", "type": "string", "required": False}]
        self.execute_fn = MagicMock(return_value=execute_return_value)

    @property
    def name(self) -> str: return self._name
    @property
    def description(self) -> str: return self._description
    @property
    def parameters(self) -> list[dict[str, str]]: return self._parameters

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        return self.execute_fn(params, agent_memory)

def test_agent_simple_text_response(tmp_path: Path):
    mock_llm_responses = ["Just some text output, task complete."]
    agent = DeveloperAgent(send_message=MockLLM(mock_llm_responses).send_message, cwd=str(tmp_path))
    result = agent.run_task("A simple text task.")
    assert result == "Just some text output, task complete."
    assert len(agent.history) == 3

def test_agent_one_tool_call_then_completion(tmp_path: Path):
    mock_llm_responses = [
        "<tool_use><tool_name>read_file</tool_name><params><path>file.txt</path></params></tool_use>",
        "<text_content>Okay, I have read the file.</text_content><tool_use><tool_name>attempt_completion</tool_name><params><result>Read file.txt successfully.</result></params></tool_use>"
    ]
    agent = DeveloperAgent(send_message=MockLLM(mock_llm_responses).send_message, cwd=str(tmp_path))

    original_read_file_tool = agent.tools_map.get("read_file")
    assert original_read_file_tool is not None

    expected_file_content = "Content of file.txt"
    with patch.object(original_read_file_tool, 'execute', return_value=expected_file_content) as mock_execute:
        result = agent.run_task("Read file.txt for me.")

        mock_execute.assert_called_once_with({"path": "file.txt"}, agent_memory=agent)
        assert result == "Okay, I have read the file.\nRead file.txt successfully."

        assert len(agent.history) == 5
        assert agent.history[3]["role"] == "user"
        assert f"Result of read_file:\n{expected_file_content}" == agent.history[3]["content"]

def test_agent_max_steps_reached(tmp_path: Path):
    tool_call_response = "<tool_use><tool_name>read_file</tool_name><params><path>f.txt</path></params></tool_use>"
    mock_llm_responses = [tool_call_response] * 5

    agent = DeveloperAgent(send_message=MockLLM(mock_llm_responses).send_message, cwd=str(tmp_path))

    original_read_file_tool = agent.tools_map.get("read_file")
    assert original_read_file_tool is not None

    with patch.object(original_read_file_tool, 'execute', return_value="mock content"):
        result = agent.run_task("A task that will exceed max steps.", max_steps=3)
        assert result == "Max steps reached without completion."
        assert len(agent.history) == 8

def test_agent_handles_unknown_tool_from_llm(tmp_path: Path):
    # MockLLM provides one response (unknown tool), then will return None.
    mock_llm_responses = ["<tool_use><tool_name>unknown_tool_xyz</tool_name><params><p>1</p></params></tool_use>"]
    agent = DeveloperAgent(send_message=MockLLM(mock_llm_responses).send_message, cwd=str(tmp_path))

    # Agent processes unknown tool, adds error to history.
    # Then loops, calls send_message again, MockLLM returns None.
    # Agent should then return the "LLM did not provide a response" message.
    result = agent.run_task("Do something with an unknown tool.")

    expected_no_reply_message = "LLM did not provide a response. Ending task."
    assert result == expected_no_reply_message

    # History:
    # 1. System (initial prompt)
    # 2. User (task: "Do something...")
    # 3. Assistant (LLM says: <unknown_tool_xyz>...)
    # 4. User (agent adds result: "Error: Unknown tool 'unknown_tool_xyz'...")
    # 5. System (agent adds: "LLM did not provide a response...")
    assert len(agent.history) == 5
    assert "Error: Unknown tool 'unknown_tool_xyz'" in agent.history[3]['content']
    assert agent.history[4]['role'] == "system"
    assert agent.history[4]['content'] == expected_no_reply_message

def test_agent_llm_runs_out_of_responses(tmp_path: Path):
    # MockLLM provides one response, then will return None.
    mock_llm_responses = ["<tool_use><tool_name>read_file</tool_name><params><path>f.txt</path></params></tool_use>"]
    agent = DeveloperAgent(send_message=MockLLM(mock_llm_responses).send_message, cwd=str(tmp_path))

    original_read_file_tool = agent.tools_map.get("read_file")
    assert original_read_file_tool is not None

    with patch.object(original_read_file_tool, 'execute', return_value="mock content"):
        # Agent gets one tool call, processes it. Adds result to history.
        # Then calls send_message again. MockLLM runs out of responses, returns None.
        # run_task should catch this and return the specific message.
        result = agent.run_task("Task that needs two LLM steps.") # max_steps defaults to 20

    expected_no_reply_message = "LLM did not provide a response. Ending task."
    assert result == expected_no_reply_message

    # History:
    # 1. System (initial prompt)
    # 2. User (task: "Task that needs...")
    # 3. Assistant (LLM says: <read_file>...)
    # 4. User (agent adds result of read_file: "mock content")
    # 5. System (agent adds: "LLM did not provide a response...")
    assert len(agent.history) == 5
    assert agent.history[4]['role'] == "system"
    assert agent.history[4]['content'] == expected_no_reply_message


def test_agent_handles_execute_command_approval_json(tmp_path: Path):
    command_to_run = "rm -rf /"
    mock_llm_responses = [
        f"<tool_use><tool_name>execute_command</tool_name><params><command>{command_to_run}</command><requires_approval>true</requires_approval></params></tool_use>",
        "<text_content>Okay, command needs approval.</text_content><tool_use><tool_name>attempt_completion</tool_name><params><result>Acknowledged approval request for rm -rf /</result></params></tool_use>"
    ]
    agent = DeveloperAgent(send_message=MockLLM(mock_llm_responses).send_message, cwd=str(tmp_path), auto_approve=False)

    result = agent.run_task("Execute a dangerous command.")

    assert result == "Okay, command needs approval.\nAcknowledged approval request for rm -rf /"

    assert len(agent.history) == 5
    tool_result_message_content = agent.history[3]['content']

    assert f"Result of execute_command:" in tool_result_message_content
    assert '"success": false' in tool_result_message_content
    assert f"Error: Command '{command_to_run}' requires approval" in tool_result_message_content
```
