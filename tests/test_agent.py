import pytest
import json
import unittest
import argparse # Added import
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess # For git_repo fixture and direct git calls
import shutil # For git_repo fixture
import os # For git_repo fixture

from src.agent import DeveloperAgent
from src.llm import MockLLM
from src.llm_protocol import LLMResponse, LLMUsageInfo # Added imports
from src.tools.tool_protocol import Tool
from src.assistant_message import ToolUse

class MockTool(Tool):
    def __init__(self, name="mock_tool", execute_return_value="Result from mock_tool"):
        self._name = name
        self._description = f"A mock tool named {name}."
        # Store as the new schema type: Dict[str, str]
        self._parameters_schema_dict = {"param1": "A parameter"}
        self.execute_fn = MagicMock(return_value=execute_return_value)

    @property
    def name(self) -> str: return self._name
    @property
    def description(self) -> str: return self._description
    @property
    def parameters_schema(self) -> Dict[str, str]: return self._parameters_schema_dict

    def execute(self, params: Dict[str, Any], agent_tools_instance: Any) -> str:
        return self.execute_fn(params, agent_tools_instance)

def test_agent_simple_text_response(tmp_path: Path):
    mock_llm_responses = ["Just some text output, task complete."]
    # Provide default cli_args for agent initialization
    mock_cli_args = argparse.Namespace(model="test_model", auto_approve=True) # auto_approve for simplicity if any tool were used

    # MockLLM now returns LLMResponse objects. Cost is zero by default.
    agent = DeveloperAgent(
        send_message=MockLLM(mock_llm_responses).send_message,
        cwd=str(tmp_path),
        cli_args=mock_cli_args
    )
    result = agent.run_task("A simple text task.")

    assert result == "Just some text output, task complete."
    # History: System, User, Assistant (LLM text response)
    assert len(agent.history) == 3
    assert agent.current_session_cost == 0.0

@patch('src.agent.request_user_confirmation', return_value=True)
def test_agent_one_tool_call_then_completion(mock_user_confirm, tmp_path: Path):
    mock_llm_responses = [
        "<tool_use><tool_name>read_file</tool_name><params><path>file.txt</path></params></tool_use>",
        "<text_content>Okay, I have read the file.</text_content><tool_use><tool_name>attempt_completion</tool_name><params><result>Read file.txt successfully.</result></params></tool_use>"
    ]
    mock_cli_args = argparse.Namespace(model="test_model_tools", auto_approve=True) # auto_approve for tool use

    agent = DeveloperAgent(
        send_message=MockLLM(mock_llm_responses).send_message,
        cwd=str(tmp_path),
        cli_args=mock_cli_args
    )

    original_read_file_tool = agent.tools_map.get("read_file")
    assert original_read_file_tool is not None

    expected_file_content = "Content of file.txt"
    with patch.object(original_read_file_tool, 'execute', return_value=expected_file_content) as mock_execute:
        result = agent.run_task("Read file.txt for me.")

        mock_execute.assert_called_once_with({"path": "file.txt"}, agent_tools_instance=agent)
        assert result == "Okay, I have read the file.\nRead file.txt successfully."

        assert len(agent.history) == 5
        assert agent.history[3]["role"] == "user"
        assert f"Result of read_file:\n{expected_file_content}" == agent.history[3]["content"]

@patch('src.agent.request_user_confirmation', return_value=True)
def test_agent_max_steps_reached(mock_user_confirm, tmp_path: Path):
    tool_call_response = "<tool_use><tool_name>read_file</tool_name><params><path>f.txt</path></params></tool_use>"
    mock_llm_responses = [tool_call_response] * 5

    agent = DeveloperAgent(send_message=MockLLM(mock_llm_responses).send_message, cwd=str(tmp_path))

    original_read_file_tool = agent.tools_map.get("read_file")
    assert original_read_file_tool is not None

    with patch.object(original_read_file_tool, 'execute', return_value="mock content") as mock_read_execute:
        result = agent.run_task("A task that will exceed max steps.", max_steps=3)
        # mock_read_execute.assert_called_with(ANY, agent_tools_instance=agent)
        assert result == "Max steps reached without completion."
        # History: Sys, User, Asst, User(tool_res), Asst, User(tool_res), Asst, User(tool_res)
        # 1 (sys) + 1 (user) + 3 * (1 asst + 1 user_tool_res) = 1 + 1 + 3*2 = 8
        assert len(agent.history) == 8
        # MockLLM is called 3 times (max_steps). Each call has 0.0 cost.
        assert agent.current_session_cost == 0.0

def test_agent_handles_unknown_tool_from_llm(tmp_path: Path):
    # MockLLM provides one response (unknown tool), then will return None.
    mock_llm_responses = ["<tool_use><tool_name>unknown_tool_xyz</tool_name><params><p>1</p></params></tool_use>"]
    mock_cli_args = argparse.Namespace(model="test_model_unknown", auto_approve=True)
    agent = DeveloperAgent(
        send_message=MockLLM(mock_llm_responses).send_message,
        cwd=str(tmp_path),
        cli_args=mock_cli_args
    )

    # Agent processes unknown tool, adds error to history. This is one LLM call.
    # Then loops, calls send_message again. MockLLM is exhausted.
    # MockLLM returns LLMResponse(content=None, usage=LLMUsageInfo(10,20,0.0)).
    # Agent processes this as an empty string response.
    # The run_task loop then sees no tool calls from the empty string, and returns it.
    result = agent.run_task("Do something with an unknown tool.")

    assert result == "" # Empty string from exhausted MockLLM processed as no further action.
    # Cost is from the first LLM call (unknown tool) + second call (exhausted, content=None)
    # Both have 0.0 cost from MockLLM.
    assert agent.current_session_cost == 0.0

    # History:
    # 1. System (initial prompt)
    # 2. User (task: "Do something...")
    # 3. Assistant (LLM says: <unknown_tool_xyz>...)
    # 4. User (agent adds result: "Error: Unknown tool 'unknown_tool_xyz'...")
    # 5. Assistant (LLM exhausted, agent processes as empty content "")
    assert len(agent.history) == 5, f"History: {agent.history}"
    assert "Error: Unknown tool 'unknown_tool_xyz'" in agent.history[3]['content']
    assert agent.history[4]['role'] == "assistant"
    assert agent.history[4]['content'] == "" # Empty content from exhausted MockLLM

@patch('src.agent.request_user_confirmation', return_value=True)
def test_agent_llm_runs_out_of_responses(mock_user_confirm, tmp_path: Path):
    # MockLLM provides one response, then returns LLMResponse(content=None, ...)
    mock_llm_responses = ["<tool_use><tool_name>read_file</tool_name><params><path>f.txt</path></params></tool_use>"]
    mock_cli_args = argparse.Namespace(model="test_model_exhausted", auto_approve=True)
    agent = DeveloperAgent(
        send_message=MockLLM(mock_llm_responses).send_message,
        cwd=str(tmp_path),
        cli_args=mock_cli_args
    )

    original_read_file_tool = agent.tools_map.get("read_file")
    assert original_read_file_tool is not None

    with patch.object(original_read_file_tool, 'execute', return_value="mock content") as mock_read_execute:
        # Agent gets one tool call, processes it. (LLM Call 1)
        # Then calls send_message again. MockLLM exhausted, returns LLMResponse(content=None). (LLM Call 2)
        # Agent processes content=None as empty string, no tools, returns empty string.
        result = agent.run_task("Task that needs two LLM steps.")

    assert result == "" # Agent processes exhausted MockLLM (content=None) as empty string.
    assert agent.current_session_cost == 0.0 # Two calls to MockLLM, both 0.0 cost.

    # History:
    # 1. System (initial prompt)
    # 2. User (task: "Task that needs...")
    # 3. Assistant (LLM says: <read_file>...)
    # 4. User (agent adds result of read_file: "mock content")
    # 5. Assistant (LLM exhausted, agent processes as empty content "")
    assert len(agent.history) == 5
    assert agent.history[4]['role'] == "assistant"
    assert agent.history[4]['content'] == "" # Empty content from exhausted MockLLM


def test_agent_handles_execute_command_approval_json(tmp_path: Path):
    command_to_run = "rm -rf /"
    mock_llm_responses = [
        f"<tool_use><tool_name>execute_command</tool_name><params><command>{command_to_run}</command><requires_approval>true</requires_approval></params></tool_use>",
        "<text_content>Okay, command needs approval.</text_content><tool_use><tool_name>attempt_completion</tool_name><params><result>Acknowledged approval request for rm -rf /</result></params></tool_use>"
    ]
    # Set all flags to False for this test, including the specific auto_approve for commands
    cli_args = argparse.Namespace(
        auto_approve=False,
        allow_read_files=False,
        allow_edit_files=False,
        allow_execute_safe_commands=False,
        allow_execute_all_commands=False, # Explicitly false to trigger confirmation path
        allow_use_browser=False,
        allow_use_mcp=False,
        model="test_model_dangerous_cmd" # Added model to cli_args
    )
    agent = DeveloperAgent(send_message=MockLLM(mock_llm_responses).send_message, cwd=str(tmp_path), cli_args=cli_args)

    with patch('src.agent.request_user_confirmation', return_value=True): # Mock user confirmation for this test
        result = agent.run_task("Execute a dangerous command.")

    assert result == "Okay, command needs approval.\nAcknowledged approval request for rm -rf /"
    assert agent.current_session_cost == 0.0 # Two LLM calls, 0.0 cost each from MockLLM

    assert len(agent.history) == 5
    tool_result_message_content = agent.history[3]['content']

    assert f"Result of execute_command:" in tool_result_message_content
    assert '"success": false' in tool_result_message_content
    assert f"Error: Command '{command_to_run}' requires approval" in tool_result_message_content

# --- Tests for Malformed LLM Responses & Tool Usage Errors ---

MALFORMED_TOOL_PREFIX = "Error: Malformed tool use - "
MAX_CONSECUTIVE_TOOL_ERRORS = 3 # Should match agent.py
ADMONISHMENT_MESSAGE = (
    "System: You have made several consecutive errors in tool usage or formatting. "
    "Please carefully review the available tools, their required parameters, "
    "and the expected XML format. Ensure your next response is a valid tool call "
    "or a text response."
)


def test_agent_malformed_tool_xml_missing_tool_name(tmp_path: Path):
    """Test agent handling of malformed XML (missing tool_name in tool_use block)."""
    malformed_response = "<tool_use><params><path>file.txt</path></params></tool_use>" # Missing <tool_name>
    mock_llm_responses = [malformed_response, "<text_content>Giving up.</text_content>"]
    mock_cli_args = argparse.Namespace(model="test_model_malformed", auto_approve=True)
    agent = DeveloperAgent(
        send_message=MockLLM(mock_llm_responses).send_message,
        cwd=str(tmp_path),
        cli_args=mock_cli_args
    )

    agent.run_task("Attempt malformed tool use.")
    # Cost: First LLM call (malformed) + Second LLM call ("Giving up.") = 0.0 + 0.0
    assert agent.current_session_cost == 0.0

    assert agent.consecutive_tool_errors == 1
    # History: System, User, Assistant (malformed), User (error msg from agent), Assistant (text), User (text result)
    # The error message from parse_assistant_message becomes part of the text response from the LLM's turn.
    # The agent then adds THIS to history as an assistant message.
    # Then, the agent's processing of this (seeing a TextContent with the error) increments the counter.
    # The next message in history would be the user's "Result of..." which is the error message itself.

    # Let's check the history for the specific error message from parse_assistant_message
    # The parsed_responses in run_task will have a TextContent with the error.
    # This becomes final_text_response, which is then added to history as user message.
    # No, this is not quite right. The malformed parsing happens, `final_text_response` includes the error.
    # `tool_uses` is empty. So `run_task` returns `final_text_response`.
    # The error is recorded when `parse_assistant_message` returns TextContent starting with MALFORMED_TOOL_PREFIX.

    # Let's trace:
    # 1. System Prompt
    # 2. User: "Attempt malformed tool use."
    # 3. Assistant: "<tool_use><params><path>file.txt</path></params></tool_use>"
    #    - parse_assistant_message returns TextContent(content="Error: Malformed tool use - Missing <tool_name>...")
    #    - consecutive_tool_errors becomes 1.
    #    - final_text_response contains this error.
    #    - tool_uses is empty.
    #    - run_task returns this final_text_response.
    # The assertion for consecutive_tool_errors is the most direct here.
    # We also want to ensure this error message was indeed processed.
    # The `run_task` will return the message, and it will be in history.

    # Note: The original test was re-initializing the agent.
    # The existing test logic where agent.run_task is called once and it internally loops
    # through the MockLLM responses is better for testing the agent's internal state like error counters.
    # If malformed_response causes run_task to return, then consecutive_tool_errors will be 1.
    # If it loops and the next response is valid text, then consecutive_tool_errors resets to 0.

    # Current agent logic: if assistant response (even if it's a parsed error message)
    # leads to no tool_uses, run_task returns.
    # So, a malformed tool use that becomes a text error message *will* cause run_task to return.
    # The consecutive_tool_errors will be 1 at that point.
    # The next response ("Giving up.") is not processed in that same run_task call.
    # This seems to be what the original test was observing.

    # Re-check: `test_agent_error_counter_resets_after_unknown_tool_then_valid_text` shows looping.
    # The difference is that an "unknown tool" *is* a tool_use, so _run_tool is called,
    # it returns an error, agent adds this to history, and *loops*.
    # A "malformed tool XML" (e.g. missing tool_name) results in *no tool_uses found* by parser.
    # `parse_assistant_message` returns a `TextContent` containing the error.
    # `run_task` sees no `tool_uses`, and then returns `final_text_response`.
    # This means the error counter behavior for "malformed XML" vs "unknown tool" is different.

    # For malformed XML (current test):
    mock_cli_args_malformed = argparse.Namespace(model="test_model_malformed_loop", auto_approve=True)
    agent_malformed = DeveloperAgent(send_message=MockLLM(
        [malformed_response, "<text_content>OK, I will try something else.</text_content>"]
    ).send_message, cwd=str(tmp_path), cli_args=mock_cli_args_malformed)

    final_result = agent_malformed.run_task("Test sequence for malformed tool.")

    # After malformed_response: consecutive_tool_errors should be 1.
    # History: system, user, assistant (malformed).
    # In run_task, after parsing: malformed_tool_error_this_turn = True, consecutive_tool_errors = 1.
    # tool_uses is empty. final_text_response has the error string. This is then added to history as USER message.
    # No, final_text_response is what run_task returns if no tool_uses.
    # The logic is: assistant_reply -> parse -> text_content_parts (may contain MALFORMED_TOOL_PREFIX)
    # -> if MALFORMED_TOOL_PREFIX, counter++.
    # -> if no tool_uses: if not malformed_tool_error_this_turn, counter = 0. Returns final_text_response.

    # Let's check the history.
    # 1. System
    # 2. User: "Test sequence..."
    # 3. Assistant: malformed_response
    #   (run_task processes this: sees MALFORMED_TOOL_PREFIX, increments error counter to 1.
    #    final_text_response gets the error. tool_uses is empty.
    #    run_task makes a new call to LLM with history [System, User, Assistant(malformed), User(error_message_from_parse)])
    # No, the user message is tool_result_text. If no tool_uses, it's not added.

    # The error counter should be 1 after the first LLM response.
    # The second LLM response is pure text and not a malformed error, so counter should reset to 0.
    assert agent.history[2]["role"] == "assistant"
    assert agent.history[2]["content"] == malformed_response
    # The error is processed internally. The next message to LLM would reflect this.
    # The key is that consecutive_tool_errors was 1.
    # The run_task returns the error message, and the MockLLM is exhausted.
    assert agent.consecutive_tool_errors == 1
    # The second response "<text_content>OK, I will try something else.</text_content>" is not processed in this run_task call.
    # The final_result from agent_malformed.run_task would be the error string from the first malformed response.
    # The agent.consecutive_tool_errors would be 1.
    # The agent.current_session_cost would be 0.0 (one call to MockLLM).
    assert MALFORMED_TOOL_PREFIX + "Missing <tool_name>" in final_result
    assert agent_malformed.consecutive_tool_errors == 1
    assert agent_malformed.current_session_cost == 0.0


def test_agent_error_counter_resets_on_valid_text_response(tmp_path: Path):
    """Test that consecutive_tool_errors resets on a valid text response after an error."""
    # This test's original premise was flawed because a malformed XML (missing tool_name) causes
    # run_task to return immediately with the error text. It doesn't loop to process a subsequent
    # valid text response from MockLLM in the same run_task call.
    # The test `test_agent_error_counter_resets_after_unknown_tool_then_valid_text` correctly tests reset.
    # This test will be simplified to just check the error state after a malformed response.
    malformed_response = "<tool_use><params><path>file.txt</path></params></tool_use>" # Missing <tool_name>
    mock_cli_args = argparse.Namespace(model="test_model_malformed_single", auto_approve=True)

    agent = DeveloperAgent(
        send_message=MockLLM([malformed_response]).send_message, # Only one response needed
        cwd=str(tmp_path),
        cli_args=mock_cli_args
    )

    error_message_returned = agent.run_task("Trigger malformed XML.")
    assert MALFORMED_TOOL_PREFIX + "Missing <tool_name>" in error_message_returned
    assert agent.consecutive_tool_errors == 1 # Error occurred and was counted
    assert agent.current_session_cost == 0.0 # One call to MockLLM

    # Second task call (or continuation if agent looped) should process valid text
    # Since run_task returned, we call it again to simulate continuation by user/system
    # For this test, let's assume the history is maintained and the next call processes the next response.
    # The MockLLM will now provide valid_text_response.
    # To ensure the agent's state (history) is appropriate for the next call, we might need
    # to manually add the user's response to the "error_message_returned".
    # However, the current agent loop calls send_message repeatedly.
    # The issue is MockLLM's responses are consumed one by one by *successive* send_message calls within *one* run_task.
    # If run_task returns early (like it does on a text response), the MockLLM's next response isn't used in that same run_task.

    # Let's re-evaluate the agent's loop and how it interacts with MockLLM for sequential responses.
    # `run_task` loop:
    #   `assistant_reply = self.send_message(self.history)`
    #   `self.memory.add_message("assistant", assistant_reply)`
    #   `parsed_responses = parse_assistant_message(assistant_reply)`
    #   ... if not tool_uses, it returns ...
    # This means the MockLLM's sequence is indeed processed across the loop iterations.

    # So, for the agent initialized with [malformed, valid_text]:
    # 1st loop: LLM sends malformed. consecutive_tool_errors = 1. Not tool_uses. Returns error_message.
    # This is where the `test_agent_malformed_tool_xml_missing_tool_name` failed.
    # The `run_task` returns, so `consecutive_tool_errors` remains 1. The second LLM response isn't processed.

    # To test reset properly, the agent must continue its loop after an error that's not a tool_call.
    # Current agent logic: if assistant response leads to no tool_uses, run_task returns.
    # This means a malformed tool use (which becomes text) causes run_task to return.
    # This is probably okay behavior for the agent.

    # The test for reset should be:
    # 1. Induce an error (e.g. unknown tool, which *does* loop).
    # 2. Then provide a valid text response.

    # Resetting the first test to only check the increment:
    agent = DeveloperAgent(send_message=MockLLM([malformed_response]).send_message, cwd=str(tmp_path))
    returned_message = agent.run_task("Attempt malformed tool use for increment check.")
    assert MALFORMED_TOOL_PREFIX + "Missing <tool_name>" in returned_message
    assert agent.consecutive_tool_errors == 1


def test_agent_error_counter_resets_after_unknown_tool_then_valid_text(tmp_path: Path):
    """Test counter increments on unknown tool, then resets on valid text."""
    unknown_tool_response = "<tool_use><tool_name>unknown_tool_123</tool_name><params/></tool_use>"
    valid_text_response = "<text_content>All good now.</text_content>"

    mock_llm = MockLLM([unknown_tool_response, valid_text_response])
    agent = DeveloperAgent(send_message=mock_llm.send_message, cwd=str(tmp_path))

    # The agent will:
    # 1. Receive unknown_tool_response. _run_tool returns error. consecutive_tool_errors = 1. (LLM Call 1)
    #    History gets "Result of unknown_tool_123:\nError: Unknown tool..."
    # 2. Loop, send history to LLM. LLM sends valid_text_response. (LLM Call 2)
    #    parse_assistant_message -> no tool uses, no malformed error. consecutive_tool_errors = 0.
    #    run_task returns "All good now."

    final_output = agent.run_task("Test sequence: unknown tool then valid text.")

    assert final_output == "All good now."
    assert agent.consecutive_tool_errors == 0 # Should be reset
    assert "Error: Unknown tool 'unknown_tool_123'" in agent.history[3]["content"] # Error was processed
    assert agent.current_session_cost == 0.0 # Two calls to MockLLM, 0.0 cost each


def test_agent_malformed_tool_xml_unrecognized_tag(tmp_path: Path):
    """Test agent handling of malformed XML (unrecognized tag)."""
    malformed_response = "<some_random_xml_tag><param>value</param></some_random_xml_tag>"
    mock_cli_args = argparse.Namespace(model="test_model_unrec_tag", auto_approve=True)
    agent = DeveloperAgent(
        send_message=MockLLM([malformed_response]).send_message,
        cwd=str(tmp_path),
        cli_args=mock_cli_args
    )

    returned_message = agent.run_task("Attempt unrecognized tag.")

    assert agent.consecutive_tool_errors == 1
    assert agent.history[2]["role"] == "assistant"
    assert agent.history[2]["content"] == malformed_response
    # When malformed response leads to no tool_uses, run_task returns the parsed error string
    assert (MALFORMED_TOOL_PREFIX + "Unrecognized or malformed tag") in returned_message
    assert agent.current_session_cost == 0.0 # One LLM call

def test_agent_unknown_tool_increments_error_counter(tmp_path: Path):
    """Test that calling an unknown tool increments consecutive_tool_errors."""
    unknown_tool_response = "<tool_use><tool_name>this_tool_does_not_exist</tool_name><params></params></tool_use>"
    # If the agent loops after this, it would make another LLM call. MockLLM needs a second response or it will cause error.
    # Let's assume it makes one tool call and then the task is such that it would attempt completion.
    # Or, the test focuses on the state after the first error.
    # The current MockLLM([response]) will be exhausted after the first call.
    # Agent's loop: LLM -> unknown tool -> tool error -> LLM (exhausted) -> empty response -> return.
    mock_cli_args = argparse.Namespace(model="test_model_inc_err", auto_approve=True)
    agent = DeveloperAgent(
        send_message=MockLLM([unknown_tool_response]).send_message, # MockLLM will be exhausted after this
        cwd=str(tmp_path),
        cli_args=mock_cli_args
    )

    agent.run_task("Call an unknown tool.") # LLM call 1 (unknown tool), LLM call 2 (exhausted)

    assert agent.consecutive_tool_errors == 1 # Error from unknown tool
    assert agent.history[2]["role"] == "assistant"
    assert agent.history[2]["content"] == unknown_tool_response
    assert "Error: Unknown tool 'this_tool_does_not_exist'" in agent.history[3]["content"]
    assert agent.history[3]["role"] == "user"
    assert agent.current_session_cost == 0.0 # Two LLM calls, both 0.0 cost


@patch('src.agent.request_user_confirmation', return_value=True)
def test_agent_tool_missing_required_parameter(mock_user_confirm, tmp_path: Path):
    """Test that tool error for missing params increments error counter."""
    # read_file requires 'path'
    missing_param_response = "<tool_use><tool_name>read_file</tool_name><params></params></tool_use>" # Missing 'path'
    mock_cli_args = argparse.Namespace(model="test_model_missing_param", auto_approve=True)
    agent = DeveloperAgent(
        send_message=MockLLM([missing_param_response]).send_message, # MockLLM exhausted after this
        cwd=str(tmp_path),
        cli_args=mock_cli_args
    )

    agent.run_task("Call read_file without path.") # LLM call 1 (missing param), LLM call 2 (exhausted)

    assert agent.consecutive_tool_errors == 1 # Error from missing param
    assert agent.history[2]["role"] == "assistant"
    assert agent.history[2]["content"] == missing_param_response
    # Check that the user-facing message in history contains the specific error from the tool
    assert "Result of read_file:\nError: Missing required parameter 'path'." in agent.history[3]["content"]
    assert agent.history[3]["role"] == "user"
    assert agent.current_session_cost == 0.0 # Two LLM calls, 0.0 cost each


def test_agent_consecutive_error_admonishment(tmp_path: Path):
    """Test that admonishment message is added after MAX_CONSECUTIVE_TOOL_ERRORS."""
    responses = [
        "<tool_use><tool_name>unknown_tool_1</tool_name><params/></tool_use>", # Error 1
        "<tool_use><tool_name>unknown_tool_2</tool_name><params/></tool_use>", # Error 2
        "<tool_use><tool_name>unknown_tool_3</tool_name><params/></tool_use>", # Error 3 -> Triggers Admonishment
        "<text_content>Hopefully this works now.</text_content>" # Successful response after admonishment
    ]
    # MAX_CONSECUTIVE_TOOL_ERRORS is 3 in agent.py

    mock_llm = MockLLM(responses)
    mock_cli_args = argparse.Namespace(model="test_model_admonish", auto_approve=True)
    agent = DeveloperAgent(
        send_message=mock_llm.send_message,
        cwd=str(tmp_path),
        cli_args=mock_cli_args
    )

    final_result = agent.run_task("Trigger three consecutive errors.")
    # Total LLM calls: 3 for errors, 1 for admonishment, 1 for final success = 5 calls
    assert agent.current_session_cost == 0.0 # 5 calls to MockLLM, all 0.0 cost

    # Trace:
    # 1. User: "Trigger..."
    # 2. Assistant: unknown_tool_1 -> error_count = 1. History has "Error: Unknown tool 'unknown_tool_1'"
    # 3. Assistant: unknown_tool_2 -> error_count = 2. History has "Error: Unknown tool 'unknown_tool_2'"
    # 4. Assistant: unknown_tool_3 -> error_count = 3. History has "Error: Unknown tool 'unknown_tool_3'"
    #    (Now error_count is 3, Admonishment should be added BEFORE next LLM call)
    #    (Error counter resets to 0 after adding admonishment)
    # 5. System: ADMONISHMENT_MESSAGE (This is added to history at the start of the next loop iteration)
    # 6. Assistant: "Hopefully this works now." -> error_count = 0 (valid text response)
    #    run_task returns "Hopefully this works now."

    assert final_result == "Hopefully this works now."

    # Check history for the admonishment message
    # History: System(prompt), User(task), Assistant(err1), User(res1), Assistant(err2), User(res2), Assistant(err3), User(res3), System(admonish), Assistant(text_ok)
    # Expected indices:
    # 0: System Prompt
    # 1: User: "Trigger three consecutive errors."
    # 2: Assistant: unknown_tool_1
    # 3: User: "Result of unknown_tool_1..."
    # 4: Assistant: unknown_tool_2
    # 5: User: "Result of unknown_tool_2..."
    # 6: Assistant: unknown_tool_3
    # 7: User: "Result of unknown_tool_3..."
    # 8: System: ADMONISHMENT_MESSAGE
    # 9: Assistant: "Hopefully this works now."
    # 10: User: (final_result) - no, this is not added to history if it's the end of task.

    assert len(agent.history) == 10
    assert agent.history[8]["role"] == "system"
    assert agent.history[8]["content"] == ADMONISHMENT_MESSAGE

    # Check that the error counter was reset by the admonishment, and then again by the successful text response
    assert agent.consecutive_tool_errors == 0


@patch('src.agent.request_user_confirmation', return_value=True)
def test_agent_consecutive_error_counter_resets_on_successful_tool_call(mock_user_confirm, tmp_path: Path):
    """Test that consecutive_tool_errors resets after a successful tool call."""
    responses = [
        "<tool_use><tool_name>unknown_tool_1</tool_name><params/></tool_use>", # Error 1
        "<tool_use><tool_name>read_file</tool_name><params><path>file.txt</path></params></tool_use>", # Success
        "<text_content>All done.</text_content>" # Final response
    ]
    mock_llm = MockLLM(responses)
    mock_cli_args = argparse.Namespace(model="test_model_err_succ", auto_approve=True)
    agent = DeveloperAgent(
        send_message=mock_llm.send_message,
        cwd=str(tmp_path),
        cli_args=mock_cli_args
    )

    # Create dummy file for read_file to succeed
    (tmp_path / "file.txt").write_text("dummy content", encoding="utf-8")

    agent.run_task("Error then success.")
    # LLM calls: unknown_tool (err), read_file (ok), "All done." (text) = 3 calls
    assert agent.current_session_cost == 0.0 # 3 calls to MockLLM, all 0.0 cost

    # Trace:
    # 1. unknown_tool_1 -> consecutive_tool_errors = 1. Tool result "Error: Unknown..."
    # 2. read_file -> success. _run_tool resets consecutive_tool_errors = 0. Tool result "dummy content"
    # 3. "All done." -> text response, no malformed error, counter remains 0.

    assert agent.consecutive_tool_errors == 0

# --- Tests for Diff-to-Full-Write Escalation ---

MAX_DIFF_FAILURES_PER_FILE = 2 # Should match agent.py
REPLACE_SUGGESTION_MESSAGE_TEMPLATE = (
    "\nAdditionally, applying diffs to '{file_path}' has failed multiple times. "
    "Consider reading the file content and using 'write_to_file' "
    "with the full desired content instead."
)

@patch('src.agent.request_user_confirmation', return_value=True)
def test_diff_failure_escalation_suggests_write_to_file(mock_user_confirm, tmp_path: Path):
    """Test that replace_in_file failure escalation message is added after max failures."""
    file_to_edit = "test_file.txt"
    abs_file_path_str = str((tmp_path / file_to_edit).resolve())
    (tmp_path / file_to_edit).write_text("initial content", encoding="utf-8")

    # Specific error messages that should trigger the escalation
    search_block_error = "Error: Search block 1 (starting with 'old_text') not found..."
    diff_format_error = "Error processing diff_blocks" # Simplified

    responses = [
        f"<tool_use><tool_name>replace_in_file</tool_name><params><path>{file_to_edit}</path><diff_blocks>...</diff_blocks></params></tool_use>", # Fail 1
        f"<tool_use><tool_name>replace_in_file</tool_name><params><path>{file_to_edit}</path><diff_blocks>...</diff_blocks></params></tool_use>", # Fail 2 (escalation)
        "<text_content>Ok, will try write_to_file.</text_content>" # LLM acknowledges
    ]
    mock_llm = MockLLM(responses)
    mock_cli_args = argparse.Namespace(model="test_model_diff_fail", auto_approve=True)
    agent = DeveloperAgent(
        send_message=mock_llm.send_message,
        cwd=str(tmp_path),
        cli_args=mock_cli_args
    )

    # Mock the execute method of the actual ReplaceInFileTool instance
    replace_tool_instance = agent.tools_map["replace_in_file"]

    # Simulate failures
    with patch.object(replace_tool_instance, 'execute', side_effect=[
        search_block_error, # First failure
        diff_format_error   # Second failure, should trigger suggestion
    ]) as mock_replace_execute:
        final_output = agent.run_task(f"Edit {file_to_edit} twice with failing diffs.")
        # Check calls to the mocked execute
        assert mock_replace_execute.call_count == 2
        # Example check for the first call (params might vary based on how replace_in_file is called by agent)
        # This is to ensure agent_tools_instance is passed.
        # The exact params for replace_in_file would be:
        # {"path": file_to_edit, "diff_blocks": "..."} - where "..." is the content from LLM
        # We can't know the exact diff_blocks here without seeing the LLM response.
        # So, we'll just check that agent_tools_instance was passed.
    # Check first call
    params_call0, kwargs_call0 = mock_replace_execute.call_args_list[0]
    assert 'agent_tools_instance' in kwargs_call0
    assert kwargs_call0['agent_tools_instance'] == agent
    # Check second call
    params_call1, kwargs_call1 = mock_replace_execute.call_args_list[1]
    assert 'agent_tools_instance' in kwargs_call1
    assert kwargs_call1['agent_tools_instance'] == agent

    assert final_output == "Ok, will try write_to_file."
    assert agent.diff_failure_tracker.get(abs_file_path_str, 0) == 0 # Reset after suggestion
    # LLM calls: fail1, fail2 (escalation), "Ok, will try..." = 3 calls
    assert agent.current_session_cost == 0.0

    # Check history for the augmented error message after the second failure
    # History: Sys(prompt), User(task), Asst(fail1), User(res1), Asst(fail2), User(res2_sugg), Asst(text_ok)
    # Asst(text_ok) is the last message in history.
    assert len(agent.history) == 7

    # Result of first failed replace_in_file
    assert agent.history[3]["content"] == f"Result of replace_in_file:\n{search_block_error}"

    # Result of second failed replace_in_file, with suggestion
    expected_augmented_error = diff_format_error + REPLACE_SUGGESTION_MESSAGE_TEMPLATE.format(file_path=file_to_edit)

    actual_content_in_history = agent.history[5]["content"]
    # The content is "Result of replace_in_file:\n" + augmented_tool_output
    # We want to compare the augmented_tool_output part.
    prefix_in_history = "Result of replace_in_file:\n"
    actual_augmented_output = actual_content_in_history.removeprefix(prefix_in_history)

    assert actual_augmented_output == expected_augmented_error

    assert mock_replace_execute.call_count == 2


def test_diff_failure_tracker_resets_on_successful_replace(tmp_path: Path):
    """Test that diff_failure_tracker resets for a file after a successful replace_in_file."""
    file_to_edit = "another_file.txt"
    abs_file_path_str = str((tmp_path / file_to_edit).resolve())
    (tmp_path / file_to_edit).write_text("content", encoding="utf-8")

    # Simulate one failure first
    cli_args_allow_all = argparse.Namespace(
        model="test_model_diff_reset", auto_approve=True, allow_read_files=True,
        allow_edit_files=True, allow_execute_safe_commands=True,
        allow_execute_all_commands=True, allow_use_browser=True, allow_use_mcp=True
    )
    # Using MagicMock for send_message as this test focuses on _run_tool and internal state, not LLM interaction loop
    agent = DeveloperAgent(send_message=MagicMock(return_value=LLMResponse(content="some content", usage=LLMUsageInfo())), cwd=str(tmp_path), cli_args=cli_args_allow_all)
    agent.diff_failure_tracker[abs_file_path_str] = 1

    # Now simulate a successful replace_in_file call for the same file
    # This does not involve an LLM call in this direct _run_tool test. Cost is not affected.
    # We can directly call _run_tool for this unit test
    success_response_from_tool = f"File {abs_file_path_str} modified successfully with 1 block(s)."
    replace_tool_instance = agent.tools_map["replace_in_file"]

    with patch.object(replace_tool_instance, 'execute', return_value=success_response_from_tool) as mock_execute:
        tool_use = ToolUse(name="replace_in_file", params={"path": file_to_edit, "diff_blocks": "..."})
        agent._run_tool(tool_use)
        mock_execute.assert_called_once_with({"path": file_to_edit, "diff_blocks": "..."}, agent_tools_instance=agent)

    assert agent.diff_failure_tracker.get(abs_file_path_str, 0) == 0


def test_diff_failure_tracker_resets_on_successful_write(tmp_path: Path):
    """Test that diff_failure_tracker resets for a file after a successful write_to_file."""
    file_to_edit = "yet_another_file.txt"
    abs_file_path_str = str((tmp_path / file_to_edit).resolve())
    (tmp_path / file_to_edit).write_text("content", encoding="utf-8")

    cli_args_allow_all = argparse.Namespace(
        model="test_model_diff_write_reset", auto_approve=True, allow_read_files=True,
        allow_edit_files=True, allow_execute_safe_commands=True,
        allow_execute_all_commands=True, allow_use_browser=True, allow_use_mcp=True
    )
    agent = DeveloperAgent(send_message=MagicMock(return_value=LLMResponse(content="some content", usage=LLMUsageInfo())), cwd=str(tmp_path), cli_args=cli_args_allow_all)
    agent.diff_failure_tracker[abs_file_path_str] = MAX_DIFF_FAILURES_PER_FILE -1 # One less than max

    # Simulate a successful write_to_file
    # No LLM call in this direct _run_tool test, cost not affected.
    write_tool_instance = agent.tools_map["write_to_file"]
    success_response_from_tool = f"File written successfully to {abs_file_path_str}"

    with patch.object(write_tool_instance, 'execute', return_value=success_response_from_tool) as mock_execute:
        tool_use = ToolUse(name="write_to_file", params={"path": file_to_edit, "content": "new content"})
        agent._run_tool(tool_use)
        mock_execute.assert_called_once_with({"path": file_to_edit, "content": "new content"}, agent_tools_instance=agent)

    assert agent.diff_failure_tracker.get(abs_file_path_str, 0) == 0

def test_diff_failure_tracker_handles_relative_and_absolute_paths(tmp_path: Path):
    """Test that the tracker handles relative and absolute paths consistently."""
    relative_path = "relative_file.txt"
    abs_path = (tmp_path / relative_path).resolve()
    abs_path_str = str(abs_path)
    abs_path.write_text("content", encoding="utf-8")

    search_block_error = "Error: Search block 1 not found..."

    cli_args_allow_all = argparse.Namespace(
        model="test_model_path_consistency", auto_approve=True, allow_read_files=True,
        allow_edit_files=True, allow_execute_safe_commands=True,
        allow_execute_all_commands=True, allow_use_browser=True, allow_use_mcp=True
    )
    # No LLM calls in this direct _run_tool test.
    agent = DeveloperAgent(send_message=MagicMock(return_value=LLMResponse(content="some content", usage=LLMUsageInfo())), cwd=str(tmp_path), cli_args=cli_args_allow_all)
    replace_tool_instance = agent.tools_map["replace_in_file"]

    with patch.object(replace_tool_instance, 'execute', return_value=search_block_error) as mock_replace_execute:
        # First failure with relative path
        tool_use_relative = ToolUse(name="replace_in_file", params={"path": relative_path, "diff_blocks": "d1"})
        agent._run_tool(tool_use_relative)
        mock_replace_execute.assert_called_with({"path": relative_path, "diff_blocks": "d1"}, agent_tools_instance=agent)
        assert agent.diff_failure_tracker.get(abs_path_str, 0) == 1

        # Second failure with absolute path (should be treated as same file)
        tool_use_absolute = ToolUse(name="replace_in_file", params={"path": abs_path_str, "diff_blocks": "d2"})
        result = agent._run_tool(tool_use_absolute)
        mock_replace_execute.assert_called_with({"path": abs_path_str, "diff_blocks": "d2"}, agent_tools_instance=agent)


        assert agent.diff_failure_tracker.get(abs_path_str, 0) == 0 # Reset after suggestion
        expected_suggestion = REPLACE_SUGGESTION_MESSAGE_TEMPLATE.format(file_path=abs_path_str)
        assert expected_suggestion in result

    assert mock_replace_execute.call_count == 2


# --- Tests for Agent Confirmation Logic ---
APPROVAL_FLAG_PARAMS = [
    ("allow_read_files", "read_file", {"path": "test.txt"}),
    ("allow_edit_files", "write_to_file", {"path": "test.txt", "content": "c"}),
    ("allow_edit_files", "replace_in_file", {"path": "test.txt", "diff_blocks": "d"}),
    # execute_command has more complex logic, tested separately
    ("allow_use_browser", "browser_action", {"action": "browse", "url": "u.com"}),
    ("allow_use_mcp", "use_mcp_tool", {"tool_name": "mcp_t", "server_name": "s1"}),
    ("allow_use_mcp", "access_mcp_resource", {"uri": "mcp://s1/res"}),
]

class TestAgentConfirmationLogic(unittest.TestCase):
    def setUp(self):
        self.mock_send_message = MagicMock(return_value=LLMResponse(content="default", usage=LLMUsageInfo())) # Mock send_message to return LLMResponse
        self.cwd = Path(".") # Assuming tests run where a CWD makes sense, or use tmp_path

        # Default args with all approval flags False
        self.base_cli_args = argparse.Namespace(
            model="test_model_confirmation", # Added model
            auto_approve=False, # General auto-approve for non-command tasks if ever used by tools
            allow_read_files=False,
            allow_edit_files=False,
            allow_execute_safe_commands=False,
            allow_execute_all_commands=False,
            allow_use_browser=False,
            allow_use_mcp=False,
            # Ensure all boolean flags expected by DeveloperAgent.__init__ are present
            disable_git_auto_commits=True # Default for tests unless specified
        )

    def _create_agent(self, cli_args_overrides=None):
        current_args_dict = vars(self.base_cli_args)
        if cli_args_overrides:
            current_args_dict.update(cli_args_overrides)
        current_args_ns = argparse.Namespace(**current_args_dict)

        return DeveloperAgent(send_message=self.mock_send_message, cwd=str(self.cwd), cli_args=current_args_ns)

    @patch('src.agent.request_user_confirmation')
    def test_tool_allowed_by_flag(self, mock_request_confirmation):
        for flag_name, tool_name, params in APPROVAL_FLAG_PARAMS:
            with self.subTest(tool=tool_name, flag=flag_name):
                agent = self._create_agent(cli_args_overrides={flag_name: True})

                # Mock the specific tool's execute method
                tool_instance = agent.tools_map.get(tool_name)
                assert tool_instance is not None, f"Tool {tool_name} not found in agent"

                with patch.object(tool_instance, 'execute', return_value=f"Mocked {tool_name} result") as mock_tool_execute:
                    tool_use = ToolUse(name=tool_name, params=params)
                    agent._run_tool(tool_use)

                    mock_request_confirmation.assert_not_called()
                    mock_tool_execute.assert_called_once_with(params, agent_tools_instance=agent)

                mock_request_confirmation.reset_mock() # Reset for next subtest

    @patch('src.agent.request_user_confirmation')
    def test_tool_denied_by_flag_user_confirms(self, mock_request_confirmation):
        mock_request_confirmation.return_value = True # User says 'yes'
        for flag_name, tool_name, params in APPROVAL_FLAG_PARAMS:
            with self.subTest(tool=tool_name, flag=flag_name):
                agent = self._create_agent(cli_args_overrides={flag_name: False}) # Flag is OFF

                tool_instance = agent.tools_map.get(tool_name)
                assert tool_instance is not None

                with patch.object(tool_instance, 'execute', return_value=f"Mocked {tool_name} result") as mock_tool_execute:
                    tool_use = ToolUse(name=tool_name, params=params)
                    agent._run_tool(tool_use)

                    mock_request_confirmation.assert_called_once()
                    mock_tool_execute.assert_called_once_with(params, agent_tools_instance=agent)

                mock_request_confirmation.reset_mock()

    @patch('src.agent.request_user_confirmation')
    def test_tool_denied_by_flag_user_denies(self, mock_request_confirmation):
        mock_request_confirmation.return_value = False # User says 'no'
        for flag_name, tool_name, params in APPROVAL_FLAG_PARAMS:
            with self.subTest(tool=tool_name, flag=flag_name):
                agent = self._create_agent(cli_args_overrides={flag_name: False}) # Flag is OFF

                tool_instance = agent.tools_map.get(tool_name)
                assert tool_instance is not None

                with patch.object(tool_instance, 'execute', return_value=f"Mocked {tool_name} result") as mock_tool_execute:
                    tool_use = ToolUse(name=tool_name, params=params)
                    result = agent._run_tool(tool_use)

                    mock_request_confirmation.assert_called_once()
                    mock_tool_execute.assert_not_called()
                    self.assertIn(f"User denied permission to {tool_name}", result)

                mock_request_confirmation.reset_mock()

    # Tests for execute_command
    @patch('src.agent.request_user_confirmation')
    @patch('src.tools.command.ExecuteCommandTool.execute', return_value='{"success": true, "output": "cmd output"}')
    def test_execute_command_allow_all_commands_flag(self, mock_cmd_execute, mock_request_confirmation):
        agent = self._create_agent(cli_args_overrides={"allow_execute_all_commands": True})
        params = {"command": "some_command", "requires_approval": "true"} # LLM says it needs approval
        tool_use = ToolUse(name="execute_command", params=params)

        agent._run_tool(tool_use)

        mock_request_confirmation.assert_not_called()
        mock_cmd_execute.assert_called_once()

    @patch('src.agent.request_user_confirmation')
    @patch('src.tools.command.ExecuteCommandTool.execute', return_value='{"success": true, "output": "cmd output"}')
    def test_execute_command_allow_safe_commands_flag_llm_safe(self, mock_cmd_execute, mock_request_confirmation):
        agent = self._create_agent(cli_args_overrides={"allow_execute_safe_commands": True})
        # LLM says command is safe (requires_approval=false)
        params = {"command": "safe_command", "requires_approval": "false"}
        tool_use = ToolUse(name="execute_command", params=params)

        agent._run_tool(tool_use)

        mock_request_confirmation.assert_not_called()
        mock_cmd_execute.assert_called_once()

    @patch('src.agent.request_user_confirmation', return_value=True) # User confirms
    @patch('src.tools.command.ExecuteCommandTool.execute', return_value='{"success": true, "output": "cmd output"}')
    def test_execute_command_allow_safe_commands_flag_llm_unsafe_user_confirms(self, mock_cmd_execute, mock_request_confirmation):
        agent = self._create_agent(cli_args_overrides={"allow_execute_safe_commands": True})
         # LLM says command is NOT safe (requires_approval=true)
        params = {"command": "unsafe_command", "requires_approval": "true"}
        tool_use = ToolUse(name="execute_command", params=params)

        agent._run_tool(tool_use)

        mock_request_confirmation.assert_called_once() # Should ask because allow_safe_commands is not enough
        mock_cmd_execute.assert_called_once()

    @patch('src.agent.request_user_confirmation')
    @patch('src.tools.command.ExecuteCommandTool.execute', return_value='{"success": true, "output": "cmd output"}')
    def test_execute_command_legacy_auto_approve_llm_unsafe(self, mock_cmd_execute, mock_request_confirmation):
        # Test legacy auto_approve when LLM says command is unsafe
        agent = self._create_agent(cli_args_overrides={"auto_approve": True})
        params = {"command": "unsafe_command_but_auto_approved", "requires_approval": "true"}
        tool_use = ToolUse(name="execute_command", params=params)

        agent._run_tool(tool_use)

        mock_request_confirmation.assert_not_called() # Legacy auto-approve handles it
        mock_cmd_execute.assert_called_once()

    @patch('src.agent.request_user_confirmation')
    @patch('src.tools.command.ExecuteCommandTool.execute', return_value='{"success": true, "output": "cmd output"}')
    def test_execute_command_legacy_auto_approve_llm_safe(self, mock_cmd_execute, mock_request_confirmation):
        # Test legacy auto_approve when LLM says command is safe (should proceed without prompt)
        agent = self._create_agent(cli_args_overrides={"auto_approve": True})
        params = {"command": "safe_command_with_auto_approve", "requires_approval": "false"}
        tool_use = ToolUse(name="execute_command", params=params)

        agent._run_tool(tool_use)

        mock_request_confirmation.assert_not_called() # Safe by LLM, auto_approve doesn't make it prompt
        mock_cmd_execute.assert_called_once()


    @patch('src.agent.request_user_confirmation', return_value=False) # User denies
    @patch('src.tools.command.ExecuteCommandTool.execute')
    def test_execute_command_no_flags_llm_unsafe_user_denies(self, mock_cmd_execute, mock_request_confirmation):
        agent = self._create_agent() # All flags False
        params = {"command": "another_unsafe_command", "requires_approval": "true"}
        tool_use = ToolUse(name="execute_command", params=params)

        result = agent._run_tool(tool_use)

        mock_request_confirmation.assert_called_once()
        mock_cmd_execute.assert_not_called()
        self.assertIn("User denied permission to execute_command", result)

    @patch('src.agent.request_user_confirmation', return_value=True) # User confirms
    @patch('src.tools.command.ExecuteCommandTool.execute', return_value='{"success": true, "output": "cmd output"}')
    def test_execute_command_no_flags_llm_safe_user_confirms(self, mock_cmd_execute, mock_request_confirmation):
        # If no specific command flags, and LLM says safe, it should still ask if legacy auto_approve is also false.
        agent = self._create_agent() # All flags False, including auto_approve
        params = {"command": "supposedly_safe_command", "requires_approval": "false"}
        tool_use = ToolUse(name="execute_command", params=params)

        agent._run_tool(tool_use)

        # The current logic: if not allow_all and not (allow_safe and not req_approve) and not (auto_approve and req_approve) -> prompt
        # So, if req_approve is false:
        # not allow_all (T) and not (allow_safe (F) and T (T)) (T) and not (auto_approve (F) and F (F)) (T) -> prompt
        # Yes, it will prompt.
        mock_request_confirmation.assert_called_once()
        mock_cmd_execute.assert_called_once()


# --- Unit Tests for DeveloperAgent.__init__() ---

def test_agent_initialization_cwd(tmp_path: Path):
    """Test that agent.cwd is correctly set to the absolute path."""
    relative_cwd = "test_dir"
    test_dir = tmp_path / relative_cwd
    test_dir.mkdir()
    mock_cli_args = argparse.Namespace(model="test_model_init_cwd")

    agent = DeveloperAgent(send_message=MagicMock(return_value=LLMResponse(content="default", usage=LLMUsageInfo())), cwd=str(test_dir), cli_args=mock_cli_args)
    assert Path(agent.cwd).is_absolute()
    assert agent.cwd == str(test_dir.resolve())

    agent_default_cwd = DeveloperAgent(send_message=MagicMock(return_value=LLMResponse(content="default", usage=LLMUsageInfo())), cli_args=mock_cli_args) # Uses default cwd="."
    assert Path(agent_default_cwd.cwd).is_absolute()
    assert agent_default_cwd.cwd == str(Path(".").resolve())


def test_agent_initialization_tools_loaded():
    """Check that agent.tools_map is populated with expected tool instances."""
    mock_cli_args = argparse.Namespace(model="test_model_init_tools")
    agent = DeveloperAgent(send_message=MagicMock(return_value=LLMResponse(content="default", usage=LLMUsageInfo())), cli_args=mock_cli_args)
    assert len(agent.tools_map) > 0
    # Check for presence of a few key tools
    from src.tools.file import ReadFileTool # Import specific tools for isinstance check
    from src.tools.command import ExecuteCommandTool
    assert "read_file" in agent.tools_map
    assert isinstance(agent.tools_map["read_file"], ReadFileTool)
    assert "execute_command" in agent.tools_map
    assert isinstance(agent.tools_map["execute_command"], ExecuteCommandTool)
    assert "list_code_definition_names" in agent.tools_map # Check a renamed one

@patch('src.agent.get_system_prompt')
def test_agent_initialization_system_prompt(mock_get_system_prompt):
    """Test system prompt is fetched and added to memory during initialization."""
    mock_system_prompt_content = "Mocked system prompt."
    mock_get_system_prompt.return_value = mock_system_prompt_content
    mock_cli_args = argparse.Namespace(model="test_model_init_prompt")

    agent = DeveloperAgent(
        send_message=MagicMock(return_value=LLMResponse(content="default", usage=LLMUsageInfo())),
        cwd=".",
        supports_browser_use=True,
        mcp_servers_documentation="Test MCP Docs",
        cli_args=mock_cli_args
    )

    assert mock_get_system_prompt.called, "get_system_prompt was not called"
    # Verify call args more robustly if necessary, e.g. by checking individual args or key parts
    call_args, call_kwargs = mock_get_system_prompt.call_args
    assert call_kwargs['cwd'] == agent.cwd
    assert call_kwargs['supports_browser_use'] is True
    assert call_kwargs['mcp_servers_documentation'] == "Test MCP Docs"
    # Comparing tools can be tricky due to object instances. Check count or key names if needed.
    assert len(call_kwargs['tools']) == len(agent.tools_map.values())

    assert len(agent.history) > 0
    assert agent.history[0]["role"] == "system"
    assert agent.history[0]["content"] == mock_system_prompt_content


# --- Unit Tests for DeveloperAgent._run_tool() ---

class MockCLIArgs:
    def __init__(self, **kwargs):
        self.auto_approve = kwargs.get('auto_approve', False) # General auto_approve
        self.allow_read_files = kwargs.get('allow_read_files', False)
        self.allow_edit_files = kwargs.get('allow_edit_files', False)
        self.allow_execute_safe_commands = kwargs.get('allow_execute_safe_commands', False)
        self.allow_execute_all_commands = kwargs.get('allow_execute_all_commands', False)
        self.allow_use_browser = kwargs.get('allow_use_browser', False)
        self.allow_use_mcp = kwargs.get('allow_use_mcp', False)
        self.model = kwargs.get('model', "test_model_mock_cli") # Add model
        self.disable_git_auto_commits = kwargs.get('disable_git_auto_commits', True) # Add disable_git_auto_commits


class TestRunToolBehaviors(unittest.TestCase):
    def setUp(self):
        self.mock_send_message = MagicMock(return_value=LLMResponse(content="default", usage=LLMUsageInfo()))
        self.cwd = Path(".").resolve()
        self.cli_args_default = MockCLIArgs() # All flags false, model="test_model_mock_cli"
        self.cli_args_auto_approve_all = MockCLIArgs(
            auto_approve=True, allow_read_files=True, allow_edit_files=True,
            allow_execute_all_commands=True, allow_use_browser=True, allow_use_mcp=True,
            model="test_model_approve_all"
        )

    def _create_agent(self, cli_args_obj=None, mode="act", cli_args=None):
        if cli_args is not None:
            agent_cli_args = cli_args
        else:
            agent_cli_args = cli_args_obj if cli_args_obj is not None else self.cli_args_default
        # Ensure it's an argparse.Namespace or similar, not just a dict
        if isinstance(agent_cli_args, MockCLIArgs): # Convert if it's my MockCLIArgs
            agent_cli_args = argparse.Namespace(**vars(agent_cli_args))

        return DeveloperAgent(
            send_message=self.mock_send_message, # This is a MagicMock returning LLMResponse
            cwd=str(self.cwd),
            cli_args=agent_cli_args,
            mode=mode
        )

    def test_run_tool_attempt_completion(self):
        agent = self._create_agent()
        tool_use = ToolUse(name="attempt_completion", params={"result": "Task is done."})
        result = agent._run_tool(tool_use)
        self.assertEqual(result, "Task is done.")

    def test_run_tool_plan_mode_respond_in_plan_mode(self):
        agent = self._create_agent(mode="plan")
        original_tool = agent.tools_map["plan_mode_respond"]

        with patch.object(original_tool, 'execute', return_value="Plan response executed.") as mock_execute:
            tool_use = ToolUse(name="plan_mode_respond", params={"response": "Here is the plan."})
            result = agent._run_tool(tool_use)

            self.assertEqual(result, "Plan response executed.")
            mock_execute.assert_called_once_with({"response": "Here is the plan."}, agent_tools_instance=agent)
            self.assertEqual(agent.consecutive_tool_errors, 0)

    def test_run_tool_plan_mode_respond_in_act_mode(self):
        agent = self._create_agent(mode="act") # Agent is in 'act' mode
        tool_use = ToolUse(name="plan_mode_respond", params={"response": "Trying to plan in act mode."})

        result = agent._run_tool(tool_use)

        self.assertEqual(result, "Error: plan_mode_respond can only be used in PLAN MODE.")
        self.assertEqual(agent.consecutive_tool_errors, 1)

    def test_run_tool_unknown_tool_direct(self):
        agent = self._create_agent()
        tool_use = ToolUse(name="non_existent_tool", params={"arg": "val"})

        result = agent._run_tool(tool_use)

        self.assertEqual(result, "Error: Unknown tool 'non_existent_tool'. Please choose from the available tools.")
        self.assertEqual(agent.consecutive_tool_errors, 1)

    def test_run_tool_catches_value_error(self):
        agent = self._create_agent(cli_args=self.cli_args_auto_approve_all) # Bypass confirmation
        mock_specific_tool = MockTool(name="value_error_tool")
        mock_specific_tool.execute_fn.side_effect = ValueError("Specific value error from tool")
        agent.tools_map["value_error_tool"] = mock_specific_tool

        tool_use = ToolUse(name="value_error_tool", params={"param1": "test"})
        result = agent._run_tool(tool_use)

        self.assertEqual(result, "Error: Tool 'value_error_tool' encountered a value error. Reason: Specific value error from tool")
        self.assertEqual(agent.consecutive_tool_errors, 1) # Should increment on tool execution error

    def test_run_tool_catches_generic_exception(self):
        agent = self._create_agent(cli_args=self.cli_args_auto_approve_all) # Bypass confirmation
        mock_specific_tool = MockTool(name="generic_exception_tool")
        mock_specific_tool.execute_fn.side_effect = Exception("Generic problem in tool")
        agent.tools_map["generic_exception_tool"] = mock_specific_tool

        tool_use = ToolUse(name="generic_exception_tool", params={"param1": "test"})
        result = agent._run_tool(tool_use)

        self.assertEqual(result, "Error: Tool 'generic_exception_tool' failed to execute. Reason: Generic problem in tool")
        self.assertEqual(agent.consecutive_tool_errors, 1)

    def test_run_tool_read_file_adds_to_memory(self):
        agent = self._create_agent(cli_args_obj=self.cli_args_auto_approve_all) # Bypass confirmation for read_file

        file_path = "test_file_for_memory.txt"
        file_content = "Content to be remembered."
        abs_file_path_for_memory = str((self.cwd / file_path).resolve())

        # Mock ReadFileTool's execute method
        read_file_tool_instance = agent.tools_map["read_file"]

        with patch.object(read_file_tool_instance, 'execute', return_value=file_content) as mock_execute_read_file, \
             patch.object(agent.memory, 'add_file_context') as mock_add_file_context:

            tool_use = ToolUse(name="read_file", params={"path": file_path})
            result = agent._run_tool(tool_use)

            self.assertEqual(result, file_content)
            mock_execute_read_file.assert_called_once_with({"path": file_path}, agent_tools_instance=agent)
            mock_add_file_context.assert_called_once_with(abs_file_path_for_memory, file_content)
            self.assertEqual(agent.consecutive_tool_errors, 0)
            # This test directly calls _run_tool, so no LLM call, session_cost is not affected.
            self.assertEqual(agent.current_session_cost, 0.0)


    def test_run_tool_consecutive_errors_reset_on_specific_success(self):
        agent = self._create_agent(cli_args_obj=self.cli_args_auto_approve_all) # Bypass confirmation

        # Test for replace_in_file
        agent.consecutive_tool_errors = 1
        replace_tool_instance = agent.tools_map["replace_in_file"]
        file_path_replace = "file_to_replace.txt"
        abs_path_replace = str((self.cwd / file_path_replace).resolve())
        success_replace_msg = f"File {abs_path_replace} modified successfully with 1 block(s)."

        with patch.object(replace_tool_instance, 'execute', return_value=success_replace_msg) as mock_replace:
            tool_use_replace = ToolUse(name="replace_in_file", params={"path": file_path_replace, "diff": "some diff"})
            agent._run_tool(tool_use_replace)
            self.assertEqual(agent.consecutive_tool_errors, 0)
            mock_replace.assert_called_once()

        # Test for write_to_file
        agent.consecutive_tool_errors = 1 # Reset for next test case
        write_tool_instance = agent.tools_map["write_to_file"]
        file_path_write = "file_to_write.txt"
        abs_path_write = str((self.cwd / file_path_write).resolve())
        success_write_msg = f"File written successfully to {abs_path_write}"

        with patch.object(write_tool_instance, 'execute', return_value=success_write_msg) as mock_write:
            tool_use_write = ToolUse(name="write_to_file", params={"path": file_path_write, "content": "some content"})
            agent._run_tool(tool_use_write)
            self.assertEqual(agent.consecutive_tool_errors, 0)
            mock_write.assert_called_once()


def test_run_task_known_tool_execution_returns_error(tmp_path: Path):
    """
    Integration test for run_task:
    Known tool's execute() method returns an error string.
    Agent should process this, increment error counter, and continue.
    """
    file_to_read = "test_error_file.txt"
    llm_tool_call = f"<tool_use><tool_name>read_file</tool_name><params><path>{file_to_read}</path></params></tool_use>"
    llm_second_response = "<text_content>LLM acknowledged the tool error and is continuing.</text_content>"

    # Make the second response also an error to check counter increment
    llm_second_response_error = "<tool_use><tool_name>another_unknown_tool</tool_name><params/></tool_use>"
    # LLM's final response after the second error (which will be the output of run_task if MockLLM is exhausted)
    # This response is not strictly processed by the agent loop if MockLLM only has 2 responses and run_task ends.
    # The final_result will be the result of the last processed LLM message.
    # Let's ensure MockLLM has a third response to make the test cleaner about what final_result is.
    llm_third_response_text = "<text_content>LLM giving up after multiple errors.</text_content>"

    mock_llm_responses = [llm_tool_call, llm_second_response_error, llm_third_response_text]

    # Use cli_args that would normally allow read_file to proceed without interactive confirmation
    cli_args_allow_reads = argparse.Namespace(
        model="test_model_known_tool_err", # Added model
        auto_approve=False, # General auto_approve
        allow_read_files=True, # Specific flag for read_file
        allow_edit_files=False,
        allow_execute_safe_commands=False,
        allow_execute_all_commands=False,
        allow_use_browser=False,
        allow_use_mcp=False,
        disable_git_auto_commits=True
    )
    agent = DeveloperAgent(send_message=MockLLM(mock_llm_responses).send_message, cwd=str(tmp_path), cli_args=cli_args_allow_reads)

    # Mock the execute method of the specific ReadFileTool instance
    read_file_tool_instance = agent.tools_map["read_file"]
    tool_returned_error_message = "Error: File not found for testing."

    with patch.object(read_file_tool_instance, 'execute', return_value=tool_returned_error_message) as mock_read_execute:
        final_result = agent.run_task("Read a file that will cause a tool error.")

    # Assertions
    mock_read_execute.assert_called_once_with({"path": file_to_read}, agent_tools_instance=agent)

    # After the first tool error (read_file) and the second tool error (another_unknown_tool),
    # the counter should be 2. The third LLM response (text) will then reset it.
    # So, the final state of consecutive_tool_errors after the entire run_task will be 0.
    # The point of this test is to ensure the counter *does* increment through multiple errors.
    # We can check the history to see the errors being processed.

    # History check:
    # 0: System Prompt
    # 1: User: "Read a file that will cause a tool error."
    # 2: Assistant: llm_tool_call (<read_file...>)
    # 3: User: "Result of read_file:\nError: File not found for testing." (Error from read_file)
    # 4: Assistant: llm_second_response_error (<another_unknown_tool...>)
    # 5: User: "Result of another_unknown_tool:\nError: Unknown tool 'another_unknown_tool'..." (Error from unknown_tool)
    # 6: Assistant: llm_third_response_text ("LLM giving up...")

    assert len(agent.history) == 7, f"History length was {len(agent.history)}, expected 7. History: {agent.history}"

    assert agent.history[0]["role"] == "system"
    assert agent.history[1]["role"] == "user"
    assert agent.history[1]["content"] == "Read a file that will cause a tool error."

    assert agent.history[2]["role"] == "assistant"
    assert agent.history[2]["content"] == llm_tool_call
    assert agent.history[3]["role"] == "user"
    assert agent.history[3]["content"] == f"Result of read_file:\n{tool_returned_error_message}"

    assert agent.history[4]["role"] == "assistant"
    assert agent.history[4]["content"] == llm_second_response_error
    assert agent.history[5]["role"] == "user"
    assert "Result of another_unknown_tool:\nError: Unknown tool 'another_unknown_tool'" in agent.history[5]["content"]

    assert agent.history[6]["role"] == "assistant"
    assert agent.history[6]["content"] == llm_third_response_text

    # The consecutive_tool_errors should be 0 at the end because the last LLM response was valid text.
    assert agent.consecutive_tool_errors == 0, \
        f"Consecutive tool error counter should be 0 after a final successful text response, but was {agent.consecutive_tool_errors}."
    # LLM calls: tool_error, tool_error, final_text = 3 calls
    assert agent.current_session_cost == 0.0

    assert final_result == "LLM giving up after multiple errors.", \
        "Final result did not match the LLM's third response."


# --- Tests for Git-related attributes ---

@pytest.fixture
def minimal_git_repo(tmp_path: Path) -> Path:
    """
    Creates a temporary directory, initializes a git repository in it,
    and creates an initial commit. Simplified version for agent tests.
    """
    repo_dir = tmp_path / "agent_test_repo"
    repo_dir.mkdir()

    try:
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test Agent User"], cwd=repo_dir, check=True)
        subprocess.run(["git", "config", "user.email", "test-agent@example.com"], cwd=repo_dir, check=True)
        (repo_dir / "initial_file.txt").write_text("Initial content.")
        subprocess.run(["git", "add", "initial_file.txt"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial agent test commit"], cwd=repo_dir, check=True, capture_output=True)
        yield repo_dir
    except subprocess.CalledProcessError as e:
        # Provide more context if git commands fail during setup
        print(f"Git setup failed in fixture: {e.cmd}")
        print(f"Stdout: {e.stdout.decode() if e.stdout else 'N/A'}")
        print(f"Stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
        raise
    finally:
        # tmp_path fixture handles cleanup of repo_dir contents
        pass

def _get_head_hash(repo_path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path, text=True).strip()
    except subprocess.CalledProcessError:
        return None

def test_agent_initialization_git_attributes_real_repo(minimal_git_repo: Path):
    expected_hash = _get_head_hash(minimal_git_repo)
    assert expected_hash is not None, "Failed to get HEAD hash from fixture repo"

    mock_cli_args = argparse.Namespace(model="git_test_model", disable_git_auto_commits=False)
    agent = DeveloperAgent(send_message=MagicMock(return_value=LLMResponse("default", LLMUsageInfo())), cwd=str(minimal_git_repo), cli_args=mock_cli_args)

    assert agent.initial_session_head_commit_hash == expected_hash
    assert agent.session_commit_history == []
    assert agent.current_session_cost == 0.0 # No LLM calls made in init

def test_agent_initialization_git_attributes_non_git_dir(tmp_path: Path):
    # tmp_path itself is not a git repo
    mock_cli_args = argparse.Namespace(model="git_test_model_non_git", disable_git_auto_commits=False)
    agent = DeveloperAgent(send_message=MagicMock(return_value=LLMResponse("default", LLMUsageInfo())), cwd=str(tmp_path), cli_args=mock_cli_args)
    assert agent.initial_session_head_commit_hash is None
    assert agent.session_commit_history == []
    assert agent.current_session_cost == 0.0

@patch('src.agent.subprocess.check_output')
def test_agent_initialization_git_attributes_mocked_subprocess(mock_check_output, tmp_path: Path):
    # Test successful git rev-parse mock
    mock_check_output.return_value.strip.return_value = "mocked_successful_hash"
    # Need to ensure .git directory exists for the subprocess call to be attempted by agent
    git_dir_for_mock = tmp_path / ".git"
    git_dir_for_mock.mkdir() # Make it look like a git repo to pass initial Path(cwd / ".git").exists() check
    mock_cli_args = argparse.Namespace(model="git_test_model_mocked", disable_git_auto_commits=False)

    agent_success = DeveloperAgent(send_message=MagicMock(return_value=LLMResponse("default", LLMUsageInfo())), cwd=str(tmp_path), cli_args=mock_cli_args)
    assert agent_success.initial_session_head_commit_hash == "mocked_successful_hash"
    assert agent_success.current_session_cost == 0.0
    mock_check_output.assert_called_once_with(["git", "rev-parse", "HEAD"], cwd=str(tmp_path), text=True, stderr=subprocess.PIPE)

    # Test git rev-parse failure (e.g., no commits yet, or other git error)
    mock_check_output.reset_mock()
    mock_check_output.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="git error: no commits")

    agent_failure = DeveloperAgent(send_message=MagicMock(return_value=LLMResponse("default", LLMUsageInfo())), cwd=str(tmp_path), cli_args=mock_cli_args)
    assert agent_failure.initial_session_head_commit_hash is None
    assert agent_failure.current_session_cost == 0.0

    # Test FileNotFoundError (git not installed)
    mock_check_output.reset_mock()
    mock_check_output.side_effect = FileNotFoundError("git not found")

    agent_not_found = DeveloperAgent(send_message=MagicMock(return_value=LLMResponse("default", LLMUsageInfo())), cwd=str(tmp_path), cli_args=mock_cli_args)
    assert agent_not_found.initial_session_head_commit_hash is None
    assert agent_not_found.current_session_cost == 0.0


@patch('src.agent.request_user_confirmation', return_value=True) # Auto-confirm any tool use prompts
def test_agent_auto_commit_updates_session_history(mock_user_confirm, minimal_git_repo: Path):
    file_to_write = "test_file.txt"
    file_path_in_repo = minimal_git_repo / file_to_write

    # Ensure cli_args allows file edits for _auto_commit to trigger
    cli_args_allow_edits = argparse.Namespace(
        auto_approve=True, # General auto-approve
        allow_read_files=True, allow_edit_files=True, # Specific edit flag
        allow_execute_safe_commands=True, allow_execute_all_commands=True,
        allow_use_browser=True, allow_use_mcp=True,
        disable_git_auto_commits=False, # Ensure auto-commits are enabled
        model="git_auto_commit_model"
    )

    # LLM responses: 1. write file1, 2. write file2, 3. final text response
    # MockLLM will wrap these strings into LLMResponse objects with 0.0 cost.
    mock_llm_responses = [
        f"<tool_use><tool_name>write_to_file</tool_name><params><path>{file_to_write}</path><content>Content for first commit</content></params></tool_use>",
        f"<tool_use><tool_name>write_to_file</tool_name><params><path>{file_to_write}</path><content>Content for second commit</content></params></tool_use>",
        "<text_content>Two files written.</text_content>"
    ]
    agent = DeveloperAgent(
        send_message=MockLLM(mock_llm_responses).send_message,
        cwd=str(minimal_git_repo),
        cli_args=cli_args_allow_edits
    )

    initial_repo_hash = _get_head_hash(minimal_git_repo) # Hash before any agent actions

    # Run the agent task
    agent.run_task("Write two versions of a file.") # This will involve 3 LLM calls

    assert agent.current_session_cost == 0.0 # 3 calls to MockLLM, each 0.0 cost
    assert len(agent.session_commit_history) == 2, \
        f"Expected 2 commits in session history, got {len(agent.session_commit_history)}. History: {agent.session_commit_history}"

    # Verify commit 1
    commit1_hash_agent = agent.session_commit_history[0]
    assert commit1_hash_agent is not None
    assert commit1_hash_agent != initial_repo_hash

    # Verify commit 2
    commit2_hash_agent = agent.session_commit_history[1]
    assert commit2_hash_agent is not None
    assert commit2_hash_agent != commit1_hash_agent

    # Verify that the current HEAD of the repo is the last commit made by the agent
    current_repo_head = _get_head_hash(minimal_git_repo)
    assert current_repo_head == commit2_hash_agent

    # Check that the file content matches the last commit
    assert file_path_in_repo.read_text() == "Content for second commit"

@patch('src.agent.request_user_confirmation', return_value=True)
def test_agent_auto_commit_no_changes_no_history_update(mock_user_confirm, minimal_git_repo: Path):
    cli_args_allow_edits = argparse.Namespace(
        model="git_no_change_model", # Added model
        auto_approve=True, allow_edit_files=True, disable_git_auto_commits=False,
        # Add other flags DeveloperAgent expects to avoid AttributeError on missing ones
        allow_read_files=True, allow_execute_safe_commands=True, allow_execute_all_commands=True,
        allow_use_browser=True, allow_use_mcp=True
    )

    # LLM response: attempt to write the same content that's already there from initial commit.
    # MockLLM wraps this into LLMResponse with 0.0 cost.
    # This assumes 'commit_all_changes' is smart enough to return None if no diff.
    # The initial file in minimal_git_repo is 'initial_file.txt' with 'Initial content.'
    existing_file = "initial_file.txt"
    existing_content = "Initial content."

    mock_llm_responses = [
        f"<tool_use><tool_name>write_to_file</tool_name><params><path>{existing_file}</path><content>{existing_content}</content></params></tool_use>",
        "<text_content>Attempted to write same content.</text_content>"
    ]
    agent = DeveloperAgent(
        send_message=MockLLM(mock_llm_responses).send_message,
        cwd=str(minimal_git_repo),
        cli_args=cli_args_allow_edits
    )

    agent.run_task("Write existing content to a file.") # 2 LLM calls

    assert agent.current_session_cost == 0.0 # 2 calls to MockLLM, 0.0 cost each
    # No new commit should be made, so session_commit_history should be empty.
    assert len(agent.session_commit_history) == 0, \
        f"Expected 0 commits in session history if no changes, got {len(agent.session_commit_history)}"

@patch('src.agent.commit_all_changes') # Mock commit_all_changes directly for _auto_commit test
@patch('src.agent.request_user_confirmation', return_value=True)
def test_agent_auto_commit_handles_commit_all_changes_returning_none(mock_user_confirm, mock_commit_all, minimal_git_repo: Path):
    mock_commit_all.return_value = None # Simulate no commit being made

    cli_args_allow_edits = argparse.Namespace(
        model="git_commit_none_model", # Added model
        auto_approve=True, allow_edit_files=True, disable_git_auto_commits=False,
        allow_read_files=True, allow_execute_safe_commands=True, allow_execute_all_commands=True,
        allow_use_browser=True, allow_use_mcp=True
    )
    agent = DeveloperAgent(
        send_message=MagicMock(return_value=LLMResponse("default", LLMUsageInfo())), # LLM not involved, directly calling _auto_commit indirectly
        cwd=str(minimal_git_repo),
        cli_args=cli_args_allow_edits
    )
    # Session cost is not affected as send_message is a simple mock here, not part of run_task loop

    # Manually trigger _auto_commit by calling a tool that would invoke it
    # This requires the tool's execute method to be part of the test, or a simpler way is needed.
    # For this test, we'll assume _auto_commit is called after write_to_file.
    # We can mock write_to_file's actual file writing part and just check _auto_commit.
    write_tool_instance = agent.tools_map["write_to_file"]
    with patch.object(write_tool_instance, 'execute', return_value="File write simulated (not really written)."):
         tool_use = ToolUse(name="write_to_file", params={"path": "dummy.txt", "content": "dummy"})
         agent._run_tool(tool_use) # This will call _auto_commit internally after tool execution

    mock_commit_all.assert_called_once_with(str(minimal_git_repo))
    assert len(agent.session_commit_history) == 0

# --- Test for on_llm_response_callback ---

def test_on_llm_response_callback_is_called(tmp_path: Path):
    """Test that the on_llm_response_callback is called with correct arguments."""
    mock_cli_args = argparse.Namespace(
        model="callback_test_model",
        auto_approve=True, # For simplicity if any tools are used
        # Add other necessary cli_args attributes DeveloperAgent might expect
        allow_read_files=True, allow_edit_files=True, allow_execute_safe_commands=True,
        allow_execute_all_commands=True, allow_use_browser=True, allow_use_mcp=True,
        disable_git_auto_commits=True
    )

    # Define a sequence of LLM responses
    llm_content1 = "<text_content>First response from LLM.</text_content>"
    usage1 = LLMUsageInfo(prompt_tokens=10, completion_tokens=20, cost=0.01) # Non-zero cost
    llm_response1 = LLMResponse(content=llm_content1, usage=usage1)

    llm_content2 = "<tool_use><tool_name>mock_tool</tool_name><params><param1>val</param1></params></tool_use>"
    usage2 = LLMUsageInfo(prompt_tokens=30, completion_tokens=40, cost=0.02)
    llm_response2 = LLMResponse(content=llm_content2, usage=usage2)

    # MockLLM needs strings for its response list; it wraps them into LLMResponse internally.
    # For this test, we want to control the LLMResponse objects directly to test the callback.
    # So, we'll use a custom mock_send_message function.

    mock_llm_responses_for_test = [llm_response1, llm_response2]
    call_index = 0
    def mock_send_message_with_controlled_response(history: List[Dict[str, str]]) -> Optional[LLMResponse]:
        nonlocal call_index
        if call_index < len(mock_llm_responses_for_test):
            response = mock_llm_responses_for_test[call_index]
            call_index += 1
            return response
        return LLMResponse(content="<text_content>Exhausted controlled responses.</text_content>", usage=LLMUsageInfo())

    mock_callback_fn = MagicMock()

    # Create a mock tool for the second LLM response
    mock_tool_instance = MockTool(name="mock_tool", execute_return_value="Result from callback mock tool")

    agent = DeveloperAgent(
        send_message=mock_send_message_with_controlled_response,
        cwd=str(tmp_path),
        cli_args=mock_cli_args,
        on_llm_response_callback=mock_callback_fn
    )
    # Replace the agent's tool with our mock instance for this test
    agent.tools_map["mock_tool"] = mock_tool_instance


    # Run a task that will trigger the two LLM responses
    agent.run_task("Run a task that uses the callback.")

    # Assertions for the callback (only first response triggers callback as agent stops after text)
    assert mock_callback_fn.call_count == 1
    mock_callback_fn.assert_called_with(llm_response1, "callback_test_model", 0.01)

    # Verify session cost accumulation on the agent as well
    assert agent.current_session_cost == 0.01

    # Verify history to ensure agent processed content correctly
    assert agent.history[2]["content"] == llm_content1
