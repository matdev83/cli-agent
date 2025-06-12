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
    agent = DeveloperAgent(send_message=MockLLM(mock_llm_responses).send_message, cwd=str(tmp_path))

    agent.run_task("Attempt malformed tool use.")

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

    result = agent.run_task("Trigger malformed XML again to check history.") # This uses new history
    # The first run_task already returned. We need to check agent state after first run.
    # Re-initialize agent for a clean check of history after one run.
    agent = DeveloperAgent(send_message=MockLLM([malformed_response, "<text_content>OK</text_content>"]).send_message, cwd=str(tmp_path))
    final_result = agent.run_task("Attempt malformed tool use.")

    assert MALFORMED_TOOL_PREFIX + "Missing <tool_name>" in final_result
    assert agent.consecutive_tool_errors == 1
    # History: System, User, Assistant(malformed_response), User(final_result from previous turn) -> this is not how it works.
    # The history is built up turn by turn.
    # After assistant sends malformed_response:
    # History[2] (assistant) = malformed_response
    # parse_assistant_message finds the error. consecutive_tool_errors increments.
    # final_text_response contains the error message.
    # The run_task loop then calls send_message again.
    # The next LLM response is "<text_content>OK</text_content>"
    # This is a pure text response, and since no *new* malformed error, counter resets.

    # Let's test the state *during* the first run_task call.
    # We need to mock send_message to inspect intermediate states or check history.

    # Simpler: check the final history of a multi-step interaction
    agent = DeveloperAgent(send_message=MockLLM([
        malformed_response, # Error 1
        "<text_content>OK, I will try something else.</text_content>" # Successful text response
    ]).send_message, cwd=str(tmp_path))

    agent.run_task("Test sequence for malformed tool.")

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

def test_agent_error_counter_resets_on_valid_text_response(tmp_path: Path):
    """Test that consecutive_tool_errors resets on a valid text response after an error."""
    malformed_response = "<tool_use><params><path>file.txt</path></params></tool_use>" # Missing <tool_name>
    valid_text_response = "<text_content>This is a valid text response.</text_content>"

    # MockLLM will provide malformed, then valid text.
    mock_llm = MockLLM([malformed_response, valid_text_response])
    agent = DeveloperAgent(send_message=mock_llm.send_message, cwd=str(tmp_path))

    # First task call triggers the error
    error_message_returned = agent.run_task("Trigger malformed XML.")
    assert MALFORMED_TOOL_PREFIX + "Missing <tool_name>" in error_message_returned
    assert agent.consecutive_tool_errors == 1 # Error occurred and was counted

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
    # 1. Receive unknown_tool_response. _run_tool returns error. consecutive_tool_errors = 1.
    #    History gets "Result of unknown_tool_123:\nError: Unknown tool..."
    # 2. Loop, send history to LLM. LLM sends valid_text_response.
    #    parse_assistant_message -> no tool uses, no malformed error. consecutive_tool_errors = 0.
    #    run_task returns "All good now."

    final_output = agent.run_task("Test sequence: unknown tool then valid text.")

    assert final_output == "All good now."
    assert agent.consecutive_tool_errors == 0 # Should be reset
    assert "Error: Unknown tool 'unknown_tool_123'" in agent.history[3]["content"] # Error was processed


def test_agent_malformed_tool_xml_unrecognized_tag(tmp_path: Path):
    """Test agent handling of malformed XML (unrecognized tag)."""
    malformed_response = "<some_random_xml_tag><param>value</param></some_random_xml_tag>"
    agent = DeveloperAgent(send_message=MockLLM([malformed_response]).send_message, cwd=str(tmp_path))

    returned_message = agent.run_task("Attempt unrecognized tag.")

    assert agent.consecutive_tool_errors == 1
    assert agent.history[2]["role"] == "assistant"
    assert agent.history[2]["content"] == malformed_response
    # When malformed response leads to no tool_uses, run_task returns the parsed error string
    assert (MALFORMED_TOOL_PREFIX + "Unrecognized or malformed tag") in returned_message

def test_agent_unknown_tool_increments_error_counter(tmp_path: Path):
    """Test that calling an unknown tool increments consecutive_tool_errors."""
    unknown_tool_response = "<tool_use><tool_name>this_tool_does_not_exist</tool_name><params></params></tool_use>"
    agent = DeveloperAgent(send_message=MockLLM([unknown_tool_response]).send_message, cwd=str(tmp_path))

    agent.run_task("Call an unknown tool.")

    assert agent.consecutive_tool_errors == 1
    assert agent.history[2]["role"] == "assistant"
    assert agent.history[2]["content"] == unknown_tool_response
    assert "Error: Unknown tool 'this_tool_does_not_exist'" in agent.history[3]["content"]
    assert agent.history[3]["role"] == "user"


def test_agent_tool_missing_required_parameter(tmp_path: Path):
    """Test that tool error for missing params increments error counter."""
    # read_file requires 'path'
    missing_param_response = "<tool_use><tool_name>read_file</tool_name><params></params></tool_use>" # Missing 'path'
    agent = DeveloperAgent(send_message=MockLLM([missing_param_response]).send_message, cwd=str(tmp_path))

    agent.run_task("Call read_file without path.")

    assert agent.consecutive_tool_errors == 1
    assert agent.history[2]["role"] == "assistant"
    assert agent.history[2]["content"] == missing_param_response
    # Check that the user-facing message in history contains the specific error from the tool
    assert "Result of read_file:\nError: Missing required parameter 'path'." in agent.history[3]["content"]
    assert agent.history[3]["role"] == "user"


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
    agent = DeveloperAgent(send_message=mock_llm.send_message, cwd=str(tmp_path))

    final_result = agent.run_task("Trigger three consecutive errors.")

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


def test_agent_consecutive_error_counter_resets_on_successful_tool_call(tmp_path: Path):
    """Test that consecutive_tool_errors resets after a successful tool call."""
    responses = [
        "<tool_use><tool_name>unknown_tool_1</tool_name><params/></tool_use>", # Error 1
        "<tool_use><tool_name>read_file</tool_name><params><path>file.txt</path></params></tool_use>", # Success
        "<text_content>All done.</text_content>" # Final response
    ]
    mock_llm = MockLLM(responses)
    agent = DeveloperAgent(send_message=mock_llm.send_message, cwd=str(tmp_path))

    # Create dummy file for read_file to succeed
    (tmp_path / "file.txt").write_text("dummy content", encoding="utf-8")

    agent.run_task("Error then success.")

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

def test_diff_failure_escalation_suggests_write_to_file(tmp_path: Path):
    """Test that replace_in_file failure escalation message is added after max failures."""
    file_to_edit = "test_file.txt"
    abs_file_path_str = str((tmp_path / file_to_edit).resolve())
    (tmp_path / file_to_edit).write_text("initial content", encoding="utf-8")

    # Specific error messages that should trigger the escalation
    search_block_error = "Error: Search block 1 (starting with 'old_text') not found..."
    diff_format_error = "Error processing diff_blocks for test_file.txt: Some diff error"

    responses = [
        f"<tool_use><tool_name>replace_in_file</tool_name><params><path>{file_to_edit}</path><diff_blocks>...</diff_blocks></params></tool_use>", # Fail 1
        f"<tool_use><tool_name>replace_in_file</tool_name><params><path>{file_to_edit}</path><diff_blocks>...</diff_blocks></params></tool_use>", # Fail 2 (escalation)
        "<text_content>Ok, will try write_to_file.</text_content>" # LLM acknowledges
    ]
    mock_llm = MockLLM(responses)
    agent = DeveloperAgent(send_message=mock_llm.send_message, cwd=str(tmp_path))

    # Mock the execute method of the actual ReplaceInFileTool instance
    replace_tool_instance = agent.tools_map["replace_in_file"]

    # Simulate failures
    with patch.object(replace_tool_instance, 'execute', side_effect=[
        search_block_error, # First failure
        diff_format_error   # Second failure, should trigger suggestion
    ]) as mock_replace_execute:
        final_output = agent.run_task(f"Edit {file_to_edit} twice with failing diffs.")

    assert final_output == "Ok, will try write_to_file."
    assert agent.diff_failure_tracker.get(abs_file_path_str, 0) == 0 # Reset after suggestion

    # Check history for the augmented error message after the second failure
    # History: Sys(prompt), User(task), Asst(fail1), User(res1), Asst(fail2), User(res2_sugg), Asst(text_ok)
    # Asst(text_ok) is the last message in history.
    assert len(agent.history) == 7

    # Result of first failed replace_in_file
    assert agent.history[3]["content"] == f"Result of replace_in_file:\n{search_block_error}"

    # Result of second failed replace_in_file, with suggestion
    expected_augmented_error = diff_format_error + REPLACE_SUGGESTION_MESSAGE_TEMPLATE.format(file_path=file_to_edit)
    assert agent.history[5]["content"] == f"Result of replace_in_file:\n{expected_augmented_error}"

    assert mock_replace_execute.call_count == 2


def test_diff_failure_tracker_resets_on_successful_replace(tmp_path: Path):
    """Test that diff_failure_tracker resets for a file after a successful replace_in_file."""
    file_to_edit = "another_file.txt"
    abs_file_path_str = str((tmp_path / file_to_edit).resolve())
    (tmp_path / file_to_edit).write_text("content", encoding="utf-8")

    # Simulate one failure first
    agent = DeveloperAgent(send_message=MagicMock(), cwd=str(tmp_path)) # Generic mock for send_message
    agent.diff_failure_tracker[abs_file_path_str] = 1

    # Now simulate a successful replace_in_file call for the same file
    # We can directly call _run_tool for this unit test
    success_response_from_tool = f"File {abs_file_path_str} modified successfully with 1 block(s)."
    replace_tool_instance = agent.tools_map["replace_in_file"]

    with patch.object(replace_tool_instance, 'execute', return_value=success_response_from_tool):
        tool_use = ToolUse(name="replace_in_file", params={"path": file_to_edit, "diff_blocks": "..."})
        agent._run_tool(tool_use)

    assert agent.diff_failure_tracker.get(abs_file_path_str, 0) == 0


def test_diff_failure_tracker_resets_on_successful_write(tmp_path: Path):
    """Test that diff_failure_tracker resets for a file after a successful write_to_file."""
    file_to_edit = "yet_another_file.txt"
    abs_file_path_str = str((tmp_path / file_to_edit).resolve())
    (tmp_path / file_to_edit).write_text("content", encoding="utf-8")

    agent = DeveloperAgent(send_message=MagicMock(), cwd=str(tmp_path))
    agent.diff_failure_tracker[abs_file_path_str] = MAX_DIFF_FAILURES_PER_FILE -1 # One less than max

    # Simulate a successful write_to_file
    write_tool_instance = agent.tools_map["write_to_file"]
    success_response_from_tool = f"File written successfully to {abs_file_path_str}"

    with patch.object(write_tool_instance, 'execute', return_value=success_response_from_tool):
        tool_use = ToolUse(name="write_to_file", params={"path": file_to_edit, "content": "new content"})
        agent._run_tool(tool_use)

    assert agent.diff_failure_tracker.get(abs_file_path_str, 0) == 0

def test_diff_failure_tracker_handles_relative_and_absolute_paths(tmp_path: Path):
    """Test that the tracker handles relative and absolute paths consistently."""
    relative_path = "relative_file.txt"
    abs_path = (tmp_path / relative_path).resolve()
    abs_path_str = str(abs_path)
    abs_path.write_text("content", encoding="utf-8")

    search_block_error = "Error: Search block 1 not found..."

    agent = DeveloperAgent(send_message=MagicMock(), cwd=str(tmp_path))
    replace_tool_instance = agent.tools_map["replace_in_file"]

    with patch.object(replace_tool_instance, 'execute', return_value=search_block_error) as mock_replace_exec:
        # First failure with relative path
        tool_use_relative = ToolUse(name="replace_in_file", params={"path": relative_path, "diff_blocks": "d1"})
        agent._run_tool(tool_use_relative)
        assert agent.diff_failure_tracker.get(abs_path_str, 0) == 1

        # Second failure with absolute path (should be treated as same file)
        tool_use_absolute = ToolUse(name="replace_in_file", params={"path": abs_path_str, "diff_blocks": "d2"})
        result = agent._run_tool(tool_use_absolute)

        assert agent.diff_failure_tracker.get(abs_path_str, 0) == 0 # Reset after suggestion
        expected_suggestion = REPLACE_SUGGESTION_MESSAGE_TEMPLATE.format(file_path=abs_path_str)
        assert expected_suggestion in result

    assert mock_replace_exec.call_count == 2
