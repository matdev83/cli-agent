import pytest
from src.tools.meta_tools import (
    NewTaskTool,
    CondenseTool,
    ReportBugTool,
    NewRuleTool,
    AskFollowupQuestionTool,
    AttemptCompletionTool,
    PlanModeRespondTool,
    LoadMcpDocumentationTool
)

# No MockAgentMemory needed as these stubs don't currently use agent_memory.

def test_new_task_tool_stub():
    tool = NewTaskTool()
    context_str = "Some context for the new task."
    params = {"context": context_str}
    result = tool.execute(params)
    assert "NewTaskTool called" in result
    assert "Full implementation of new task creation is pending." in result
    assert f"context: '{context_str}'" in result # Current stub format

def test_new_task_tool_stub_missing_context():
    tool = NewTaskTool()
    params = {} # Context is missing
    result = tool.execute(params)
    assert "Error: Missing required parameter 'context' for tool 'new_task'." in result

def test_new_task_tool_stub_empty_context():
    tool = NewTaskTool()
    params = {"context": ""} # Empty context
    result = tool.execute(params)
    assert "NewTaskTool called" in result
    assert "Full implementation of new task creation is pending." in result
    assert "context: ''" in result # As per stub's behavior with empty string

def test_condense_tool_stub():
    tool = CondenseTool()
    context_str = "Some context to condense."
    params = {"context": context_str}
    result = tool.execute(params)
    assert "CondenseTool called" in result
    assert "Full implementation of context condensation is pending." in result
    assert f"context: '{context_str}'" in result

def test_condense_tool_stub_missing_context():
    tool = CondenseTool()
    params = {}
    result = tool.execute(params)
    assert "Error: Missing required parameter 'context' for tool 'condense'." in result

def test_report_bug_tool_stub():
    tool = ReportBugTool()
    params = {
        "title": "Test Bug",
        "what_happened": "Something unexpected.",
        "steps_to_reproduce": "1. Do this. 2. Do that.",
        "api_request_output": "Some API output",
        "additional_context": "Extra info"
    }
    result = tool.execute(params)
    assert "ReportBugTool called" in result
    assert "Full implementation of bug reporting is pending." in result
    assert f"Title: '{params['title']}'" in result
    assert f"What Happened: '{params['what_happened']}'" in result
    assert f"Steps: '{params['steps_to_reproduce']}'" in result
    assert f"API Output: '{params['api_request_output']}'" in result
    assert f"Additional: '{params['additional_context']}'" in result

def test_report_bug_tool_stub_only_required_params():
    tool = ReportBugTool()
    params = {
        "title": "Minimal Bug",
        "what_happened": "It broke.",
        "steps_to_reproduce": "1. Click button."
    }
    # Optional params will be missing from the params dict if not provided
    expected_params_in_message = params.copy() # What the tool will receive

    result = tool.execute(params)
    assert "ReportBugTool called" in result
    assert "Full implementation of bug reporting is pending." in result
    assert f"Title: '{params['title']}'" in result
    assert f"What Happened: '{params['what_happened']}'" in result
    assert f"Steps: '{params['steps_to_reproduce']}'" in result
    assert "API Output: 'N/A'" in result
    assert "Additional: 'N/A'" in result


def test_new_rule_tool_stub():
    tool = NewRuleTool()
    path_str = ".clinerules/test-rule.md"
    content_str = "## Test Rule\n- Do this."
    params = {
        "path": path_str,
        "content": content_str
    }
    result = tool.execute(params)
    assert "NewRuleTool called" in result
    assert "Full implementation of rule file creation is pending." in result
    assert f"Path: '{path_str}'" in result
    content_preview = content_str[:50] + "..."
    assert f"Content (preview): '{content_preview}'" in result

def test_new_rule_tool_stub_missing_params():
    tool = NewRuleTool()
    # Test with path missing
    params_no_path = {"content": "Some content"}
    result_no_path = tool.execute(params_no_path)
    assert "Error: Missing required parameters for 'new_rule': path." in result_no_path

    # Test with content missing
    params_no_content = {"path": "some/path.md"}
    result_no_content = tool.execute(params_no_content)
    assert "Error: Missing required parameters for 'new_rule': content." in result_no_content

    # Test with both missing
    params_both_missing = {}
    result_both_missing = tool.execute(params_both_missing)
    # Order depends on dict iteration or how missing_required list is built
    assert "Error: Missing required parameters for 'new_rule': path, content." in result_both_missing or \
           "Error: Missing required parameters for 'new_rule': content, path." in result_both_missing


# --- AskFollowupQuestionTool Tests ---
def test_ask_followup_question_tool_instantiation():
    tool = AskFollowupQuestionTool()
    assert tool.name == "ask_followup_question"
    assert "Ask the user a question to gather additional information" in tool.description
    assert len(tool.parameters) == 2
    param_names = {p["name"] for p in tool.parameters}
    assert {"question", "options"} == param_names

def test_ask_followup_question_tool_execute_with_options():
    tool = AskFollowupQuestionTool()
    question_str = "What is your favorite color?"
    options_str = '["Red", "Green", "Blue"]'
    params = {"question": question_str, "options": options_str}
    result = tool.execute(params)
    assert "Success: AskFollowupQuestionTool called." in result
    assert f"Question: '{question_str}'" in result
    assert f"Options: '{options_str}'" in result
    assert "Full implementation pending." in result

def test_ask_followup_question_tool_execute_no_options():
    tool = AskFollowupQuestionTool()
    question_str = "How are you today?"
    params = {"question": question_str}
    result = tool.execute(params) # Options not provided
    assert "Success: AskFollowupQuestionTool called." in result
    assert f"Question: '{question_str}'" in result
    assert "Options: 'No options provided.'" in result # Default from stub
    assert "Full implementation pending." in result

def test_ask_followup_question_tool_execute_missing_question():
    tool = AskFollowupQuestionTool()
    options_str = '["Yes", "No"]'
    params = {"options": options_str} # Question is missing
    result = tool.execute(params)
    assert "Error: Missing required parameter 'question' for tool 'ask_followup_question'." in result

# --- AttemptCompletionTool Tests ---
def test_attempt_completion_tool_instantiation():
    tool = AttemptCompletionTool()
    assert tool.name == "attempt_completion"
    assert "present the result of your work to the user" in tool.description
    assert len(tool.parameters) == 2
    param_names = {p["name"] for p in tool.parameters}
    assert {"result", "command"} == param_names

def test_attempt_completion_tool_execute_with_command():
    tool = AttemptCompletionTool()
    result_str = "The task is complete."
    command_str = "python main.py --run"
    params = {"result": result_str, "command": command_str}
    result_msg = tool.execute(params)
    assert "Success: AttemptCompletionTool called." in result_msg
    assert f"Result: '{result_str}'" in result_msg
    assert f"Command: '{command_str}'" in result_msg
    assert "Full implementation pending." in result_msg

def test_attempt_completion_tool_execute_no_command():
    tool = AttemptCompletionTool()
    result_str = "Build successful."
    params = {"result": result_str} # Command not provided
    result_msg = tool.execute(params)
    assert "Success: AttemptCompletionTool called." in result_msg
    assert f"Result: '{result_str}'" in result_msg
    assert "Command: 'No command provided.'" in result_msg # Default from stub
    assert "Full implementation pending." in result_msg

def test_attempt_completion_tool_execute_missing_result():
    tool = AttemptCompletionTool()
    command_str = "git commit -am 'fix'"
    params = {"command": command_str} # Result is missing
    result_msg = tool.execute(params)
    assert "Error: Missing required parameter 'result' for tool 'attempt_completion'." in result_msg

# --- PlanModeRespondTool Tests ---
def test_plan_mode_respond_tool_instantiation():
    tool = PlanModeRespondTool()
    assert tool.name == "plan_mode_respond"
    assert "Respond to the user's inquiry in an effort to plan a solution" in tool.description
    assert len(tool.parameters) == 1
    assert tool.parameters[0]["name"] == "response"

def test_plan_mode_respond_tool_execute():
    tool = PlanModeRespondTool()
    response_str = "Here is my plan: 1. Do X. 2. Do Y."
    params = {"response": response_str}
    result_msg = tool.execute(params)
    assert "Success: PlanModeRespondTool called." in result_msg
    assert f"Response: '{response_str}'" in result_msg
    assert "Full implementation pending." in result_msg

def test_plan_mode_respond_tool_execute_missing_response():
    tool = PlanModeRespondTool()
    params = {} # Response is missing
    result_msg = tool.execute(params)
    assert "Error: Missing required parameter 'response' for tool 'plan_mode_respond'." in result_msg

# --- LoadMcpDocumentationTool Tests ---
def test_load_mcp_documentation_tool_instantiation():
    tool = LoadMcpDocumentationTool()
    assert tool.name == "load_mcp_documentation"
    assert "Load documentation about creating MCP servers" in tool.description
    assert len(tool.parameters) == 0 # No parameters

def test_load_mcp_documentation_tool_execute():
    tool = LoadMcpDocumentationTool()
    params = {} # No parameters needed
    result_msg = tool.execute(params)
    assert "Success: LoadMcpDocumentationTool called." in result_msg
    assert "Full implementation pending." in result_msg