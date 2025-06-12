import pytest
from src.tools.meta_tools import (
    NewTaskTool,
    CondenseTool,
    ReportBugTool,
    NewRuleTool
)

# No MockAgentMemory needed as these stubs don't currently use agent_memory.

def test_new_task_tool_stub():
    tool = NewTaskTool()
    context_str = "Some context for the new task."
    params = {"context": context_str}
    result = tool.execute(params)
    assert "NewTaskTool called" in result
    assert "Full implementation pending" in result
    assert f"context: '{context_str}'" in result # Current stub format

def test_new_task_tool_stub_missing_context():
    tool = NewTaskTool()
    params = {} # Context is missing
    result = tool.execute(params)
    assert "NewTaskTool called" in result
    assert "Full implementation pending" in result
    assert "context: 'No context provided.'" in result # As per stub's default

def test_new_task_tool_stub_empty_context():
    tool = NewTaskTool()
    params = {"context": ""} # Empty context
    result = tool.execute(params)
    assert "NewTaskTool called" in result
    assert "Full implementation pending" in result
    assert "context: ''" in result # As per stub's behavior with empty string

def test_condense_tool_stub():
    tool = CondenseTool()
    context_str = "Some context to condense."
    params = {"context": context_str}
    result = tool.execute(params)
    assert "CondenseTool called" in result
    assert "Full implementation pending" in result
    assert f"context: '{context_str}'" in result

def test_condense_tool_stub_missing_context():
    tool = CondenseTool()
    params = {}
    result = tool.execute(params)
    assert "CondenseTool called" in result
    assert "Full implementation pending" in result
    assert "context: 'No context provided.'" in result

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
    assert "Full implementation pending" in result
    assert f"Title: '{params['title']}'" in result
    assert f"What Happened: '{params['what_happened']}'" in result
    assert f"Received params: {str(params)}" in result # Stub includes all params

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
    assert "Full implementation pending" in result
    assert f"Title: '{params['title']}'" in result
    assert f"What Happened: '{params['what_happened']}'" in result
    assert f"Received params: {str(expected_params_in_message)}" in result


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
    assert "Full implementation pending" in result
    assert f"Path: '{path_str}'" in result
    content_preview = content_str[:50] + "..."
    assert f"Content (preview): '{content_preview}'" in result

def test_new_rule_tool_stub_missing_params():
    tool = NewRuleTool()
    # Test with path missing
    params_no_path = {"content": "Some content"}
    result_no_path = tool.execute(params_no_path)
    assert "Path: 'No path provided.'" in result_no_path
    assert "Content (preview): 'Some content...'" in result_no_path # Preview of first 50 chars

    # Test with content missing
    params_no_content = {"path": "some/path.md"}
    result_no_content = tool.execute(params_no_content)
    assert "Path: 'some/path.md'" in result_no_content
    assert "Content (preview): 'No content provided....'" in result_no_content # .get default + preview logic

    # Test with both missing
    params_both_missing = {}
    result_both_missing = tool.execute(params_both_missing)
    assert "Path: 'No path provided.'" in result_both_missing
    assert "Content (preview): 'No content provided....'" in result_both_missing

