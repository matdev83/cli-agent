import pytest
import json
from pathlib import Path

from src.tools.code import (
    ListCodeDefinitionNamesTool
)
# BrowserActionTool, UseMCPTool, AccessMCPResourceTool are now in their own test files.

# Dummy AgentToolsInstance class
class MockAgentToolsInstance:
    def __init__(self, cwd: str):
        self.cwd = str(Path(cwd).resolve()) # Ensure cwd is absolute for consistent pathing

# --- ListCodeDefinitionsTool Tests ---
def test_list_code_definitions_tool_properties():
    tool = ListCodeDefinitionNamesTool()
    assert tool.name == "list_code_definition_names"
    assert isinstance(tool.description, str)
    assert tool.parameters_schema == {
        "path": "The path of the directory to list top level source code definitions for."
    }
def test_list_code_definitions_success(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ListCodeDefinitionNamesTool()

    py_file_content = """
class MyClass:
    def method_one(self):
        pass

def my_function():
    pass

async def my_async_function():
    pass
"""
    (tmp_path / "example.py").write_text(py_file_content)
    (tmp_path / "another.txt").write_text("some text") # Non-python file

    result_str = tool.execute({"path": "."}, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)

    assert "results" in result
    assert len(result["results"]) == 1 # Only one .py file
    assert result["results"][0]["file"] == "example.py"
    # The tool prefixes definitions with "| "
    assert "| class MyClass:" in result["results"][0]["definitions"]
    assert "| def my_function():" in result["results"][0]["definitions"]
    assert "| async def my_async_function():" in result["results"][0]["definitions"]
    assert "Successfully listed code definitions." in result.get("message", "")

def test_list_code_definitions_js_file(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ListCodeDefinitionNamesTool()

    js_content = """
export function foo() {
  return 1;
}

class Bar {
  constructor() {}
}
"""
    (tmp_path / "example.js").write_text(js_content)

    result_str = tool.execute({"path": "."}, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)

    assert "results" in result
    assert len(result["results"]) == 1
    assert result["results"][0]["file"] == "example.js"
    defs = result["results"][0]["definitions"]
    assert any("function foo" in d for d in defs)
    assert any("class Bar" in d for d in defs)

def test_list_code_definitions_no_python_files(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ListCodeDefinitionNamesTool()
    (tmp_path / "another.txt").write_text("some text")
    (tmp_path / "README.md").write_text("# Readme")

    result_str = tool.execute({"path": "."}, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)

    # Check for the specific message
    # Case 1: No .py files found at all.
    # Case 2: .py files found, but they contain no definitions.
    message = result.get("message", "")
    assert message in (
        "No supported source files found in the directory.",
        "No definitions found in source files."
    )

    if message == "No supported source files found in the directory.":
        assert "results" not in result
    elif message == "No definitions found in source files.":
        assert result.get("results") == []


def test_list_code_definitions_empty_directory(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ListCodeDefinitionNamesTool()

    # Create an empty subdirectory to test
    empty_dir = tmp_path / "empty_subdir"
    empty_dir.mkdir()

    result_str = tool.execute({"path": str(empty_dir)}, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)
    # If the directory is empty, no .py files will be found.
    assert result.get("message", "") == "No supported source files found in the directory."
    assert "results" not in result # Or assert result.get("results") is None


def test_list_code_definitions_python_file_with_syntax_error(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ListCodeDefinitionNamesTool()
    (tmp_path / "broken.py").write_text("def func_one(:\n pass") # Syntax error

    result_str = tool.execute({"path": "."}, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)

    assert "results" in result
    assert len(result["results"]) == 1
    assert result["results"][0]["file"] == "broken.py"
    assert len(result["results"][0]["definitions"]) > 0 # Should contain the error message
    assert any("Error: Syntax error" in d for d in result["results"][0]["definitions"])
    assert "Found source files, but encountered errors while parsing definitions." in result.get("message", "")

def test_list_code_definitions_non_existent_directory(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ListCodeDefinitionNamesTool()
    result_str = tool.execute({"path": "non_existent_dir"}, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)
    assert "error" in result
    assert "Directory not found" in result["error"]

def test_list_code_definitions_path_is_file(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ListCodeDefinitionNamesTool()
    file_path = tmp_path / "a_file.py"
    file_path.write_text("def x(): pass")
    result_str = tool.execute({"path": str(file_path)}, agent_tools_instance=mock_agent_tools_instance) # Pass file path
    result = json.loads(result_str)
    assert "error" in result
    assert "is not a directory" in result["error"]

def test_list_code_definitions_no_directory_path_param(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ListCodeDefinitionNamesTool()
    result_str = tool.execute({}, agent_tools_instance=mock_agent_tools_instance) # Empty params
    result = json.loads(result_str)
    assert "error" in result
    assert "Missing required parameter 'path'" in result["error"]


# Stubbed tool tests for BrowserActionTool, UseMCPTool, and AccessMCPResourceTool
# have been removed as these tools are now implemented and tested in their respective
# test_browser_tools.py and test_mcp_tools.py files.
