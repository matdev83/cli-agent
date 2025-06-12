import pytest
import json
from pathlib import Path

from src.tools.code import (
    ListCodeDefinitionsTool,
    BrowserActionTool,
    UseMCPTool,
    AccessMCPResourceTool
)

# Dummy AgentMemory class
class MockAgentMemory:
    def __init__(self, cwd: str):
        self.cwd = str(Path(cwd).resolve()) # Ensure cwd is absolute for consistent pathing

# --- ListCodeDefinitionsTool Tests ---
def test_list_code_definitions_success(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ListCodeDefinitionsTool()

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

    result_str = tool.execute({"directory_path": "."}, agent_memory=mock_memory)
    result = json.loads(result_str)

    assert "results" in result
    assert len(result["results"]) == 1 # Only one .py file
    assert result["results"][0]["file"] == "example.py"
    # The tool prefixes definitions with "| "
    assert "| class MyClass:" in result["results"][0]["definitions"]
    assert "| def my_function():" in result["results"][0]["definitions"]
    assert "| async def my_async_function():" in result["results"][0]["definitions"]
    assert "Successfully listed code definitions." in result.get("message", "")

def test_list_code_definitions_no_python_files(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ListCodeDefinitionsTool()
    (tmp_path / "another.txt").write_text("some text")
    (tmp_path / "README.md").write_text("# Readme")

    result_str = tool.execute({"directory_path": "."}, agent_memory=mock_memory)
    result = json.loads(result_str)

    # Check for the specific message when no .py files with definitions are found or directory is empty of .py files
    assert "No Python files with definitions found or directory is empty." in result.get("message", "") or \
           "No definitions found in Python files." in result.get("message", "")
    assert "results" in result # results might be an empty list
    assert result.get("results") == []


def test_list_code_definitions_empty_directory(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ListCodeDefinitionsTool()

    # Create an empty subdirectory to test
    empty_dir = tmp_path / "empty_subdir"
    empty_dir.mkdir()

    result_str = tool.execute({"directory_path": str(empty_dir)}, agent_memory=mock_memory)
    result = json.loads(result_str)
    assert "No Python files with definitions found or directory is empty." in result.get("message", "")
    assert result.get("results") == []


def test_list_code_definitions_python_file_with_syntax_error(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ListCodeDefinitionsTool()
    (tmp_path / "broken.py").write_text("def func_one(:\n pass") # Syntax error

    result_str = tool.execute({"directory_path": "."}, agent_memory=mock_memory)
    result = json.loads(result_str)

    assert "results" in result
    assert len(result["results"]) == 1
    assert result["results"][0]["file"] == "broken.py"
    assert len(result["results"][0]["definitions"]) > 0 # Should contain the error message
    assert any("Error: Syntax error" in d for d in result["results"][0]["definitions"])
    assert "Found Python files, but encountered errors while parsing definitions." in result.get("message", "")

def test_list_code_definitions_non_existent_directory(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ListCodeDefinitionsTool()
    result_str = tool.execute({"directory_path": "non_existent_dir"}, agent_memory=mock_memory)
    result = json.loads(result_str)
    assert "error" in result
    assert "Directory not found" in result["error"]

def test_list_code_definitions_path_is_file(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ListCodeDefinitionsTool()
    file_path = tmp_path / "a_file.py"
    file_path.write_text("def x(): pass")
    result_str = tool.execute({"directory_path": str(file_path)}, agent_memory=mock_memory) # Pass file path
    result = json.loads(result_str)
    assert "error" in result
    assert "is not a directory" in result["error"]

def test_list_code_definitions_no_directory_path_param(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ListCodeDefinitionsTool()
    result_str = tool.execute({}, agent_memory=mock_memory) # Empty params
    result = json.loads(result_str)
    assert "error" in result
    assert "Missing required parameter 'directory_path'" in result["error"]


# --- Stubbed Tool Tests ---
def test_browser_action_tool_stub():
    tool = BrowserActionTool()
    with pytest.raises(NotImplementedError, match="browser_action is not implemented yet."):
        tool.execute({"action": "open", "action_params": {"url": "example.com"}})

def test_use_mcp_tool_stub():
    tool = UseMCPTool()
    with pytest.raises(NotImplementedError, match="use_mcp_tool is not implemented yet."):
        tool.execute({"tool_name": "some_mcp_tool", "tool_inputs": {}})

def test_access_mcp_resource_tool_stub():
    tool = AccessMCPResourceTool()
    with pytest.raises(NotImplementedError, match="access_mcp_resource is not implemented yet."):
        tool.execute({"resource_id": "some_resource", "access_params": {}})

```
