import subprocess
import json
from unittest.mock import patch, MagicMock

from src.tools.command import ExecuteCommandTool
# Assuming src.utils.to_bool will be used by the tool, no direct import needed in test file itself.

# Dummy AgentToolsInstance class
class MockAgentToolsInstance:
    def __init__(self, cwd: str, auto_approve: bool = False): # auto_approve is used by the tool
        self.cwd = cwd
        # For ExecuteCommandTool, it checks agent_tools_instance.cli_args.auto_approve
        # or agent_tools_instance.auto_approve. We'll use the latter for this mock.
        self.auto_approve = auto_approve

def test_execute_command_tool_properties():
    tool = ExecuteCommandTool()
    assert tool.name == "execute_command"
    assert isinstance(tool.description, str)
    assert tool.parameters_schema == {
        "command": "The command and its arguments to execute (e.g., 'ls -l /tmp').",
        "requires_approval": "A boolean ('true' or 'false') indicating if explicit user approval is needed.",
        "timeout_seconds": f"Optional timeout in seconds for the command execution. Defaults to {ExecuteCommandTool.DEFAULT_TIMEOUT}s."
    }

@patch('subprocess.run')
def test_execute_command_success(mock_run, tmp_path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ExecuteCommandTool()

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "Success output"
    mock_process.stderr = ""
    mock_run.return_value = mock_process

    params = {"command": "echo hello"}
    result_str = tool.execute(params, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)

    mock_run.assert_called_once_with(
        ["echo", "hello"], shell=False, text=True, capture_output=True, timeout=ExecuteCommandTool.DEFAULT_TIMEOUT, check=False
    )
    assert result == {"success": True, "output": "Success output"}

@patch('subprocess.run')
def test_execute_command_failure(mock_run, tmp_path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ExecuteCommandTool()

    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.stdout = "Some output"
    mock_process.stderr = "Error output"
    mock_run.return_value = mock_process

    params = {"command": "exit 1"}
    result_str = tool.execute(params, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)

    expected_output = "Some outputError output"
    assert result == {"success": False, "output": expected_output}

@patch('subprocess.run')
def test_execute_command_with_specific_timeout(mock_run, tmp_path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ExecuteCommandTool()

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "Output within timeout"
    mock_process.stderr = ""
    mock_run.return_value = mock_process

    params = {"command": "echo test", "timeout_seconds": 10}
    result_str = tool.execute(params, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)

    mock_run.assert_called_once_with(
        ["echo", "test"], shell=False, text=True, capture_output=True, timeout=10.0, check=False
    )
    assert result == {"success": True, "output": "Output within timeout"}

@patch('subprocess.run')
def test_execute_command_timeout_expired(mock_run, tmp_path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ExecuteCommandTool()

    mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 5", timeout=1)

    params = {"command": "sleep 5", "timeout_seconds": 1}
    result_str = tool.execute(params, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)

    assert result == {"success": False, "output": "Error: Command 'sleep 5' timed out after 1.0 seconds."}

@patch('subprocess.run')
def test_execute_command_requires_approval_true_string_not_auto_approved(mock_run, tmp_path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path), auto_approve=False)
    tool = ExecuteCommandTool()

    params = {"command": "rm -rf /", "requires_approval": "true"}
    result_str = tool.execute(params, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)

    mock_run.assert_not_called()
    assert result == {
        "success": False,
        "output": "Error: Command 'rm -rf /' requires approval, and auto-approve is not enabled. User interaction is needed."
    }

@patch('subprocess.run')
def test_execute_command_requires_approval_true_bool_not_auto_approved(mock_run, tmp_path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path), auto_approve=False)
    tool = ExecuteCommandTool()

    params = {"command": "rm -rf /", "requires_approval": True} # Boolean True
    result_str = tool.execute(params, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)

    mock_run.assert_not_called()
    assert result == {
        "success": False,
        "output": "Error: Command 'rm -rf /' requires approval, and auto-approve is not enabled. User interaction is needed."
    }


@patch('subprocess.run')
def test_execute_command_requires_approval_auto_approved(mock_run, tmp_path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path), auto_approve=True)
    tool = ExecuteCommandTool()

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "Files deleted"
    mock_process.stderr = ""
    mock_run.return_value = mock_process

    params = {"command": "rm -rf /", "requires_approval": "true"}
    result_str = tool.execute(params, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)

    mock_run.assert_called_once()
    assert result == {"success": True, "output": "Files deleted"}

@patch('subprocess.run')
def test_execute_command_requires_approval_false_string_runs_normally(mock_run, tmp_path):
    """Test that string 'false' for requires_approval is correctly converted to False."""
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path), auto_approve=False)
    tool = ExecuteCommandTool()

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "LS output"
    mock_process.stderr = ""
    mock_run.return_value = mock_process

    params = {"command": "ls", "requires_approval": "false"} # String "false"
    result_str = tool.execute(params, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)

    mock_run.assert_called_once()
    assert result == {"success": True, "output": "LS output"}

@patch('subprocess.run')
def test_execute_command_requires_approval_false_bool_runs_normally(mock_run, tmp_path):
    """Test that boolean False for requires_approval runs normally."""
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path), auto_approve=False)
    tool = ExecuteCommandTool()

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "LS output"
    mock_process.stderr = ""
    mock_run.return_value = mock_process

    params = {"command": "ls", "requires_approval": False} # Boolean False
    result_str = tool.execute(params, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)

    mock_run.assert_called_once()
    assert result == {"success": True, "output": "LS output"}

@patch('subprocess.run')
def test_execute_command_requires_approval_omitted_runs_normally(mock_run, tmp_path):
    """Test that omitting requires_approval defaults to False and runs normally."""
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path), auto_approve=False)
    tool = ExecuteCommandTool()

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "LS output"
    mock_process.stderr = ""
    mock_run.return_value = mock_process

    params = {"command": "ls"} # requires_approval omitted
    result_str = tool.execute(params, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)

    mock_run.assert_called_once()
    assert result == {"success": True, "output": "LS output"}

def test_execute_command_missing_command_param(tmp_path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ExecuteCommandTool()
    params = {}
    result_str = tool.execute(params, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)
    assert result == {"success": False, "output": "Error: Missing required parameter 'command'."}

def test_execute_command_invalid_timeout_type(tmp_path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ExecuteCommandTool()
    params = {"command": "echo test", "timeout_seconds": "not-an-int"}
    result_str = tool.execute(params, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)
    assert result == {"success": False, "output": "Error: Invalid value for 'timeout_seconds', must be a number: 'not-an-int'."}

def test_execute_command_negative_timeout_value(tmp_path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ExecuteCommandTool()
    params = {"command": "echo test", "timeout_seconds": -5}
    result_str = tool.execute(params, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)
    assert result == {"success": False, "output": "Error: 'timeout_seconds' must be a positive number."}

@patch('subprocess.run')
def test_execute_command_filenotfound_error(mock_run, tmp_path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ExecuteCommandTool()

    mock_run.side_effect = FileNotFoundError("No such file or directory: non_existent_command")

    params = {"command": "non_existent_command"}
    result_str = tool.execute(params, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)

    assert result == {"success": False, "output": "Error: Command not found: non_existent_command"}

@patch('subprocess.run')
def test_execute_command_invalid_requires_approval_string(mock_run, tmp_path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ExecuteCommandTool()
    params = {"command": "echo test", "requires_approval": "blah"}
    result_str = tool.execute(params, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)

    mock_run.assert_not_called()
    assert result["success"] is False
    assert "Error: Invalid boolean value for 'requires_approval': 'blah'" in result["output"]
    assert "Expected 'true' or 'false'" in result["output"]
