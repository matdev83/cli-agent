from __future__ import annotations

import subprocess
import json
import shlex  # Import shlex
from typing import Dict, Any, Optional, Tuple

from .tool_protocol import Tool
from src.utils import to_bool


class ExecuteCommandTool(Tool):
    """
    A tool to execute shell commands.
    Note: Commands are executed directly, not through a shell, unless the command itself is a shell.
    Shell features like pipes, redirection, variable expansion in the command string will not work.
    """

    DEFAULT_TIMEOUT = 60.0

    @property
    def name(self) -> str:
        return "execute_command"

    @property
    def description(self) -> str:
        return (
            "Executes a command and its arguments directly (no shell interpretation). "
            "Returns a JSON string with 'success' (boolean) and 'output' (string) of the command. "
            "Shell features like pipes, redirection, or variable expansion in the command string are not supported."
        )

    @property
    def parameters_schema(self) -> Dict[str, str]:
        return {
            "command": "The command and its arguments to execute (e.g., 'ls -l /tmp').",
            "requires_approval": "A boolean ('true' or 'false') indicating if explicit user approval is needed.",
            "timeout_seconds": f"Optional timeout in seconds for the command execution. Defaults to {self.DEFAULT_TIMEOUT}s.",
        }

    def execute(self, params: Dict[str, Any], agent_tools_instance: Any) -> str:
        """Executes the given command. Expects 'command' and optionally 'timeout_seconds' and 'requires_approval' in params."""
        command_str = params.get("command")
        timeout_val_str = params.get("timeout_seconds")
        requires_approval_val = params.get("requires_approval")

        if not command_str:
            return json.dumps(
                {"success": False, "output": "Error: Missing required parameter 'command'."}
            )

        if not isinstance(command_str, str):
            return json.dumps({"success": False, "output": "Error: Command must be a string."})

        timeout: float = self.DEFAULT_TIMEOUT
        if timeout_val_str is not None:
            try:
                requested_timeout = float(timeout_val_str)
                if requested_timeout > 0:
                    timeout = requested_timeout
                else:
                    return json.dumps(
                        {
                            "success": False,
                            "output": "Error: 'timeout_seconds' must be a positive number.",
                        }
                    )
            except ValueError:
                return json.dumps(
                    {
                        "success": False,
                        "output": f"Error: Invalid value for 'timeout_seconds', must be a number: '{timeout_val_str}'.",
                    }
                )
        try:
            requires_approval_bool = to_bool(requires_approval_val)
        except ValueError as e:
            return json.dumps(
                {
                    "success": False,
                    "output": f"Error: Invalid boolean value for 'requires_approval': '{requires_approval_val}'. Expected 'true' or 'false'. Details: {e}",
                }
            )
        except TypeError as e:  # Should not happen if LLM sends string or param is omitted
            return json.dumps(
                {
                    "success": False,
                    "output": f"Error: Invalid type for 'requires_approval': '{requires_approval_val}'. Expected string, boolean or None. Details: {e}",
                }
            )

        auto_approved = False
        if (
            agent_tools_instance
            and hasattr(agent_tools_instance, "cli_args")
            and hasattr(agent_tools_instance.cli_args, "auto_approve")
        ):
            auto_approved = agent_tools_instance.cli_args.auto_approve
        elif agent_tools_instance and hasattr(
            agent_tools_instance, "auto_approve"
        ):  # Fallback for direct attribute
            auto_approved = agent_tools_instance.auto_approve

        if requires_approval_bool and not auto_approved:
            return json.dumps(
                {
                    "success": False,
                    "output": f"Error: Command '{command_str}' requires approval, and auto-approve is not enabled. User interaction is needed.",
                }
            )

        try:
            cmd_args = shlex.split(command_str)
            if not cmd_args:  # shlex.split on empty or whitespace-only string returns empty list
                return json.dumps(
                    {"success": False, "output": "Error: Command string is empty or whitespace."}
                )

            completed_process = subprocess.run(
                cmd_args,
                shell=False,  # Set to False
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,  # We check returncode manually
            )
            output_msg = (completed_process.stdout or "") + (completed_process.stderr or "")
            success_flag = completed_process.returncode == 0
            return json.dumps({"success": success_flag, "output": output_msg.strip()})
        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "success": False,
                    "output": f"Error: Command '{command_str}' timed out after {timeout} seconds.",
                }
            )
        except FileNotFoundError:  # Specific error for command not found
            # cmd_args[0] would be the command if cmd_args is not empty
            cmd_name_for_error = cmd_args[0] if cmd_args else command_str
            return json.dumps(
                {
                    "success": False,
                    "output": f"Error: Command not found: {cmd_name_for_error}",
                }
            )
        except Exception as exc:  # Catch-all for other execution errors
            return json.dumps(
                {"success": False, "output": f"Error executing command '{command_str}': {exc}"}
            )


# --- Wrapper function for old tests ---
def execute_command(
    command: str,
    requires_approval: bool = False,
    timeout: Optional[float] = None,
    agent_tools_instance: Any = None,
) -> Tuple[bool, str]:
    tool = ExecuteCommandTool()
    params = {"command": command, "requires_approval": requires_approval}
    if timeout is not None:
        params["timeout_seconds"] = timeout  # Pass timeout as float/int directly

    result_str = tool.execute(params, agent_tools_instance=agent_tools_instance)
    data = json.loads(result_str)

    # Test 'test_execute_command_rejected' is problematic.
    # If approval was required and not given by auto_approve, the tool returns an error message.
    # The test mocks `builtins.input` and expects "rejected" in the output.
    # This suggests the old `execute_command` might have had interactive approval.
    # The current tool does not. If the tool's output for "approval needed" is passed,
    # the test will likely fail as it won't contain "rejected".
    # For "test_execute_command_error" on "nonexistentcommand_xyz", the tool's FileNotFoundError handling is good.
    # For "test_execute_command_timeout", the tool's TimeoutExpired handling is good.

    return data.get("success", False), data.get("output", "")
