from __future__ import annotations

import subprocess
import json
from typing import Dict, Any, List, Optional

from .tool_protocol import Tool
from src.utils import to_bool # Import the new utility function

class ExecuteCommandTool(Tool):
    @property
    def name(self) -> str:
        return "execute_command"

    @property
    def description(self) -> str:
        return ("Executes a shell command and returns a JSON string with "
                "'success' (boolean) and 'output' (string) of the command.")

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {
                "name": "command",
                "description": "The shell command to execute.",
                "type": "string",
                "required": True
            },
            {
                "name": "timeout",
                "description": "Optional timeout in seconds for the command execution. Defaults to None (no timeout).",
                "type": "integer",
                "required": False
            },
            {
                "name": "requires_approval",
                "description": "If true (string 'true' or boolean True), command execution may require approval based on agent's settings. Defaults to False if not provided.",
                "type": "boolean", # LLM sees this as boolean, sends "true"/"false"
                "required": False
            }
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        command = params.get("command")
        timeout_str = params.get("timeout")
        requires_approval_val = params.get("requires_approval")

        if not command:
            return json.dumps({"success": False, "output": "Error: Missing required parameter 'command'."})

        timeout: Optional[int] = None
        if timeout_str is not None:
            try:
                timeout = int(timeout_str) # Convert timeout to int
                if timeout <= 0:
                    return json.dumps({"success": False, "output": "Error: 'timeout' must be a positive integer."})
            except ValueError:
                return json.dumps({"success": False, "output": f"Error: Invalid value for 'timeout', must be an integer: '{timeout_str}'."})

        try:
            # Convert requires_approval_val using the utility function
            # If requires_approval_val is None (param not provided), to_bool defaults it to False.
            # If it's a boolean, it's used as is.
            # If it's a string "true" or "false", it's converted.
            # If it's any other string, to_bool will raise ValueError.
            requires_approval_bool = to_bool(requires_approval_val)
        except ValueError as e:
            return json.dumps({
                "success": False,
                "output": f"Error: Invalid boolean value for 'requires_approval': '{requires_approval_val}'. Expected 'true' or 'false'. Details: {e}"
            })
        except TypeError as e: # Should not happen if LLM sends string or param is omitted
             return json.dumps({
                "success": False,
                "output": f"Error: Invalid type for 'requires_approval': '{requires_approval_val}'. Expected string, boolean or None. Details: {e}"
            })


        auto_approved = False
        if agent_memory and hasattr(agent_memory, 'auto_approve'):
            auto_approved = agent_memory.auto_approve

        if requires_approval_bool and not auto_approved:
            return json.dumps({
                "success": False,
                "output": f"Error: Command '{command}' requires approval, and auto-approve is not enabled. User interaction is needed."
            })

        try:
            completed_process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            output_msg = (completed_process.stdout or "") + (completed_process.stderr or "")
            success_flag = completed_process.returncode == 0
            return json.dumps({"success": success_flag, "output": output_msg.strip()})
        except subprocess.TimeoutExpired:
            return json.dumps({"success": False, "output": f"Error: Command '{command}' timed out after {timeout} seconds."})
        except FileNotFoundError:
             return json.dumps({"success": False, "output": f"Error: Command not found: {command.split()[0] if command else 'N/A'}"})
        except Exception as exc:
            return json.dumps({"success": False, "output": f"Error executing command '{command}': {exc}"})
