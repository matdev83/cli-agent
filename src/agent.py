from __future__ import annotations

import os
from typing import Callable, Dict, List

from .memory import Memory

from .assistant_message import parse_assistant_message, ToolUse, TextContent
from .prompts.system import get_system_prompt
from . import tools


class DeveloperAgent:
    """Simple developer agent coordinating tools and LLM messages."""

    def __init__(
        self,
        send_message: Callable[[List[Dict[str, str]]], str],
        *,
        cwd: str = ".",
        auto_approve: bool = False,
        supports_browser_use: bool = False,
    ) -> None:
        self.send_message = send_message
        self.cwd = cwd
        self.auto_approve = auto_approve
        self.supports_browser_use = supports_browser_use
        self.memory = Memory()
        self.history: List[Dict[str, str]] = self.memory.history
        self.tools: Dict[str, Callable[..., object]] = {
            "execute_command": tools.execute_command,
            "read_file": tools.read_file,
            "write_to_file": tools.write_to_file,
            "replace_in_file": tools.replace_in_file,
            "search_files": tools.search_files,
            "list_files": tools.list_files,
            "list_code_definition_names": tools.list_code_definition_names,
            "browser_action": tools.browser_action,
            "use_mcp_tool": tools.use_mcp_tool,
            "access_mcp_resource": tools.access_mcp_resource,
        }

        os.chdir(cwd)
        system_prompt = get_system_prompt(cwd, supports_browser_use=supports_browser_use)
        self.memory.add_message("system", system_prompt)

    # Utility to convert bool-like strings
    @staticmethod
    def _to_bool(value: str | bool | None) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return value.lower() in {"true", "1", "yes", "y"}

    def _run_tool(self, tool_use: ToolUse) -> str:
        name = tool_use.name
        func = self.tools.get(name)
        if func is None:
            raise ValueError(f"Unknown tool: {name}")
        params = {k: v for k, v in tool_use.params.items() if v is not None}
        if name == "execute_command":
            cmd = params.get("command", "")
            req = self._to_bool(params.get("requires_approval"))
            success, output = func(cmd, req, auto_approve=self.auto_approve)
            return output
        if name in {"browser_action", "use_mcp_tool", "access_mcp_resource"}:
            try:
                result = func(**params)
            except NotImplementedError as exc:  # pragma: no cover - not implemented
                return str(exc)
            return str(result)
        if name == "replace_in_file":
            func(params.get("path", ""), params.get("diff", ""))
            return ""
        if name == "write_to_file":
            func(params.get("path", ""), params.get("content", ""))
            return ""
        if name == "read_file":
            path = params.get("path", "")
            content = func(path)
            self.memory.add_file_context(os.path.join(self.cwd, path), content)
            return content
        if name == "list_files":
            recursive = self._to_bool(params.get("recursive"))
            return "\n".join(func(params.get("path", ""), recursive))
        if name == "search_files":
            matches = func(params.get("path", ""), params.get("regex", ""), params.get("file_pattern"))
            return "\n".join(f"{m['file']}:{m['line']}:{m['content']}" for m in matches)
        if name == "list_code_definition_names":
            return func(params.get("path", ""))
        result = func(**params)
        return str(result)

    def run_task(self, user_input: str, max_steps: int = 20) -> str:
        """Run the agent loop until attempt_completion or step limit reached."""

        self.memory.add_message("user", user_input)
        for _ in range(max_steps):
            assistant_reply = self.send_message(self.history)
            self.memory.add_message("assistant", assistant_reply)
            parsed = parse_assistant_message(assistant_reply)
            tool_uses = [p for p in parsed if isinstance(p, ToolUse)]
            if not tool_uses:
                # if no tool use, return assistant reply text
                text = "\n".join(
                    p.content for p in parsed if isinstance(p, TextContent)
                ).strip()
                return text
            tool = tool_uses[0]
            if tool.name == "attempt_completion":
                return tool.params.get("result", "")
            result_text = self._run_tool(tool)
            self.memory.add_message("user", f"Result of {tool.name}:\n{result_text}")
        raise RuntimeError("Max steps reached without completion")

