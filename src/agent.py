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
        """Convert assistant-provided boolean strings to bool."""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() == "true"

    def _run_tool(self, tool_use: ToolUse) -> str:
        """Dispatch a tool call using registered handler methods."""

        handlers: Dict[str, Callable[[dict], str]] = {
            "execute_command": self._handle_execute_command,
            "read_file": self._handle_read_file,
            "write_to_file": self._handle_write_to_file,
            "replace_in_file": self._handle_replace_in_file,
            "search_files": self._handle_search_files,
            "list_files": self._handle_list_files,
            "list_code_definition_names": self._handle_list_code_defs,
            "browser_action": self._handle_simple_tool,
            "use_mcp_tool": self._handle_simple_tool,
            "access_mcp_resource": self._handle_simple_tool,
        }

        handler = handlers.get(tool_use.name)
        if handler is None:
            raise ValueError(f"Unknown tool: {tool_use.name}")
        params = {k: v for k, v in tool_use.params.items() if v is not None}
        return handler({**params, "tool_name": tool_use.name})

    # individual tool handlers
    def _handle_execute_command(self, params: Dict[str, str]) -> str:
        cmd = params.get("command", "")
        req = self._to_bool(params.get("requires_approval"))
        success, output = self.tools["execute_command"](
            cmd, req, auto_approve=self.auto_approve
        )
        return output

    def _handle_read_file(self, params: Dict[str, str]) -> str:
        path = params.get("path", "")
        content = self.tools["read_file"](path)
        self.memory.add_file_context(os.path.join(self.cwd, path), content)
        return content

    def _handle_write_to_file(self, params: Dict[str, str]) -> str:
        self.tools["write_to_file"](params.get("path", ""), params.get("content", ""))
        return ""

    def _handle_replace_in_file(self, params: Dict[str, str]) -> str:
        self.tools["replace_in_file"](params.get("path", ""), params.get("diff", ""))
        return ""

    def _handle_list_files(self, params: Dict[str, str]) -> str:
        recursive = self._to_bool(params.get("recursive"))
        files = self.tools["list_files"](params.get("path", ""), recursive)
        return "\n".join(files)

    def _handle_search_files(self, params: Dict[str, str]) -> str:
        matches = self.tools["search_files"](
            params.get("path", ""),
            params.get("regex", ""),
            params.get("file_pattern"),
        )
        return "\n".join(f"{m['file']}:{m['line']}:{m['content']}" for m in matches)

    def _handle_list_code_defs(self, params: Dict[str, str]) -> str:
        return self.tools["list_code_definition_names"](params.get("path", ""))

    def _handle_simple_tool(self, params: Dict[str, str]) -> str:
        name = params.pop("tool_name")
        try:
            result = self.tools[name](**params)
        except NotImplementedError as exc:  # pragma: no cover - not implemented
            return str(exc)
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

