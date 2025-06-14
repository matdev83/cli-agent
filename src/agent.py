from __future__ import annotations

import os
import traceback
import logging
from typing import Callable, Dict, List, Optional
import argparse
from pathlib import Path
import subprocess

from .memory import Memory
from .assistant_message import parse_assistant_message, ToolUse, TextContent
from .prompts.system import get_system_prompt
from .confirmations import request_user_confirmation
from src.utils import (
    to_bool,
    commit_all_changes,
)
from src.llm_protocol import LLMResponse, LLMUsageInfo
from src.mentions import extract_file_mentions

from src.tools import (
    Tool,
    ReadFileTool,
    WriteToFileTool,
    ReplaceInFileTool,
    ListFilesTool,
    SearchFilesTool,
    ExecuteCommandTool,
    ListCodeDefinitionNamesTool,
    BrowserActionTool,
    UseMCPTool,
    AccessMCPResourceTool,
    NewTaskTool,
    CondenseTool,
    ReportBugTool,
    NewRuleTool,
    AskFollowupQuestionTool,
    PlanModeRespondTool,
    LoadMcpDocumentationTool,
)

MALFORMED_TOOL_PREFIX = "Error: Malformed tool use - "
MAX_CONSECUTIVE_TOOL_ERRORS = 3
ADMONISHMENT_MESSAGE = (
    "System: You have made several consecutive errors in tool usage or "
    "formatting. Please carefully review the available tools, their required "
    "parameters, and the expected XML format. Ensure your next response is a "
    "valid tool call or a text response."
)

MAX_DIFF_FAILURES_PER_FILE = 2
REPLACE_SUGGESTION_MESSAGE_TEMPLATE = (
    "\nAdditionally, applying diffs to '{file_path}' has failed multiple "
    "times. Consider reading the file content and using 'write_to_file' "
    "with the full desired content instead."
)


class DeveloperAgent:
    """Simple developer agent coordinating tools and LLM messages."""

    def __init__(
        self,
        send_message: Callable[
            [List[Dict[str, str]]], Optional[LLMResponse]
        ],
        *,
        cwd: str = ".",
        cli_args: Optional[argparse.Namespace] = None,
        supports_browser_use: bool = False,
        browser_settings: dict | None = None,
        mcp_servers_documentation: str = "(No MCP servers currently connected)",
        matching_strictness: int = 100,
        mode: str = "act",
        disable_git_auto_commits: bool = False,
        on_llm_response_callback: Optional[
            Callable[[LLMResponse, str, float], None]
        ] = None,
    ) -> None:
        self.send_message = send_message
        self.cwd: str = os.path.abspath(cwd)
        self.cli_args = cli_args if cli_args is not None else argparse.Namespace()
        self.model_name: str = getattr(self.cli_args, "model", "unknown_model")
        self.on_llm_response_callback = on_llm_response_callback

        default_cli_attributes = {
            "auto_approve": False,
            "allow_read_files": False,
            "allow_edit_files": False,
            "allow_execute_safe_commands": False,
            "allow_execute_all_commands": False,
            "allow_use_browser": False,
            "allow_use_mcp": False,
            "disable_git_auto_commits": disable_git_auto_commits,
        }

        for attr, default_value in default_cli_attributes.items():
            if not hasattr(self.cli_args, attr):
                setattr(self.cli_args, attr, default_value)

        self.disable_git_auto_commits = self.cli_args.disable_git_auto_commits

        self.supports_browser_use: bool = supports_browser_use
        self.matching_strictness: int = matching_strictness
        self.mode = mode.lower()

        self.memory = Memory()
        self.history: List[Dict[str, str]] = self.memory.history
        self.consecutive_tool_errors: int = 0
        self.diff_failure_tracker: Dict[str, int] = {}
        self.session_commit_history: List[str] = []
        self.initial_session_head_commit_hash: Optional[str] = None
        self.current_session_cost: float = 0.0
        self.mentioned_file_contents: list[dict[str, str]] = []

        self._tool_permission_config: Dict[str, Dict[str, any]] = {
            "execute_command": {
                "cli_arg_flag": "allow_execute_all_commands",
                "cli_arg_safe_flag": "allow_execute_safe_commands",
                "prompt_verb": "executing command",
                "resource_param_key": "command",
                "requires_approval_param": "requires_approval",
            },
            "write_to_file": {
                "cli_arg_flag": "allow_edit_files",
                "prompt_verb": "editing file",
                "resource_param_key": "path",
            },
            "replace_in_file": {
                "cli_arg_flag": "allow_edit_files",
                "prompt_verb": "editing file",
                "resource_param_key": "path",
            },
            "new_rule": {
                "cli_arg_flag": "allow_edit_files",
                "prompt_verb": "creating/editing rule",
                "resource_param_key": "path",
            },
            "read_file": {
                "cli_arg_flag": "allow_read_files",
                "prompt_verb": "reading file",
                "resource_param_key": "path",
            },
            "list_files": {
                "cli_arg_flag": "allow_read_files",
                "prompt_verb": "listing directory",
                "resource_param_key": "path",
            },
            "search_files": {
                "cli_arg_flag": "allow_read_files",
                "prompt_verb": "searching files in",
                "resource_param_key": "path",
            },
            "list_code_definition_names": {
                "cli_arg_flag": "allow_read_files",
                "prompt_verb": "listing code definitions in",
                "resource_param_key": "path",
            },
            "browser_action": {
                "cli_arg_flag": "allow_use_browser",
                "prompt_verb": "using browser for",
                "resource_param_key": "url",
                "action_param_key": "action",
            },
            "use_mcp_tool": {
                "cli_arg_flag": "allow_use_mcp",
                "prompt_verb": "using MCP tool",
                "resource_param_key": "tool_name",
            },
            "access_mcp_resource": {
                "cli_arg_flag": "allow_use_mcp",
                "prompt_verb": "accessing MCP resource",
                "resource_param_key": "uri",
            },
            "new_task": {"proceed_by_default": True},
            "condense": {"proceed_by_default": True},
            "report_bug": {"proceed_by_default": True},
            "ask_followup_question": {"proceed_by_default": True},
            "plan_mode_respond": {"proceed_by_default": True},
            "load_mcp_documentation": {"proceed_by_default": True},
        }

        tool_instances: List[Tool] = [
            ReadFileTool(),
            WriteToFileTool(),
            ReplaceInFileTool(),
            ListFilesTool(),
            SearchFilesTool(),
            ExecuteCommandTool(),
            ListCodeDefinitionNamesTool(),
            BrowserActionTool(),
            UseMCPTool(),
            AccessMCPResourceTool(),
            NewTaskTool(),
            CondenseTool(),
            ReportBugTool(),
            NewRuleTool(),
            AskFollowupQuestionTool(),
            PlanModeRespondTool(),
            LoadMcpDocumentationTool(),
        ]
        self.tools_map: Dict[str, Tool] = {
            tool.name: tool for tool in tool_instances
        }

        os.chdir(self.cwd)
        try:
            git_path = Path(self.cwd) / ".git"
            if git_path.exists() and git_path.is_dir():
                self.initial_session_head_commit_hash = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.cwd, text=True, stderr=subprocess.PIPE
                ).strip()
                logging.info(
                    f"Initial session HEAD: {self.initial_session_head_commit_hash}"
                )
            else:
                logging.info(
                    "Not a git repository. Initial HEAD commit hash not set."
                )
        except subprocess.CalledProcessError as e:
            logging.warning(
                f"Failed to get initial HEAD commit in {self.cwd}: {e.stderr}"
            )
            self.initial_session_head_commit_hash = None
        except FileNotFoundError:
            logging.warning(
                "git not found. Failed to get initial HEAD commit hash."
            )
            self.initial_session_head_commit_hash = None
        except Exception as e:
            logging.error(
                f"Unexpected error getting initial HEAD commit: {e}"
            )
            self.initial_session_head_commit_hash = None

        system_prompt_str = get_system_prompt(
            tools=list(self.tools_map.values()),
            cwd=self.cwd,
            supports_browser_use=self.supports_browser_use,
            browser_settings=browser_settings,
            mcp_servers_documentation=mcp_servers_documentation,
        )
        self.memory.add_message("system", system_prompt_str)

    def set_mode(self, mode: str) -> None:
        """Set the agent's operational mode."""
        if mode.lower() not in {"act", "plan"}:
            raise ValueError("mode must be 'act' or 'plan'")
        self.mode = mode.lower()

    def get_mode(self) -> str:
        """Return the current operational mode."""
        return self.mode

    def _auto_commit(self) -> None:
        if self.disable_git_auto_commits:
            return
        commit_hash = commit_all_changes(self.cwd)
        if commit_hash:
            self.session_commit_history.append(commit_hash)
            log_msg = (
                f"Auto-commit successful. New commit: {commit_hash}. "
                f"History length: {len(self.session_commit_history)}"
            )
            logging.info(log_msg)
        else:
            logging.info(
                "Auto-commit: No changes to commit or an error occurred."
            )

    def _check_tool_permission(
        self, tool_name: str, tool_params: Dict[str, any]
    ) -> tuple[bool, Optional[str]]:
        """Checks if the agent has permission to run the tool."""
        config = self._tool_permission_config.get(tool_name)

        if not config:
            logging.warning(
                f"Tool '{tool_name}' not in permission config. "
                "Allowing by default."
            )
            return True, None

        if config.get("proceed_by_default", False):
            return True, None

        user_denial_message_template = f"User denied permission to {tool_name}"
        cli_arg_flag_name = config.get("cli_arg_flag")
        resource_param_key = config.get("resource_param_key")

        unknown_res_name_default = "unknown resource"
        if resource_param_key:
            unknown_res_name_default = f"unknown {resource_param_key}"

        resource_name = tool_params.get(
            resource_param_key, unknown_res_name_default
        )
        prompt_verb = config.get("prompt_verb", f"perform {tool_name}")

        if self.cli_args.auto_approve:
            return True, None

        if tool_name == "execute_command":
            requires_approval_param_key = config.get("requires_approval_param")
            llm_requires_approval = to_bool(
                tool_params.get(requires_approval_param_key, True)
            )  # Default to True for safety

            allow_all = getattr(self.cli_args, config.get("cli_arg_flag"))
            allow_safe_if_llm_safe = (
                getattr(self.cli_args, config.get("cli_arg_safe_flag")) and
                not llm_requires_approval
            )

            if allow_all or allow_safe_if_llm_safe:
                return True, None

            prompt_msg = f"Allow {prompt_verb}: '{resource_name}'? (y/n)"
            if request_user_confirmation(prompt_msg):
                return True, None
            return False, f"{user_denial_message_template}: {resource_name}"

        if cli_arg_flag_name and getattr(
            self.cli_args, cli_arg_flag_name, False
        ):
            # General permission flag for the tool type
            return True, None

        # Fallback to manual confirmation
        confirm_action_prompt = f"Allow {prompt_verb}: '{resource_name}'? (y/n)"
        if tool_name == "browser_action":
            action_key = config.get("action_param_key", "action")
            action = tool_params.get(action_key, "unknown action")
            confirm_action_prompt = (
                f"Allow {prompt_verb} '{action}' on '{resource_name}'? (y/n)"
            )

        if request_user_confirmation(confirm_action_prompt):
            return True, None

        return False, f"{user_denial_message_template}: {resource_name}"

    def _run_tool(self, tool_use: ToolUse) -> str:
        tool_name = tool_use.name
        tool_params = tool_use.params

        if tool_name == "attempt_completion":
            return tool_params.get(
                "result", "Task considered complete by the assistant."
            )

        if tool_name == "plan_mode_respond" and self.mode != "plan":
            self.consecutive_tool_errors += 1
            return "Error: plan_mode_respond can only be used in PLAN MODE."

        tool_to_execute = self.tools_map.get(tool_name)

        if tool_to_execute is None:
            self.consecutive_tool_errors += 1
            return (
                f"Error: Unknown tool '{tool_name}'. Please choose from the "
                "available tools."
            )

        proceed, denial_msg = self._check_tool_permission(tool_name, tool_params)

        if not proceed:
            log_message = denial_msg or (
                f"Action '{tool_name}' was not approved to proceed due to an "
                "unspecified reason."
            )
            logging.info(f"Tool use denied: {log_message}")
            return log_message

        try:
            result_str = tool_to_execute.execute(
                tool_params, agent_tools_instance=self
            )

            if result_str.startswith("Error:"):
                self.consecutive_tool_errors += 1
                if tool_name == "replace_in_file" and (
                    "Error: Search block" in result_str or
                    "Error processing diff_blocks" in result_str
                ):
                    file_path = tool_params.get("path")
                    if file_path:
                        abs_path_str = str((Path(self.cwd)/file_path).resolve())
                        failures = self.diff_failure_tracker.get(abs_path_str, 0) + 1
                        self.diff_failure_tracker[abs_path_str] = failures
                        if failures >= MAX_DIFF_FAILURES_PER_FILE:
                            suggestion = REPLACE_SUGGESTION_MESSAGE_TEMPLATE.format(
                                file_path=file_path
                            )
                            result_str += suggestion
                            self.diff_failure_tracker[abs_path_str] = 0
            else:  # Successful tool execution
                self.consecutive_tool_errors = 0
                if tool_name == "replace_in_file":
                    file_path = tool_params.get("path")
                    if file_path:
                        abs_path_str = str((Path(self.cwd)/file_path).resolve())
                        if abs_path_str in self.diff_failure_tracker:
                            self.diff_failure_tracker[abs_path_str] = 0
                elif tool_name == "write_to_file":
                    file_path = tool_params.get("path")
                    if file_path:
                        abs_path_str = str((Path(self.cwd)/file_path).resolve())
                        if abs_path_str in self.diff_failure_tracker:
                            self.diff_failure_tracker[abs_path_str] = 0
                if tool_name in ["write_to_file", "replace_in_file"]:
                    self._auto_commit()

            if tool_name == "read_file" and not result_str.startswith("Error:"):
                file_path = tool_params.get("path")
                if file_path:
                    abs_path = Path(file_path)
                    if not abs_path.is_absolute():
                        abs_path = Path(self.cwd) / abs_path
                    self.memory.add_file_context(str(abs_path.resolve()), result_str)
            return result_str

        except NotImplementedError as e:
            logging.warning(f"Tool '{tool_name}' not fully implemented: {e}")
            return (
                f"Note: Tool '{tool_name}' is recognized but not fully "
                f"implemented. {str(e)}"
            )
        except ValueError as e:
            self.consecutive_tool_errors += 1
            logging.error(
                f"ValueError in tool '{tool_name}': {e}\n{traceback.format_exc()}"
            )
            return f"Error: Tool '{tool_name}' value error. Reason: {str(e)}"
        except Exception as e:  # noqa: E722
            self.consecutive_tool_errors += 1
            logging.error(
                f"Unexpected error in tool '{tool_name}': {e}\n{traceback.format_exc()}"
            )
            return f"Error: Tool '{tool_name}' failed. Reason: {str(e)}"

    def _process_file_mentions(
        self, user_input: str
    ) -> list[dict[str, str]]:
        if not user_input:
            return []

        mentioned_file_data: list[dict[str, str]] = []
        processed_abs_paths = set()
        path_strings = extract_file_mentions(user_input)

        for rel_path_str in path_strings:
            abs_path = os.path.abspath(os.path.join(self.cwd, rel_path_str))
            if abs_path in processed_abs_paths:
                continue

            try:
                if not os.path.exists(abs_path):
                    logging.warning(f"Mentioned file {abs_path} (from "
                                    f"'{rel_path_str}') not found.")
                    continue
                if not os.path.isfile(abs_path):
                    logging.warning(f"Mentioned path {abs_path} (from "
                                    f"'{rel_path_str}') is not a file.")
                    continue
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
                mentioned_file_data.append({"path": rel_path_str, "content": content})
                processed_abs_paths.add(abs_path)
                logging.info(
                    f"Read mentioned file: {abs_path} (from '{rel_path_str}')"
                )
            except FileNotFoundError:  # Should be caught by os.path.exists
                logging.warning(f"Mentioned file not found (race condition?): "
                                f"{abs_path} (from mention '{rel_path_str}')")
            except PermissionError:
                logging.warning(
                    f"Permission denied for file: {abs_path} "
                    f"(from mention '{rel_path_str}')")
            except IOError as e:
                logging.warning(f"IOError for file {abs_path} "
                                f"(from mention '{rel_path_str}'): {e}")
            except Exception as e:  # Catch any other unexpected errors
                logging.error(f"Error processing file {abs_path} "
                              f"(from mention '{rel_path_str}'): {e}",
                              exc_info=True)
        return mentioned_file_data

    def run_task(self, user_input: str, max_steps: int = 20) -> str:
        self.mentioned_file_contents = self._process_file_mentions(user_input)
        if self.mentioned_file_contents:
            logging.info(
                f"Processed @-mentions. Found {len(self.mentioned_file_contents)} "
                "valid files to load into prompt."
            )
            prepended_texts = [
                f'<file_content path="{fd["path"]}">\n{fd["content"]}\n</file_content>'
                for fd in self.mentioned_file_contents
            ]
            if prepended_texts:
                prepended_string = "\n".join(prepended_texts)
                user_input_for_history = (
                    f"{prepended_string}\n\n---\nOriginal user input:\n{user_input}"
                )
                logging.debug(
                    "Prepending mentioned file content. New input length: %d",
                    len(user_input_for_history)
                )
            else:
                user_input_for_history = user_input
        else:
            user_input_for_history = user_input

        self.memory.add_message("user", user_input_for_history)
        for _i in range(max_steps):
            if self.consecutive_tool_errors >= MAX_CONSECUTIVE_TOOL_ERRORS:
                self.memory.add_message("system", ADMONISHMENT_MESSAGE)
                self.consecutive_tool_errors = 0

            llm_response_obj = self.send_message(self.history)

            if llm_response_obj is None:
                no_reply_message = "LLM did not provide a response. Ending task."
                logging.error(no_reply_message)
                self.memory.add_message("system", no_reply_message)
                return no_reply_message

            assistant_response_content: Optional[str]
            if isinstance(llm_response_obj, LLMResponse):
                if llm_response_obj.usage:
                    self.current_session_cost += llm_response_obj.usage.cost
                if self.on_llm_response_callback:
                    self.on_llm_response_callback(
                        llm_response_obj, self.model_name, self.current_session_cost
                    )
                assistant_response_content = llm_response_obj.content
            else:
                assistant_response_content = llm_response_obj

            if assistant_response_content is None:
                logging.info("LLM responded with no content. Cost accumulated.")
                assistant_response_content = ""

            self.memory.add_message("assistant", assistant_response_content)
            parsed_responses = parse_assistant_message(assistant_response_content)

            text_parts = [
                p.content for p in parsed_responses if isinstance(p, TextContent)
            ]
            malformed_error = any(
                p.content.startswith(MALFORMED_TOOL_PREFIX) for p in parsed_responses
                if isinstance(p, TextContent)
            )
            if malformed_error:
                self.consecutive_tool_errors += 1
            final_text_response = "\n".join(text_parts).strip()

            tool_uses = [p for p in parsed_responses if isinstance(p, ToolUse)]

            if not tool_uses:
                if not malformed_error and final_text_response:
                    self.consecutive_tool_errors = 0
                return final_text_response

            tool_to_run = tool_uses[0]
            if tool_to_run.name == "attempt_completion":
                res = tool_to_run.params.get("result", "")
                return f"{final_text_response}\n{res}".strip() if final_text_response else res

            tool_result = self._run_tool(tool_to_run)
            self.memory.add_message(
                "user", f"Result of {tool_to_run.name}:\n{tool_result}"
            )

        return "Max steps reached without completion."


if __name__ == "__main__":
    from src.llm import MockLLM

    def mock_send_message_for_agent_constructor(
        history: List[Dict[str, str]],
    ) -> Optional[LLMResponse]:
        logging.info("\n--- Mock LLM Call (via agent's send_message) ---")
        for msg in history[-2:]:
            logging.info(
                f"  {msg['role'].upper()}: {msg['content'][:200]}"
                f"{'...' if len(msg['content']) > 200 else ''}"
            )
        last_user_msg_content = history[-1]["content"] if history and \
                                history[-1]["role"] == "user" else ""
        dummy_usage = LLMUsageInfo(prompt_tokens=5, completion_tokens=5, cost=0.0)

        if "list files in src" in last_user_msg_content:
            return LLMResponse(content="<tool_use><tool_name>list_files</tool_name>"
                                       "<params><path>src</path></params></tool_use>",
                               usage=dummy_usage)
        return LLMResponse(content=("<text_content>Default mock response "
                                    "for __main__.</text_content>"),
                           usage=dummy_usage)

    Path("src").mkdir(exist_ok=True)
    (Path("src") / "agent.py").write_text(
        "class DeveloperAgent:\n  pass # Dummy for tests"
    )

    logging.info("--- Testing Agent with Response Exhaustion ---")
    exhausted_responses = [(
        "<tool_use><tool_name>read_file</tool_name>"
        "<params><path>src/agent.py</path></params></tool_use>"
    )]
    exhausted_llm = MockLLM(exhausted_responses)
    agent_exhaustion_test = DeveloperAgent(
        send_message=exhausted_llm.send_message, cwd="."
    )

    try:
        result1 = agent_exhaustion_test.run_task("Read agent.py", max_steps=2)
        logging.info(f"Result 1 (exhaustion test): {result1}")
        assert result1 == ""
        assert agent_exhaustion_test.history[-1]["role"] == "user"
        assert agent_exhaustion_test.history[-2]["role"] == "assistant"
        assert agent_exhaustion_test.history[-2]["content"] == ""
    except Exception: # pylint: disable=broad-except
        assert agent_exhaustion_test.history[-1]["role"] == "system"
    except Exception as e: # pylint: disable=broad-except
        logging.error(
            f"Error during agent exhaustion test: {e}\n{traceback.format_exc()}"
        )
    finally:
        if (Path("src") / "agent.py").exists():
            (Path("src") / "agent.py").unlink()
        try:
            Path("src").rmdir()
        except OSError:
            logging.warning(
                "Note: src directory may not be empty or other tests use it."
            )
        pass
