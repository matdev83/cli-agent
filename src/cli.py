from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

import os

from .agent import DeveloperAgent
from .llm import MockLLM, OpenRouterLLM


def setup_logging(log_file: str = "agent.log") -> None:
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def run_agent(
    task: str,
    responses_file: str | None = None,
    *,
    auto_approve: bool = False,
    allow_read_files: bool = False,
    allow_edit_files: bool = False,
    allow_execute_safe_commands: bool = False,
    allow_execute_all_commands: bool = False,
    allow_use_browser: bool = False,
    allow_use_mcp: bool = False,
    disable_git_auto_commits: bool = False,
    cwd: str = ".",
    model: str = "mock",
    return_history: bool = False,
    llm_timeout: Optional[float] = None,
    matching_strictness: int = 100, # Kept as it's not purely an approval flag
    cli_args: Optional[argparse.Namespace] = None # Added cli_args
    # allow_read_files: bool = False, # Removed
    # allow_edit_files: bool = False, # Removed
    # allow_execute_safe_commands: bool = False, # Removed
    # allow_execute_all_commands: bool = False, # Removed
    # allow_use_browser: bool = False, # Removed
    # allow_use_mcp: bool = False, # Removed
) -> str | tuple[str, list[dict[str, str]]]:
    if cli_args is None:  # Provide default if not passed, though main() should always pass it.
        cli_args = argparse.Namespace(
            auto_approve=auto_approve,
            allow_read_files=allow_read_files,
            allow_edit_files=allow_edit_files,
            allow_execute_safe_commands=allow_execute_safe_commands,
            allow_execute_all_commands=allow_execute_all_commands,
            allow_use_browser=allow_use_browser,
            allow_use_mcp=allow_use_mcp,
        )
    else:
        cli_args.auto_approve = auto_approve
        cli_args.allow_read_files = allow_read_files
        cli_args.allow_edit_files = allow_edit_files
        cli_args.allow_execute_safe_commands = allow_execute_safe_commands
        cli_args.allow_execute_all_commands = allow_execute_all_commands
        cli_args.allow_use_browser = allow_use_browser
        cli_args.allow_use_mcp = allow_use_mcp
        cli_args.disable_git_auto_commits = disable_git_auto_commits

    if model == "mock":
        if not responses_file:
            raise ValueError("responses_file is required for mock model")
        # Ensure responses_file exists if model is mock (FileNotFoundError will be caught by main)
        llm = MockLLM.from_file(responses_file)
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            # Raising RuntimeError here, which will be caught by the generic Exception in main.
            # Could be a custom error or handled more specifically if desired.
            raise RuntimeError("OPENROUTER_API_KEY environment variable not set, required for non-mock models.")
        llm = OpenRouterLLM(model=model, api_key=api_key, timeout=llm_timeout)

    # The LLMWrapper protocol expects send_message to take temperature and max_tokens.
    # The DeveloperAgent's __init__ expects a send_message callable that matches
    # `Callable[[List[Dict[str, str]]], str]`.
    # The current LLM implementations `send_message` methods were updated to `Optional[str]`
    # and to accept optional temp/tokens.
    # DeveloperAgent needs to be updated to pass these if it's to use the full protocol,
    # or the LLM send_message methods need to align with what DeveloperAgent expects.
    # For now, assuming DeveloperAgent.send_message call signature is what llm.send_message provides.
    # The current DeveloperAgent.send_message(self.history) matches the basic signature.
    agent = DeveloperAgent(
        llm.send_message,
        cwd=cwd,
        cli_args=cli_args, # Pass the whole namespace
        matching_strictness=matching_strictness
        ,disable_git_auto_commits=disable_git_auto_commits
        # Removed individual approval flags, they are now in cli_args
    )
    result = agent.run_task(task)
    if return_history:
        return result, list(agent.history)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CLI developer agent")
    parser.add_argument("task", help="Task description")
    parser.add_argument(
        "--model",
        default="mock",
        help="Specify the LLM to use. 'mock' uses MockLLM (requires --responses-file). "
             "Other values (e.g., 'anthropic/claude-3-opus') use OpenRouterLLM (requires OPENROUTER_API_KEY)."
    )
    parser.add_argument(
        "--responses-file",
        help="Path to JSON file with mock LLM responses (required for 'mock' model).",
    )
    parser.add_argument("--auto-approve", action="store_true", help="Auto approve commands (legacy, granular flags take precedence)")
    parser.add_argument("--cwd", default=".", help="Working directory")
    parser.add_argument(
        "--allow-read-files",
        action="store_true",
        default=False,
        help="Automatically approve file reads (includes listing files/directories).",
    )
    parser.add_argument(
        "--allow-edit-files",
        action="store_true",
        default=False,
        help="Automatically approve file edits (includes creating new files).",
    )
    parser.add_argument(
        "--allow-execute-safe-commands",
        action="store_true",
        default=False,
        help="Automatically approve commands marked as 'safe' by the LLM.",
    )
    parser.add_argument(
        "--allow-execute-all-commands",
        action="store_true",
        default=False,
        help="Automatically approve ALL commands, including those not marked as 'safe'. Implies --allow-execute-safe-commands.",
    )
    parser.add_argument(
        "--allow-use-browser",
        action="store_true",
        default=False,
        help="Automatically approve browser usage.",
    )
    parser.add_argument(
        "--allow-use-mcp",
        action="store_true",
        default=False,
        help="Automatically approve MCP (Multi-Capability Plugin) usage.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for LLM API calls.",
    )
    parser.add_argument(
        "--matching-strictness",
        type=int,
        default=100,
        help="Set the string matching strictness for file edits (0-100, 100 is exact match).",
    )
    parser.add_argument(
        "--disable-git-auto-commits",
        action="store_true",
        default=False,
        help="Disable automatic git commits after file modifications.",
    )
    args = parser.parse_args(argv)

    if not (0 <= args.matching_strictness <= 100):
        parser.error("--matching-strictness must be between 0 and 100.")

    setup_logging()
    logging.info("Starting agent for task: %s with model %s", args.task, args.model)

    if args.model == "mock" and not args.responses_file:
        # This specific check is good, but ValueError in run_agent also covers it.
        # parser.error will exit, which is fine.
        parser.error("--responses-file is required when using the mock model.")

    try:
        result = run_agent(
            task=args.task,
            responses_file=args.responses_file,
            auto_approve=args.auto_approve,
            allow_read_files=args.allow_read_files,
            allow_edit_files=args.allow_edit_files,
            allow_execute_safe_commands=args.allow_execute_safe_commands,
            allow_execute_all_commands=args.allow_execute_all_commands,
            allow_use_browser=args.allow_use_browser,
            allow_use_mcp=args.allow_use_mcp,
            disable_git_auto_commits=args.disable_git_auto_commits,
            cwd=args.cwd,
            model=args.model,
            llm_timeout=args.llm_timeout,
            matching_strictness=args.matching_strictness,
            cli_args=args # Pass the whole args namespace
        )
        print(result) # Print result only on success
        logging.info("Agent completed successfully")
        return 0
    except FileNotFoundError as e:
        logging.error("File not found: %s", e, exc_info=True)
        print(f"Error: A required file was not found. Details: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        logging.error("Invalid value provided: %s", e, exc_info=True)
        # This can be from responses_file not provided for mock, or other ValueErrors in agent/tools.
        print(f"Error: An invalid value was encountered. Details: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e: # Catching RuntimeError specifically e.g. for API key missing
        logging.error("Runtime error encountered: %s", e, exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as exc:  # Catch-all for other unexpected failures
        logging.exception("Agent run failed due to an unexpected error.") # Logs full traceback
        print(f"An unexpected error occurred: {exc}", file=sys.stderr)
        # The prompt to continue was removed as per typical CLI error handling.
        # If debugging is needed, logs are the place.
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
