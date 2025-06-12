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
    cwd: str = ".",
    model: str = "mock",
    return_history: bool = False,
    llm_timeout: Optional[float] = None,
    matching_strictness: int = 100,
) -> str | tuple[str, list[dict[str, str]]]:
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
        auto_approve=auto_approve,
        matching_strictness=matching_strictness,
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
    parser.add_argument("--auto-approve", action="store_true", help="Auto approve commands")
    parser.add_argument("--cwd", default=".", help="Working directory")
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
            args.task,
            args.responses_file,
            auto_approve=args.auto_approve,
            cwd=args.cwd,
            model=args.model,
            llm_timeout=args.llm_timeout,
            matching_strictness=args.matching_strictness,
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
