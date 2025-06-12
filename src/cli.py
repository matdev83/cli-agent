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
) -> str | tuple[str, list[dict[str, str]]]:
    if model == "mock":
        if not responses_file:
            raise ValueError("responses_file is required for mock model")
        llm = MockLLM.from_file(responses_file)
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        llm = OpenRouterLLM(model=model, api_key=api_key)

    agent = DeveloperAgent(llm.send_message, cwd=cwd, auto_approve=auto_approve)
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
        help="Model name to use ('mock' or OpenRouter model name)",
    )
    parser.add_argument(
        "--responses-file",
        help="Path to JSON file with mock LLM responses",
    )
    parser.add_argument("--auto-approve", action="store_true", help="Auto approve commands")
    parser.add_argument("--cwd", default=".", help="Working directory")
    args = parser.parse_args(argv)

    setup_logging()
    logging.info("Starting agent for task: %s", args.task)

    if args.model == "mock" and not args.responses_file:
        parser.error("--responses-file is required when using the mock model")

    try:
        result = run_agent(
            args.task,
            args.responses_file,
            auto_approve=args.auto_approve,
            cwd=args.cwd,
            model=args.model,
        )
    except Exception as exc:  # pragma: no cover - unexpected failures
        logging.exception("Agent run failed")
        print(f"Error: {exc}", file=sys.stderr)
        resp = input("Continue anyway? [y/N]: ").strip().lower()
        if resp not in {"y", "yes"}:
            print("Aborted.", file=sys.stderr)
            return 1
        return 0

    print(result)
    logging.info("Agent completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
