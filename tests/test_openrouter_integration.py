import os
import shutil
import argparse
from pathlib import Path
import pytest

from src.cli import run_agent

APP_DIR = Path(__file__).resolve().parent.parent / "dev" / "app1"


@pytest.mark.skipif("OPENROUTER_API_KEY" not in os.environ, reason="requires OPENROUTER_API_KEY")
def test_openrouter_list_files(tmp_path: Path):
    workdir = tmp_path / "app"
    shutil.copytree(APP_DIR, workdir)

    cli_args = argparse.Namespace(
        model="deepseek/deepseek-chat-v3-0324:free",
        responses_file=None,
        auto_approve=True,
        allow_read_files=True,
        allow_edit_files=True,
        allow_execute_safe_commands=True,
        allow_execute_all_commands=True,
        allow_use_browser=True,
        allow_use_mcp=True,
        llm_timeout=120.0,
        cwd=str(workdir),
        matching_strictness=100,
        disable_git_auto_commits=True,
    )

    result, history = run_agent(
        "List the files in this directory using the list_files tool and then finish.",
        return_history=True,
        cli_args=cli_args,
    )

    # Assert that the list_files tool was called and its output contains the expected file
    assert any(
        isinstance(msg, dict) and msg.get("role") == "user" and "Result of list_files:" in msg.get("content", "") and "tictactoe.py" in msg.get("content", "")
        for msg in history
    )
    # Optionally, assert that the agent attempted completion
    assert len(history) > 0
