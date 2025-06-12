import os
import shutil
from pathlib import Path
import pytest
from typing import Any, Dict, List

from src.cli import run_agent

APP_DIR = Path(__file__).resolve().parent.parent / "dev" / "app1"


@pytest.mark.skipif("OPENROUTER_API_KEY" not in os.environ, reason="requires OPENROUTER_API_KEY")
def test_openrouter_list_files(tmp_path: Path):
    workdir = tmp_path / "app"
    shutil.copytree(APP_DIR, workdir)

    result, history = run_agent(
        "List the files in this directory using the list_files tool and then finish.",
        responses_file=None,
        auto_approve=True,
        cwd=str(workdir),
        model="deepseek/deepseek-chat-v3-0324:free",
        return_history=True,
    )

    # Assert that the list_files tool was called and its output contains the expected file
    assert any(
        isinstance(msg, dict) and msg.get("role") == "user" and "Result of list_files:" in msg.get("content", "") and "tictactoe.py" in msg.get("content", "")
        for msg in history
    )
    # Optionally, assert that the agent attempted completion
    assert any(
        isinstance(msg, dict) and msg.get("role") == "assistant" and "<attempt_completion>" in msg.get("content", "")
        for msg in history
    )
