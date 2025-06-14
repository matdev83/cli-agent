import subprocess
import argparse
from pathlib import Path
import logging # Moved here

from src.agent import DeveloperAgent


def init_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)


def commit_count(path: Path) -> int:
    out = subprocess.check_output(["git", "rev-list", "--count", "HEAD"], cwd=path)
    return int(out.decode().strip())


def test_auto_commit_after_write(tmp_path: Path, caplog):
    caplog.set_level(logging.INFO) # Ensure INFO messages are captured
    init_repo(tmp_path)
    responses = [
        "<write_to_file><path>a.txt</path><content>data</content></write_to_file>",
        "<attempt_completion><result>done</result></attempt_completion>",
    ]

    def fake_send(_):
        return responses.pop(0)

    cli_args = argparse.Namespace(
        auto_approve=True,
        allow_read_files=True,
        allow_edit_files=True,
        allow_execute_safe_commands=True,
        allow_execute_all_commands=True,
        allow_use_browser=True,
        allow_use_mcp=True,
        disable_git_auto_commits=False,
    )

    agent = DeveloperAgent(fake_send, cwd=str(tmp_path), cli_args=cli_args)
    result = agent.run_task("task")
    assert result == "done"
    assert commit_count(tmp_path) == 1
    log = subprocess.check_output(["git", "log", "--oneline"], cwd=tmp_path).decode().splitlines()[0]
    commit_id = log.split()[0]
    assert "Auto-commit" in log
    # Check logging output for the auto-commit message
    assert f"Auto-commit id: {commit_id}" in caplog.text


def test_disable_git_auto_commit(tmp_path: Path):
    init_repo(tmp_path)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "initial"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    initial = commit_count(tmp_path)

    responses = [
        "<write_to_file><path>a.txt</path><content>data</content></write_to_file>",
        "<attempt_completion><result>done</result></attempt_completion>",
    ]

    def fake_send(_):
        return responses.pop(0)

    cli_args = argparse.Namespace(
        auto_approve=True,
        allow_read_files=True,
        allow_edit_files=True,
        allow_execute_safe_commands=True,
        allow_execute_all_commands=True,
        allow_use_browser=True,
        allow_use_mcp=True,
        disable_git_auto_commits=True,
    )

    agent = DeveloperAgent(fake_send, cwd=str(tmp_path), cli_args=cli_args)
    result = agent.run_task("task")
    assert result == "done"
    assert commit_count(tmp_path) == initial

