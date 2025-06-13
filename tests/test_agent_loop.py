from pathlib import Path
import argparse # Added import

from src.agent import DeveloperAgent


def test_agent_basic_loop(tmp_path: Path):
    responses = [
        "<write_to_file><path>out.txt</path><content>hello</content></write_to_file>",
        "<attempt_completion><result>done</result></attempt_completion>",
    ]

    def fake_send(history):
        return responses.pop(0)

    cli_args = argparse.Namespace(
        auto_approve=True, allow_read_files=True, allow_edit_files=True,
        allow_execute_safe_commands=True, allow_execute_all_commands=True,
        allow_use_browser=True, allow_use_mcp=True
    )
    agent = DeveloperAgent(fake_send, cwd=str(tmp_path), cli_args=cli_args)
    result = agent.run_task("start")
    assert result == "done"
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hello"


def test_agent_max_steps(tmp_path: Path):
    responses = ["<write_to_file><path>a.txt</path><content>x</content></write_to_file>"] * 3

    def fake_send(history):
        return responses.pop(0) if responses else ""

    cli_args = argparse.Namespace(
        auto_approve=True, allow_read_files=True, allow_edit_files=True,
        allow_execute_safe_commands=True, allow_execute_all_commands=True,
        allow_use_browser=True, allow_use_mcp=True
    )
    agent = DeveloperAgent(fake_send, cwd=str(tmp_path), cli_args=cli_args)
    result = agent.run_task("start", max_steps=2)
    assert result == "Max steps reached without completion."
