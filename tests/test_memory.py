from pathlib import Path
import argparse # Added import
from src.agent import DeveloperAgent


def test_memory_tracking(tmp_path: Path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("hello", encoding="utf-8")
    responses = [
        f"<read_file><path>{file_path.name}</path></read_file>",
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
    assert agent.memory.file_context[str(file_path)] == "hello"
    assert len(agent.memory.history) == 5
