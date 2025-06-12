from pathlib import Path
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

    agent = DeveloperAgent(fake_send, cwd=str(tmp_path), auto_approve=True)
    result = agent.run_task("start")
    assert result == "done"
    assert agent.memory.file_context[str(file_path)] == "hello"
    assert len(agent.memory.history) == 5
