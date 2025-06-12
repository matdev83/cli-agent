from pathlib import Path

from src.agent import DeveloperAgent


def test_agent_basic_loop(tmp_path: Path):
    responses = [
        "<write_to_file><path>out.txt</path><content>hello</content></write_to_file>",
        "<attempt_completion><result>done</result></attempt_completion>",
    ]

    def fake_send(history):
        return responses.pop(0)

    agent = DeveloperAgent(fake_send, cwd=str(tmp_path), auto_approve=True)
    result = agent.run_task("start")
    assert result == "done"
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hello"


def test_agent_max_steps(tmp_path: Path):
    responses = ["<write_to_file><path>a.txt</path><content>x</content></write_to_file>"] * 3

    def fake_send(history):
        return responses.pop(0) if responses else ""

    agent = DeveloperAgent(fake_send, cwd=str(tmp_path), auto_approve=True)
    result = agent.run_task("start", max_steps=2)
    assert result == "Max steps reached without completion."
