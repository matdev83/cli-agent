import argparse
from src.agent import DeveloperAgent
from src.llm import MockLLM


def create_agent(responses, mode="act", cwd="."):
    llm = MockLLM(responses)
    cli_args = argparse.Namespace(
        auto_approve=True,
        allow_read_files=True,
        allow_edit_files=True,
        allow_execute_safe_commands=True,
        allow_execute_all_commands=True,
        allow_use_browser=True,
        allow_use_mcp=True,
    )
    return DeveloperAgent(llm.send_message, cwd=cwd, cli_args=cli_args, mode=mode)


def test_plan_mode_respond_allowed_in_plan_mode(tmp_path):
    responses = [
        "<plan_mode_respond><response>Hello</response></plan_mode_respond>",
        "<text_content>done</text_content>"
    ]
    agent = create_agent(responses, mode="plan", cwd=str(tmp_path))
    result = agent.run_task("plan talk")
    assert result == "done"
    # tool result should be stored in history
    assert any("PlanModeRespondTool" in m["content"] for m in agent.history)


def test_plan_mode_respond_disallowed_in_act_mode(tmp_path):
    responses = [
        "<plan_mode_respond><response>oops</response></plan_mode_respond>",
        "<text_content>done</text_content>"
    ]
    agent = create_agent(responses, mode="act", cwd=str(tmp_path))
    result = agent.run_task("act task")
    assert result == "done"
    assert any("plan_mode_respond can only be used in PLAN MODE" in m["content"] for m in agent.history)


def test_set_mode_switches_behavior(tmp_path):
    responses = [
        "<plan_mode_respond><response>first</response></plan_mode_respond>",
        "<text_content>ok</text_content>",
        "<plan_mode_respond><response>second</response></plan_mode_respond>",
        "<text_content>done</text_content>"
    ]
    agent = create_agent(responses, mode="act", cwd=str(tmp_path))
    # first attempt in ACT mode should log error in history
    result1 = agent.run_task("foo")
    assert result1 == "ok"
    assert any("plan_mode_respond can only be used in PLAN MODE" in m["content"] for m in agent.history)
    agent.set_mode("plan")
    result2 = agent.run_task("bar")
    assert result2 == "done"
    assert any("PlanModeRespondTool" in m["content"] for m in agent.history)
