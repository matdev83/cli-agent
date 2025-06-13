import pytest
import argparse
from typing import List, Optional, Any

from src.slash_commands import (
    SlashCommand,
    SlashCommandRegistry,
    ModelCommand,
    SetTimeoutCommand,
    PlanModeCommand,
    ActModeCommand,
    AgentCliContext
)

# Mock display update function for context
def mock_display_update_func(message: str):
    # print(f"UI_UPDATE: {message}") # For test debugging
    pass

# Helper to create a default AgentCliContext
def create_default_context(initial_args: Optional[dict] = None) -> AgentCliContext:
    parser = argparse.ArgumentParser()
    # Add arguments that our commands might interact with
    parser.add_argument("--model", default="mock_default_model")
    parser.add_argument("--responses-file", default="mock_responses.json")
    parser.add_argument("--llm-timeout", type=float, default=120.0)

    # Simulate parsed args. If initial_args is provided, use it.
    default_initial_args = []
    if initial_args:
        for k, v in initial_args.items():
            default_initial_args.append(f"--{k.replace('_', '-')}")
            default_initial_args.append(str(v))

    cli_ns = parser.parse_args(default_initial_args if default_initial_args else [])
    return AgentCliContext(cli_args_namespace=cli_ns, display_update_func=mock_display_update_func)

class TestSlashCommandRegistry:
    def test_register_command(self):
        registry = SlashCommandRegistry()
        command = ModelCommand()
        registry.register(command)
        assert registry.get_command("model") == command

    def test_register_duplicate_command_raises_error(self):
        registry = SlashCommandRegistry()
        command1 = ModelCommand()
        command2 = ModelCommand() # Another instance, but same name
        registry.register(command1)
        with pytest.raises(ValueError, match="Command 'model' is already registered."):
            registry.register(command2)

    def test_register_command_empty_name_raises_error(self):
        registry = SlashCommandRegistry()
        class EmptyNameCommand(SlashCommand):
            @property
            def name(self) -> str: return ""
            def execute(self, args: List[str], agent_context: Any=None) -> Optional[str]: return None

        with pytest.raises(ValueError, match="Command name cannot be empty."):
            registry.register(EmptyNameCommand())

    def test_register_command_name_with_spaces_raises_error(self):
        registry = SlashCommandRegistry()
        class SpaceNameCommand(SlashCommand):
            @property
            def name(self) -> str: return "cmd with space"
            def execute(self, args: List[str], agent_context: Any=None) -> Optional[str]: return None

        with pytest.raises(ValueError, match="Command name cannot contain spaces."):
            registry.register(SpaceNameCommand())

    def test_get_unknown_command_returns_none(self):
        registry = SlashCommandRegistry()
        assert registry.get_command("unknown") is None

    def test_execute_unknown_command(self):
        registry = SlashCommandRegistry()
        context = create_default_context()
        result = registry.execute_command("unknown", [], context)
        assert "Error: Unknown command '/unknown'" in result

    def test_get_all_commands(self):
        registry = SlashCommandRegistry()
        cmd1 = ModelCommand()
        cmd2 = SetTimeoutCommand()
        registry.register(cmd1)
        registry.register(cmd2)
        all_cmds = registry.get_all_commands()
        assert len(all_cmds) == 2
        assert cmd1 in all_cmds
        assert cmd2 in all_cmds

class TestModelCommand:
    def test_model_command_execution(self):
        command = ModelCommand()
        context = create_default_context()
        result = command.execute(["new_model_name"], context)
        assert result == "Model set to: new_model_name"
        assert context.cli_args.model == "new_model_name"

    def test_model_command_no_args(self):
        command = ModelCommand()
        context = create_default_context()
        result = command.execute([], context)
        assert "Error: Missing model name" in result
        assert context.cli_args.model == "mock_default_model" # Should not change

    def test_model_command_mock_model_no_responses_file_warning(self):
        command = ModelCommand()
        # Simulate context where responses_file is not set initially
        context = create_default_context()
        context.cli_args.responses_file = None # Explicitly set to None

        result = command.execute(["mock"], context)
        assert "Warning: Model set to 'mock', but --responses-file is not currently set." in result
        assert context.cli_args.model == "mock"

    def test_model_command_execution_no_context(self):
        command = ModelCommand()
        result = command.execute(["new_model_name"], None)
        assert "Would set model to 'new_model_name' (agent_context not fully available for update)" in result

class TestSetTimeoutCommand:
    def test_set_timeout_command_execution(self):
        command = SetTimeoutCommand()
        context = create_default_context()
        result = command.execute(["300.5"], context)
        assert result == "LLM timeout set to: 300.5 seconds."
        assert context.cli_args.llm_timeout == 300.5

    def test_set_timeout_command_no_args(self):
        command = SetTimeoutCommand()
        context = create_default_context()
        result = command.execute([], context)
        assert "Error: Missing timeout value" in result
        assert context.cli_args.llm_timeout == 120.0 # Default, should not change

    def test_set_timeout_command_invalid_value(self):
        command = SetTimeoutCommand()
        context = create_default_context()
        result = command.execute(["abc"], context)
        assert "Error: Invalid timeout value. Must be a number." in result

    def test_set_timeout_command_negative_value(self):
        command = SetTimeoutCommand()
        context = create_default_context()
        result = command.execute(["-50"], context)
        assert "Error: Timeout must be a positive number." in result

    def test_set_timeout_command_execution_no_context(self):
        command = SetTimeoutCommand()
        result = command.execute(["200"], None)
        assert "Would set timeout to 200.0s (agent_context not fully available for update)" in result


class TestPlanModeCommand:
    def test_plan_mode_command_execution(self):
        command = PlanModeCommand()
        context = create_default_context() # Basic context
        # Mock the set_mode method for this test
        def mock_set_mode(mode_name):
            context.mode = mode_name
        context.set_mode = mock_set_mode

        result = command.execute([], context)
        assert result == "Agent switched to PLAN MODE."
        assert hasattr(context, 'mode') and context.mode == "PLAN"

    def test_plan_mode_command_execution_no_set_mode_in_context(self):
        command = PlanModeCommand()
        # Pass None to simulate a context where set_mode is not available
        result = command.execute([], None)
        assert "PLAN MODE activated (actual mode switching pending full implementation)" in result

class TestActModeCommand:
    def test_act_mode_command_execution(self):
        command = ActModeCommand()
        context = create_default_context() # Basic context
        def mock_set_mode(mode_name):
            context.mode = mode_name
        context.set_mode = mock_set_mode

        result = command.execute([], context)
        assert result == "Agent switched to ACT MODE."
        assert hasattr(context, 'mode') and context.mode == "ACT"

    def test_act_mode_command_execution_no_set_mode_in_context(self):
        command = ActModeCommand()
        # Pass None to simulate a context where set_mode is not available
        result = command.execute([], None)
        assert "ACT MODE activated (actual mode switching pending full implementation)" in result

# Minimal AgentCliContext for tests that don't deeply inspect it
class MinimalAgentCliContext:
    def __init__(self, cli_args_dict=None, update_func=None):
        self.cli_args = argparse.Namespace(**(cli_args_dict or {}))
        self.display_update_func = update_func or (lambda x: None)
        self.mode = "ACT" # Default

    def set_mode(self, mode_name: str):
        self.mode = mode_name
        self.display_update_func(f"Context: Mode changed to {mode_name}")

# Example of testing command execution through registry
def test_registry_executes_model_command():
    registry = SlashCommandRegistry()
    registry.register(ModelCommand())
    context = create_default_context({"model": "old_model"})

    result = registry.execute_command("model", ["new_model"], context)
    assert result == "Model set to: new_model"
    assert context.cli_args.model == "new_model"
