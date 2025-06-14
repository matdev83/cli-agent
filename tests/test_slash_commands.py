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
    HelpCommand
    # AgentCliContext will be imported from src.cli
)
from src.cli import AgentCliContext # Corrected import


# --- Mock Commands for HelpCommand Tests ---
class MockCmdA(SlashCommand):
    @property
    def name(self) -> str: return "a_mock"
    @property
    def description(self) -> str: return "Mock A description."
    @property
    def usage_examples(self) -> List[str]: return ["/a_mock example1", "/a_mock example2"]
    def execute(self, args: List[str], agent_context: Any=None) -> Optional[str]: return "Executed A"

class MockCmdB(SlashCommand):
    @property
    def name(self) -> str: return "b_mock"
    @property
    def description(self) -> str: return "Mock B description."
    @property
    def usage_examples(self) -> List[str]: return ["/b_mock example"]
    def execute(self, args: List[str], agent_context: Any=None) -> Optional[str]: return "Executed B"

class MockCmdCNoExamples(SlashCommand):
    @property
    def name(self) -> str: return "c_no_examples"
    @property
    def description(self) -> str: return "Mock C description (no examples)."
    @property
    def usage_examples(self) -> List[str]: return [] # Or None
    def execute(self, args: List[str], agent_context: Any=None) -> Optional[str]: return "Executed C"

# --- End Mock Commands ---

# Mock display update function for context
def mock_display_update_func(message: str):
    # print(f"UI_UPDATE: {message}") # For test debugging
    pass

# Helper to create a default AgentCliContext
def create_default_context(initial_args: Optional[dict] = None, file_cache_instance: Optional[Any] = None) -> AgentCliContext:
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
    # If no file_cache_instance is provided, default to None or a MagicMock
    # For simplicity in this diff, let's assume None is acceptable for most of these tests.
    # Tests requiring a real FileCache might need to pass it explicitly.
    return AgentCliContext(cli_args_namespace=cli_ns, display_update_func=mock_display_update_func, file_cache_instance=file_cache_instance)

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
            @property
            def description(self) -> str: return "Test description"
            @property
            def usage_examples(self) -> List[str]: return ["/test"]
            def execute(self, args: List[str], agent_context: Any=None) -> Optional[str]: return None

        with pytest.raises(ValueError, match="Command name cannot be empty."):
            registry.register(EmptyNameCommand())

    def test_register_command_name_with_spaces_raises_error(self):
        registry = SlashCommandRegistry()
        class SpaceNameCommand(SlashCommand):
            @property
            def name(self) -> str: return "cmd with space"
            @property
            def description(self) -> str: return "Test description"
            @property
            def usage_examples(self) -> List[str]: return ["/test"]
            def execute(self, args: List[str], agent_context: Any=None) -> Optional[str]: return None

        with pytest.raises(ValueError, match="Command name cannot contain spaces."):
            registry.register(SpaceNameCommand())

    def test_get_unknown_command_returns_none(self):
        registry = SlashCommandRegistry()
        assert registry.get_command("unknown") is None

    def test_execute_unknown_command(self):
        registry = SlashCommandRegistry()
        context = create_default_context() # Pass None for file_cache by default
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
        context = create_default_context() # Pass None for file_cache
        result = command.execute(["new_model_name"], context)
        assert result == "Model set to: new_model_name"
        assert context.cli_args.model == "new_model_name"

    def test_model_command_no_args(self):
        command = ModelCommand()
        context = create_default_context() # Pass None for file_cache
        result = command.execute([], context)
        assert "Error: Missing model name" in result
        assert context.cli_args.model == "mock_default_model" # Should not change

    def test_model_command_mock_model_no_responses_file_warning(self):
        command = ModelCommand()
        # Simulate context where responses_file is not set initially
        context = create_default_context() # Pass None for file_cache
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
        context = create_default_context() # Pass None for file_cache
        result = command.execute(["300.5"], context)
        assert result == "LLM timeout set to: 300.5 seconds."
        assert context.cli_args.llm_timeout == 300.5

    def test_set_timeout_command_no_args(self):
        command = SetTimeoutCommand()
        context = create_default_context() # Pass None for file_cache
        result = command.execute([], context)
        assert "Error: Missing timeout value" in result
        assert context.cli_args.llm_timeout == 120.0 # Default, should not change

    def test_set_timeout_command_invalid_value(self):
        command = SetTimeoutCommand()
        context = create_default_context() # Pass None for file_cache
        result = command.execute(["abc"], context)
        assert "Error: Invalid timeout value. Must be a number." in result

    def test_set_timeout_command_negative_value(self):
        command = SetTimeoutCommand()
        context = create_default_context() # Pass None for file_cache
        result = command.execute(["-50"], context)
        assert "Error: Timeout must be a positive number." in result

    def test_set_timeout_command_execution_no_context(self):
        command = SetTimeoutCommand()
        result = command.execute(["200"], None)
        assert "Would set timeout to 200.0s (agent_context not fully available for update)" in result


class TestPlanModeCommand:
    def test_plan_mode_command_execution(self):
        command = PlanModeCommand()
        context = create_default_context() # Basic context, pass None for file_cache
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
        context = create_default_context() # Basic context, pass None for file_cache
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


class TestHelpCommand:
    def test_help_command_registered_and_displays_self(self):
        registry = SlashCommandRegistry()
        help_cmd = HelpCommand(registry)
        registry.register(help_cmd)

        context = create_default_context() # Pass None for file_cache
        # result = help_cmd.execute([], context) # Direct execution
        result = registry.execute_command("help", [], context) # Execution via registry

        assert result is not None
        assert "Available commands:" in result
        assert "/help" in result
        assert "Displays help information for all available slash commands." in result
        assert "/help" in result # Example for /help itself

    def test_help_command_empty_registry(self):
        empty_registry = SlashCommandRegistry()
        help_cmd = HelpCommand(empty_registry)
        # Note: HelpCommand itself is not added to this "empty" registry
        # for the purpose of this specific test of its behavior with no *other* commands.
        # If HelpCommand is always pre-registered, this test might need adjustment
        # or interpretation (e.g., it shows only itself).
        # However, the HelpCommand's execute logic fetches from the registry it's given.
        # If it's registered, it will find itself. If we want to test what it says
        # when the registry it *knows about* has no *other* commands, that's fine.
        # The prompt implies testing HelpCommand's output given a truly empty registry.
        # So, we pass an empty one to its constructor, but don't register HelpCommand itself *into that specific registry instance*.

        # If the test means "what if help is the *only* command registered":
        # registry = SlashCommandRegistry()
        # help_cmd_instance = HelpCommand(registry)
        # registry.register(help_cmd_instance)
        # result = help_cmd_instance.execute([])
        # This would show help for "help".

        # Per prompt: "Execute the help_command" with an empty registry.
        # This implies the HelpCommand instance operates on a registry that has no commands.
        result = help_cmd.execute([], None) # AgentContext not used by HelpCommand
        assert result == "No commands available."

    def test_help_command_lists_commands_alphabetically_with_details(self):
        registry = SlashCommandRegistry()
        cmd_b = MockCmdB()
        cmd_a = MockCmdA()
        help_cmd = HelpCommand(registry) # help command itself

        registry.register(cmd_b) # Register b first
        registry.register(cmd_a) # Register a second
        registry.register(help_cmd) # Register help

        context = create_default_context() # Pass None for file_cache
        result = registry.execute_command("help", [], context)

        assert result is not None
        # Check order and content
        # Expected order: a_mock, b_mock, help
        a_mock_pos = result.find("/a_mock")
        b_mock_pos = result.find("/b_mock")
        help_pos = result.find("/help") # help command itself

        assert all(pos != -1 for pos in [a_mock_pos, b_mock_pos, help_pos]), "All commands should be listed"
        assert a_mock_pos < b_mock_pos < help_pos, "Commands should be in alphabetical order"

        # Check details for a_mock
        assert "  /a_mock\n" in result
        assert "    Description: Mock A description.\n" in result
        assert "    Examples:\n" in result
        assert "      /a_mock example1\n" in result
        assert "      /a_mock example2\n" in result

        # Check details for b_mock
        assert "  /b_mock\n" in result
        assert "    Description: Mock B description.\n" in result
        assert "    Examples:\n" in result
        assert "      /b_mock example\n" in result

        # Check details for help
        assert "  /help\n" in result
        assert "    Description: Displays help information for all available slash commands.\n" in result
        assert "    Examples:\n" in result
        assert "      /help\n" in result # Usage example for /help itself

    def test_help_command_with_no_usage_examples(self):
        registry = SlashCommandRegistry()
        cmd_c_no_ex = MockCmdCNoExamples()
        help_cmd = HelpCommand(registry)

        registry.register(cmd_c_no_ex)
        registry.register(help_cmd)

        context = create_default_context() # Pass None for file_cache
        result = registry.execute_command("help", [], context)

        assert result is not None
        assert "/c_no_examples" in result
        assert "Mock C description (no examples)." in result
        assert "      /c_no_examples example" not in result # Ensure no example line is printed if list is empty
        assert "    Examples:\n" not in result[result.find("/c_no_examples") : result.find("/help")] # Check specific section
        # A more robust check for "Examples:" section for c_no_examples:
        c_section_start = result.find("/c_no_examples")
        help_section_start = result.find("/help") # Assuming help is next due to alphabetical or it's the only other one

        c_section_text = ""
        if c_section_start != -1 and help_section_start != -1 and help_section_start > c_section_start:
            c_section_text = result[c_section_start:help_section_start]
        elif c_section_start != -1: # c_no_examples is the last command
            c_section_text = result[c_section_start:]

        assert "    Examples:" not in c_section_text
        assert "    Description: Mock C description (no examples).\n" in c_section_text # Ensure description is there


# --- Test for command properties ---
concrete_command_classes = [
    ModelCommand,
    SetTimeoutCommand,
    PlanModeCommand,
    ActModeCommand,
    # HelpCommand needs a registry, so we instantiate it directly in parametrize
]

@pytest.mark.parametrize("command_instance", [
    ModelCommand(),
    SetTimeoutCommand(),
    PlanModeCommand(),
    ActModeCommand(),
    HelpCommand(SlashCommandRegistry()) # Pass a dummy registry
])
def test_all_commands_have_valid_description_and_usage_examples(command_instance: SlashCommand):
    """
    Tests that all concrete SlashCommand instances have valid description
    and usage_examples properties.
    """
    command = command_instance

    # Test description
    assert isinstance(command.description, str), \
        f"Command /{command.name} description should be a string, got {type(command.description)}"
    assert len(command.description) > 0, \
        f"Command /{command.name} description should not be empty"

    # Test usage_examples
    assert isinstance(command.usage_examples, list), \
        f"Command /{command.name} usage_examples should be a list, got {type(command.usage_examples)}"
    assert len(command.usage_examples) > 0, \
        f"Command /{command.name} usage_examples should not be empty"

    for example in command.usage_examples:
        assert isinstance(example, str), \
            f"Command /{command.name} usage_example item '{example}' should be a string, got {type(example)}"
        assert len(example) > 0, \
            f"Command /{command.name} usage_example item '{example}' should not be empty"
