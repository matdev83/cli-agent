import unittest
from unittest.mock import MagicMock, patch
import argparse # For creating a dummy cli_args namespace

# Assuming slash_commands.py and cli.py are in the same src directory
# and cli.py's AgentCliContext is the one used.
from slash_commands import RefreshCommand, SlashCommandRegistry, HelpCommand, ModelCommand, SetTimeoutCommand, PlanModeCommand, ActModeCommand, UndoCommand, UndoAllCommand
from cli import AgentCliContext # Using AgentCliContext from cli.py
from file_cache import FileCache # For type hinting and mocking

class TestRefreshCommand(unittest.TestCase):

    def setUp(self):
        self.mock_cli_args = argparse.Namespace(cwd=".") # Dummy args
        self.mock_display_update_func = MagicMock()
        self.mock_file_cache = MagicMock(spec=FileCache)

        # Create AgentCliContext with the real structure from cli.py
        self.agent_context = AgentCliContext(
            cli_args_namespace=self.mock_cli_args,
            display_update_func=self.mock_display_update_func,
            file_cache_instance=self.mock_file_cache,
            agent_instance=None # Not needed for this command
        )
        self.refresh_command = RefreshCommand()

    def test_refresh_command_executes_successfully(self):
        self.mock_file_cache.refresh.return_value = ["file1.txt", "file2.txt"]

        result = self.refresh_command.execute([], self.agent_context)

        self.mock_file_cache.refresh.assert_called_once()
        self.mock_display_update_func.assert_any_call("Starting file cache refresh...")
        expected_success_msg = "File cache refreshed successfully. Found 2 files."
        self.mock_display_update_func.assert_any_call(expected_success_msg)
        self.assertEqual(result, expected_success_msg)

    def test_refresh_command_handles_no_file_cache_in_context(self):
        self.agent_context.file_cache = None # Simulate missing file_cache

        result = self.refresh_command.execute([], self.agent_context)

        expected_error_msg = "Error: File cache context not available for refresh command."
        self.mock_display_update_func.assert_called_with(expected_error_msg)
        self.assertEqual(result, expected_error_msg)
        self.mock_file_cache.refresh.assert_not_called()

    def test_refresh_command_handles_invalid_file_cache_object(self):
        self.agent_context.file_cache = object() # Not a FileCache instance

        result = self.refresh_command.execute([], self.agent_context)

        expected_error_msg = "Error: File cache object is invalid or does not support refresh."
        self.mock_display_update_func.assert_called_with(expected_error_msg)
        self.assertEqual(result, expected_error_msg)

    def test_refresh_command_handles_exception_during_refresh(self):
        self.mock_file_cache.refresh.side_effect = Exception("Big disk error!")

        result = self.refresh_command.execute([], self.agent_context)

        self.mock_file_cache.refresh.assert_called_once()
        self.mock_display_update_func.assert_any_call("Starting file cache refresh...")
        expected_error_msg = "Error during file cache refresh: Big disk error!"
        self.mock_display_update_func.assert_any_call(expected_error_msg)
        self.assertEqual(result, expected_error_msg)

    def test_refresh_command_name_description_examples(self):
        self.assertEqual(self.refresh_command.name, "refresh")
        self.assertIsInstance(self.refresh_command.description, str)
        self.assertGreater(len(self.refresh_command.description), 0)
        self.assertEqual(self.refresh_command.usage_examples, ["/refresh"])

class TestSlashCommandRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = SlashCommandRegistry()
        self.mock_agent_context = MagicMock()

    def test_register_and_get_command(self):
        cmd = RefreshCommand() # Using RefreshCommand as a concrete example
        self.registry.register(cmd)
        self.assertIs(self.registry.get_command("refresh"), cmd)

    def test_register_duplicate_command_raises_error(self):
        cmd1 = RefreshCommand()
        cmd2 = RefreshCommand() # Another instance, but same name
        self.registry.register(cmd1)
        with self.assertRaisesRegex(ValueError, "Command 'refresh' is already registered."):
            self.registry.register(cmd2)

    def test_register_command_empty_name_raises_error(self):
        cmd = MagicMock(spec=RefreshCommand) # Use a real command type for spec
        cmd.name = ""
        with self.assertRaisesRegex(ValueError, "Command name cannot be empty."):
            self.registry.register(cmd)

    def test_register_command_name_with_space_raises_error(self):
        cmd = MagicMock(spec=RefreshCommand)
        cmd.name = "refresh cache"
        with self.assertRaisesRegex(ValueError, "Command name cannot contain spaces."):
            self.registry.register(cmd)

    def test_execute_unknown_command(self):
        result = self.registry.execute_command("unknown", [], self.mock_agent_context)
        self.assertTrue("Error: Unknown command '/unknown'" in result)

    def test_execute_command_success(self):
        cmd = MagicMock(spec=RefreshCommand)
        cmd.name = "testcmd"
        cmd.execute.return_value = "success"
        self.registry.register(cmd)

        result = self.registry.execute_command("testcmd", ["arg1"], self.mock_agent_context)

        cmd.execute.assert_called_once_with(["arg1"], self.mock_agent_context)
        self.assertEqual(result, "success")

    def test_execute_command_exception_in_command(self):
        cmd = MagicMock(spec=RefreshCommand)
        cmd.name = "failcmd"
        cmd.execute.side_effect = Exception("Test exception")
        self.registry.register(cmd)

        result = self.registry.execute_command("failcmd", [], self.mock_agent_context)
        self.assertTrue("Error executing command '/failcmd': Test exception" in result)

    def test_get_all_commands(self):
        cmd1 = RefreshCommand()
        cmd2 = ModelCommand() # Another concrete command
        self.registry.register(cmd1)
        self.registry.register(cmd2)

        all_cmds = self.registry.get_all_commands()
        self.assertIn(cmd1, all_cmds)
        self.assertIn(cmd2, all_cmds)
        self.assertEqual(len(all_cmds), 2)

# It might be good to have a small test for HelpCommand too,
# as its constructor takes the registry.
class TestHelpCommand(unittest.TestCase):
    def test_help_command_with_empty_registry(self):
        registry = SlashCommandRegistry()
        help_cmd = HelpCommand(registry)
        result = help_cmd.execute([], None)
        self.assertEqual(result, "No commands available.")

    def test_help_command_with_commands(self):
        registry = SlashCommandRegistry()
        registry.register(RefreshCommand()) # Example command
        registry.register(ModelCommand())   # Another one

        help_cmd = HelpCommand(registry)
        result = help_cmd.execute([], None)

        self.assertIn("/refresh", result)
        self.assertIn(RefreshCommand().description, result)
        self.assertIn("/model", result)
        self.assertIn(ModelCommand().description, result)
        self.assertTrue(result.startswith("Available commands:\n"))


if __name__ == '__main__':
    unittest.main()
