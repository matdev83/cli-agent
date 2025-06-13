from __future__ import annotations

import abc
from typing import List, Optional, Any, Dict

# It's generally better to import specific names if utils is a module
# For example: from .utils import revert_to_commit, revert_to_state_before_commit
# However, the prompt asked for `from . import utils`
from . import utils
# For type hinting agent_context.agent. DeveloperAgent might cause circular dependency if imported directly.
# Using 'Any' for now as per instructions, but consider forward reference 'DeveloperAgent' if appropriate.
# from .agent import DeveloperAgent # Example of potential direct import

class SlashCommand(abc.ABC):
    """
    Abstract base class for slash commands.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the slash command (e.g., 'model', 'set-timeout')."""
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """A brief description of what the command does."""
        pass

    @property
    @abc.abstractmethod
    def usage_examples(self) -> List[str]:
        """A list of usage examples for the command."""
        pass

    @abc.abstractmethod
    def execute(self, args: List[str], agent_context: Any = None) -> Optional[str]:
        """
        Executes the slash command.

        Args:
            args: A list of arguments passed to the command.
            agent_context: An optional context object that can provide
                           access to agent or CLI state.

        Returns:
            An optional string message to be displayed to the user.
        """
        pass

# Example of how agent_context might be structured (optional, for illustration)
# class AgentCliContext:
#     def __init__(self, agent, cli_args, display_update_func):
#         self.agent = agent
#         self.cli_args = cli_args
#         self.display_update_func = display_update_func

class ModelCommand(SlashCommand):
    """
    Sets the language model to be used by the agent.
    Example: /model anthropic/claude-3-opus
    """
    @property
    def name(self) -> str:
        return "model"

    @property
    def description(self) -> str:
        return "Sets the language model to be used by the agent."

    @property
    def usage_examples(self) -> List[str]:
        return ["/model anthropic/claude-3-opus", "/model mock"]

    def execute(self, args: List[str], agent_context: Any = None) -> Optional[str]:
        if not args:
            return "Error: Missing model name. Usage: /model <model_name>"

        model_name = args[0]

        if agent_context and hasattr(agent_context, 'cli_args'):
            # In a real scenario, we might want to validate the model_name further
            # or check if it's different from the current one.
            agent_context.cli_args.model = model_name
            # We also need to handle the case where the model is 'mock'
            # and ensure responses_file is handled or cleared if necessary.
            if model_name == "mock" and not agent_context.cli_args.responses_file:
                # This is a tricky situation. The original responses_file might have been
                # set via command line. If the user switches to mock via slash command
                # without specifying a new one, what should happen?
                # For now, we'll assume it's an issue if not set.
                # A more robust solution might involve prompting or having a default.
                return f"Warning: Model set to 'mock', but --responses-file is not currently set. The agent may fail if a task is run."
            elif model_name != "mock" and agent_context.cli_args.model == "mock":
                # If switching away from mock, perhaps nullify responses_file if it was only for mock?
                # This depends on desired behavior. For now, just update model.
                pass

            return f"Model set to: {model_name}"
        else:
            # This message indicates that the command was called without the necessary context
            # to actually change the model. This helps in debugging integration.
            return f"ModelCommand: Would set model to '{model_name}' (agent_context not fully available for update)."

class SetTimeoutCommand(SlashCommand):
    """
    Sets the timeout for LLM API calls.
    Example: /set-timeout 60
    """
    @property
    def name(self) -> str:
        return "set-timeout"

    @property
    def description(self) -> str:
        return "Sets the timeout in seconds for LLM API calls."

    @property
    def usage_examples(self) -> List[str]:
        return ["/set-timeout 60", "/set-timeout 120.5"]

    def execute(self, args: List[str], agent_context: Any = None) -> Optional[str]:
        if not args:
            return "Error: Missing timeout value. Usage: /set-timeout <seconds>"

        try:
            timeout_seconds = float(args[0])
            if timeout_seconds <= 0:
                return "Error: Timeout must be a positive number."
        except ValueError:
            return "Error: Invalid timeout value. Must be a number."

        if agent_context and hasattr(agent_context, 'cli_args'):
            agent_context.cli_args.llm_timeout = timeout_seconds
            return f"LLM timeout set to: {timeout_seconds} seconds."
        else:
            return f"SetTimeoutCommand: Would set timeout to {timeout_seconds}s (agent_context not fully available for update)."

class PlanModeCommand(SlashCommand):
    """
    Activates PLAN MODE for the agent.
    (Functionality to be fully implemented as part of MVP Task 3.1)
    """
    @property
    def name(self) -> str:
        return "plan"

    @property
    def description(self) -> str:
        return "Activates PLAN MODE for the agent, where it focuses on planning."

    @property
    def usage_examples(self) -> List[str]:
        return ["/plan"]

    def execute(self, args: List[str], agent_context: Any = None) -> Optional[str]:
        # Placeholder for actual mode switching logic
        # This will likely involve setting a flag in agent_context or DeveloperAgent
        if agent_context and hasattr(agent_context, 'set_mode'):
            agent_context.set_mode("PLAN") # Assuming a method to set mode
            return "Agent switched to PLAN MODE."
        else:
            # MVP Task 3.1: Implement ACT MODE vs. PLAN MODE Logic
            return "PlanModeCommand: PLAN MODE activated (actual mode switching pending full implementation)."

class ActModeCommand(SlashCommand):
    """
    Activates ACT MODE for the agent.
    (Functionality to be fully implemented as part of MVP Task 3.1)
    """
    @property
    def name(self) -> str:
        return "act"

    @property
    def description(self) -> str:
        return "Activates ACT MODE for the agent, where it focuses on executing tasks."

    @property
    def usage_examples(self) -> List[str]:
        return ["/act"]

    def execute(self, args: List[str], agent_context: Any = None) -> Optional[str]:
        # Placeholder for actual mode switching logic
        if agent_context and hasattr(agent_context, 'set_mode'):
            agent_context.set_mode("ACT") # Assuming a method to set mode
            return "Agent switched to ACT MODE."
        else:
            # MVP Task 3.1: Implement ACT MODE vs. PLAN MODE Logic
            return "ActModeCommand: ACT MODE activated (actual mode switching pending full implementation)."

# Definition for the AgentCliContext class (can be refined later)
# This will be instantiated in cli.py and passed to commands.
class AgentCliContext:
    def __init__(self, cli_args_namespace: Any, display_update_func: callable, agent_instance: Any = None):
        self.cli_args = cli_args_namespace  # This would be the args object from argparse
        self.display_update_func = display_update_func # For commands that might want to directly update UI
        self.agent = agent_instance # Optional, if commands need to interact with agent directly
        self.mode = "ACT" # Default mode

    def set_mode(self, mode_name: str):
        # This is a placeholder for where mode switching logic would go.
        # It might involve updating self.cli_args, self.agent, or other state.
        self.mode = mode_name
        self.display_update_func(f"Context: Mode changed to {mode_name}")


class SlashCommandRegistry:
    """
    Manages the registration and execution of slash commands.
    """
    def __init__(self):
        self._commands: Dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        """
        Registers a slash command instance.

        Args:
            command: An instance of a class derived from SlashCommand.

        Raises:
            ValueError: If a command with the same name is already registered.
        """
        if command.name in self._commands:
            raise ValueError(f"Command '{command.name}' is already registered.")
        if not command.name:
            raise ValueError("Command name cannot be empty.")
        if ' ' in command.name:
            raise ValueError("Command name cannot contain spaces.")

        self._commands[command.name] = command
        # print(f"DEBUG: Registered command /{command.name}") # For debugging

    def get_command(self, name: str) -> Optional[SlashCommand]:
        """
        Retrieves a registered command by its name.

        Args:
            name: The name of the command.

        Returns:
            The SlashCommand instance if found, otherwise None.
        """
        return self._commands.get(name)

    def execute_command(self, name: str, args: List[str], agent_context: Any = None) -> Optional[str]:
        """
        Executes a registered command by its name.

        Args:
            name: The name of the command to execute.
            args: A list of arguments for the command.
            agent_context: The context to pass to the command's execute method.

        Returns:
            The result message from the command's execution, or an error message
            if the command is not found or an execution error occurs.
        """
        command = self.get_command(name)
        if not command:
            return f"Error: Unknown command '/{name}'. Type /help for available commands." # Future: /help command

        try:
            return command.execute(args, agent_context)
        except Exception as e:
            # Log the full error for debugging
            # import traceback
            # print(f"""Error executing command /{name}: {e}""")
            # print(f"""{traceback.format_exc()}""") # For debugging
            return f"Error executing command '/{name}': {e}"

    def get_all_commands(self) -> List[SlashCommand]:
        """Returns a list of all registered command objects."""
        return list(self._commands.values())


class HelpCommand(SlashCommand):
    """
    Displays help information for available slash commands.
    """
    def __init__(self, registry: SlashCommandRegistry): # HelpCommand takes registry in constructor
        self._registry = registry

    @property
    def name(self) -> str:
        return "help"

    @property
    def description(self) -> str:
        return "Displays help information for all available slash commands."

    @property
    def usage_examples(self) -> List[str]:
        return ["/help"]

    def execute(self, args: List[str], agent_context: Any = None) -> Optional[str]:
        commands = self._registry.get_all_commands()
        if not commands:
            return "No commands available."

        # Sort commands alphabetically by name
        commands.sort(key=lambda cmd: cmd.name)

        help_lines = ["Available commands:\n"]
        for command in commands:
            help_lines.append(f"  /{command.name}\n")
            help_lines.append(f"    Description: {command.description}\n")
            if command.usage_examples:
                help_lines.append(f"    Examples:\n")
                for example in command.usage_examples:
                    help_lines.append(f"      {example}\n")
        return "".join(help_lines)


class UndoCommand(SlashCommand):
    """
    Reverts file changes based on auto-commits made during the session.
    """
    @property
    def name(self) -> str:
        return "undo"

    @property
    def description(self) -> str:
        return "Reverts file changes. No args: reverts last auto-commit. With commit ID: reverts to state *before* that commit."

    @property
    def usage_examples(self) -> List[str]:
        return ["/undo", "/undo <commit_id>"]

    def execute(self, args: List[str], agent_context: Any = None) -> Optional[str]:
        if not agent_context or not hasattr(agent_context, 'agent') or not agent_context.agent:
            return "Error: Agent context not available or not initialized correctly."

        # Using Any for agent type hint as per instructions, ideally this would be DeveloperAgent
        agent = agent_context.agent
        if not hasattr(agent, 'session_commit_history') or not hasattr(agent, 'cwd'):
            return "Error: Agent instance is not correctly configured (missing session_commit_history or cwd)."

        if len(args) == 0:
            if not agent.session_commit_history:
                return "No auto-commits in the current session to undo."

            last_commit_hash = agent.session_commit_history[-1]
            success = utils.revert_to_state_before_commit(agent.cwd, last_commit_hash)

            if success:
                agent.session_commit_history.pop()
                return f"Successfully reverted changes from the last auto-commit ({last_commit_hash[:7]})."
            else:
                return f"Error: Failed to revert changes from commit {last_commit_hash[:7]}."

        elif len(args) == 1:
            commit_id_to_undo = args[0]
            # Optional: Basic validation for commit_id_to_undo
            if not all(c in "0123456789abcdefABCDEF" for c in commit_id_to_undo) or not (4 <= len(commit_id_to_undo) <= 40):
                return f"Error: Invalid commit ID format: {commit_id_to_undo}."

            success = utils.revert_to_state_before_commit(agent.cwd, commit_id_to_undo)

            if success:
                try:
                    # Attempt to find and remove the commit and subsequent ones from history
                    found_index = -1
                    for i, chash in enumerate(agent.session_commit_history):
                        if chash.startswith(commit_id_to_undo):
                            found_index = i
                            break

                    if found_index != -1:
                        # Make sure we are removing the correct one if short hash was used
                        # For safety, could compare with full hash if available from a utils.get_full_hash()
                        # For now, assume prefix match is sufficient if found in session_commit_history
                        removed_commits_count = len(agent.session_commit_history) - found_index
                        del agent.session_commit_history[found_index:]
                        return f"Successfully reverted to the state before commit {commit_id_to_undo[:7]}. {removed_commits_count} commit(s) removed from session history."
                    else:
                        # If not found in session history, it might be an older commit or one not tracked by this session.
                        # The git operation still succeeded.
                        return f"Successfully reverted to the state before commit {commit_id_to_undo[:7]}. (Commit not found in current session history or history was already diverged)"
                except ValueError:
                    # Should not happen if using enumerate, but as a fallback.
                    return f"Successfully reverted to the state before commit {commit_id_to_undo[:7]}. (Commit not found in current session history)"
            else:
                return f"Error: Failed to revert to the state before commit {commit_id_to_undo[:7]}."
        else:
            return "Error: /undo accepts 0 or 1 argument (commit_id)."


class UndoAllCommand(SlashCommand):
    """
    Reverts all auto-committed changes made during the current CLI agent session.
    """
    @property
    def name(self) -> str:
        return "undo-all"

    @property
    def description(self) -> str:
        return "Reverts all auto-committed changes made during the current CLI agent session."

    @property
    def usage_examples(self) -> List[str]:
        return ["/undo-all"]

    def execute(self, args: List[str], agent_context: Any = None) -> Optional[str]:
        if not agent_context or not hasattr(agent_context, 'agent') or not agent_context.agent:
            return "Error: Agent context not available or not initialized correctly."

        agent = agent_context.agent
        if not hasattr(agent, 'session_commit_history') or \
           not hasattr(agent, 'initial_session_head_commit_hash') or \
           not hasattr(agent, 'cwd'):
            return "Error: Agent instance is not correctly configured."

        if not agent.session_commit_history:
            # Check if already at initial state, perhaps.
            # For now, if no session commits, nothing to "undo-all" from session perspective.
            # However, user might have made manual commits. This command is about session's auto-commits.
            try:
                current_head = utils.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=agent.cwd, text=True).strip()
                if agent.initial_session_head_commit_hash and current_head == agent.initial_session_head_commit_hash:
                    return "No auto-commits in the current session to undo. Repository is already at the initial session state."
            except Exception: # Ignore if git check fails, proceed to standard message
                pass
            return "No auto-commits recorded in the current session to undo."

        initial_hash = agent.initial_session_head_commit_hash
        if initial_hash is None:
            return "Error: Cannot determine the initial state of the repository for this session (e.g., not a git repo, or failed to get initial HEAD)."

        success = utils.revert_to_commit(agent.cwd, initial_hash)

        if success:
            count_before_clear = len(agent.session_commit_history)
            agent.session_commit_history.clear()
            return f"Successfully reverted all {count_before_clear} auto-committed changes made during this session to state {initial_hash[:7]}."
        else:
            # If initial_hash was a specific commit, and it's gone (e.g. rebase), this might fail.
            # Or if repo became dirty.
            return f"Error: Failed to revert all session changes to the initial session state {initial_hash[:7]}."

# Registration of commands (assuming a global registry instance as per common pattern)
# This part might be in cli.py or another central place.
# If COMMAND_REGISTRY is defined and used in this file, new commands would be added:
# COMMAND_REGISTRY.register(UndoCommand())
# COMMAND_REGISTRY.register(UndoAllCommand())
# Since the file read didn't show this, I'm only defining the classes as per primary task.
# The subtask states: "actual registration might be in cli.py later".
