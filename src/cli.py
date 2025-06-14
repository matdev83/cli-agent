from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Callable
import traceback  # For detailed error logging in the UI
import threading  # For stop_agent_event

import os
from functools import partial

from io import StringIO  # For capturing stdout

# prompt_toolkit imports
from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.widgets import TextArea, SearchToolbar
from prompt_toolkit.styles import Style

from src.agent import DeveloperAgent  # Changed to absolute import
from src.llm import MockLLM, OpenRouterLLM  # Changed to absolute import
from src.llm_protocol import LLMResponse  # Changed to absolute import
from src.slash_commands import (  # Changed to absolute import
    SlashCommandRegistry,
    ModelCommand,
    SetTimeoutCommand,
    PlanModeCommand,
    ActModeCommand,
    HelpCommand,
    RefreshCommand,  # AgentCliContext is defined below
    UndoCommand,
    UndoAllCommand,  # Added new commands
)
from src.file_cache import FileCache  # Changed to absolute import
from src.autocompletion import AtMentionCompleter  # For autocomplete


# Define AgentCliContext class at the module level so it can be imported
class AgentCliContext:
    def __init__(
        self,
        cli_args_namespace: argparse.Namespace,
        display_update_func: Callable[[str], None],
        file_cache_instance: Optional[FileCache],
        agent_instance: Optional[DeveloperAgent] = None,
    ):
        self.cli_args = cli_args_namespace
        self.display_update_func = display_update_func
        self.agent = agent_instance
        self.file_cache = file_cache_instance
        self.mode = "ACT"  # Default mode

    def set_mode(self, mode_name: str):
        self.mode = mode_name
        self.display_update_func(f"Context: Mode changed to {mode_name.upper()}")

    # Method to update the agent instance if needed, e.g., after re-creation
    def set_agent_instance(self, agent_instance: Optional[DeveloperAgent]):
        self.agent = agent_instance


# Global list to store display messages for FormattedTextControl
# Global list to store display messages for FormattedTextControl
display_text_fragments = []
stop_agent_event = threading.Event()


# This function is called from various threads (agent, logging) to update the UI
def update_display_text_safely(text_line: str, app_ref: Application, dc_ref: FormattedTextControl):
    """
    Safely updates the display_text_fragments and the FormattedTextControl.
    To be called via app_ref.loop.call_soon_threadsafe if called from a non-main thread.
    """
    display_text_fragments.append(text_line + "\n")
    new_text = "".join(display_text_fragments)
    # Ensure the actual update to the prompt_toolkit control happens in the main event loop
    app_ref.loop.call_soon_threadsafe(setattr, dc_ref, "text", new_text)


class UIUpdateLogHandler(logging.Handler):
    def __init__(self, app_ref: Application, dc_ref: FormattedTextControl):
        super().__init__()
        self.app_ref = app_ref
        self.dc_ref = dc_ref
        self.setFormatter(logging.Formatter("[LOG] %(message)s"))  # Simple format for UI logs

    def emit(self, record):
        log_entry = self.format(record)
        # The update_display_text_safely function handles call_soon_threadsafe
        update_display_text_safely(log_entry, self.app_ref, self.dc_ref)


def setup_logging(
    log_file: str = "agent.log",
    app_for_ui_logging: Optional[Application] = None,
    dc_for_ui_logging: Optional[FormattedTextControl] = None,
) -> None:
    handlers_list = []

    # File Handler (always add)
    file_handler = logging.FileHandler(log_file, mode="a")  # Append mode
    file_handler.setLevel(logging.INFO)  # Set level for this handler
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    handlers_list.append(file_handler)

    # UI Handler (if app and display control are provided)
    if app_for_ui_logging and dc_for_ui_logging:
        ui_handler = UIUpdateLogHandler(app_for_ui_logging, dc_for_ui_logging)
        ui_handler.setLevel(logging.INFO)  # Set level for UI logs
        handlers_list.append(ui_handler)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,  # Overall minimum level; handlers can be more restrictive
        handlers=handlers_list,
    )
    # If basicConfig was already called, this might not reconfigure root logger as expected.
    # It's safer to get the root logger and manipulate its handlers.
    # For simplicity here, assuming this is the first and only call to basicConfig,
    # or that it successfully replaces handlers.


def run_agent_and_update_display(
    task: str,
    current_cli_args: argparse.Namespace,
    app_ref: Application,
    dc_ref: FormattedTextControl,
    stop_event: threading.Event,
):
    """
    Runs the agent in a separate thread and updates the display control.
    """

    def _cli_llm_response_callback(response: LLMResponse, model_name: str, session_cost: float):
        if response.usage:
            prompt_tokens_str = (
                str(response.usage.prompt_tokens)
                if response.usage.prompt_tokens is not None
                else "N/A"
            )
            completion_tokens_str = (
                str(response.usage.completion_tokens)
                if response.usage.completion_tokens is not None
                else "N/A"
            )
            cost_str = f"${response.usage.cost:.6f}" if response.usage.cost is not None else "N/A"

            stats_str = (
                f"STATS: Model: {model_name}, "
                f"Prompt: {prompt_tokens_str}, "
                f"Completion: {completion_tokens_str}, "
                f"Cost: {cost_str}, "
                f"Session Cost: ${session_cost:.6f}"
            )
        else:
            stats_str = (
                f"STATS: Model: {model_name}, "
                f"Prompt: N/A, "
                f"Completion: N/A, "
                f"Cost: N/A, "
                f"Session Cost: ${session_cost:.6f}"
            )
        update_display_text_safely(stats_str, app_ref, dc_ref)

    def _update_ui(message: str):
        # Ensures UI updates happen on the main event loop
        update_display_text_safely(message, app_ref, dc_ref)

    # _update_ui(f"Task submitted: {task}") # Removed, as accept_input_handler already prints "New task received"

    if stop_event.is_set():
        _update_ui("Task execution cancelled: Stop event was active before starting.")
        return

    original_stdout = sys.stdout
    captured_stdout = None
    try:
        captured_stdout = StringIO()
        sys.stdout = captured_stdout

        # The actual agent execution
        # Note: run_agent itself is synchronous / blocking.
        # The stop_event is checked before, and its effect after (if it completed).
        # A true cooperative stop would require run_agent to take stop_event.
        result = run_agent(
            task=task,
            cli_args=current_cli_args,  # Pass the potentially modified cli_args
            on_llm_response_callback=_cli_llm_response_callback,  # Pass the callback
        )

        sys.stdout = original_stdout  # Restore stdout
        agent_stdout_output = captured_stdout.getvalue()

        if stop_event.is_set():
            _update_ui("Agent task stopped by user during execution.")
            if agent_stdout_output:
                _update_ui(f"Partial STDOUT:\n{agent_stdout_output}")
        else:
            if agent_stdout_output:
                _update_ui(f"Agent STDOUT:\n{agent_stdout_output}")
            _update_ui(f"Agent Result:\n{result}")
            logging.info("Agent task completed successfully.")  # Goes to all handlers

    except Exception as e:
        if captured_stdout:  # Ensure stdout is restored if it was changed
            sys.stdout = original_stdout
        logging.error(f"Agent task failed: {e}", exc_info=True)  # Goes to all handlers
        # Also send specific error to UI directly if logging handler for UI is restrictive
        _update_ui(f"Error during agent execution: {e}\n{traceback.format_exc()}")
    finally:
        if captured_stdout:  # Ensure stdout is restored in all cases
            sys.stdout = original_stdout

        final_message = "Task processing finished."
        if stop_event.is_set():
            final_message = "Task processing stopped due to user signal."
        _update_ui(final_message)


def run_agent(
    task: str,
    responses_file: str | None = None,
    *,
    # auto_approve: bool = False, # From cli_args
    # ... other direct approval flags removed as they are in cli_args
    # cwd: str = ".", # No longer a direct param, will come from cli_args
    # model: str = "mock", # Will come from cli_args
    return_history: bool = False,
    # llm_timeout: Optional[float] = None, # Will come from cli_args
    # matching_strictness: int = 100, # From cli_args
    cli_args: Optional[argparse.Namespace] = None,
    on_llm_response_callback: Optional[
        Callable[[LLMResponse, str, float], None]
    ] = None,  # New parameter
) -> str | tuple[str, list[dict[str, str]]]:
    if cli_args is None:
        # This case should ideally not happen if called from CLI with context
        # Create a default if it does, for programmatic calls.
        # This needs to be comprehensive or raise an error.
        # For now, let's assume cli_args is always provided from main()
        raise ValueError("cli_args must be provided to run_agent")

    # Extract necessary settings from cli_args
    current_model_name = cli_args.model
    current_llm_timeout = cli_args.llm_timeout
    current_cwd = cli_args.cwd
    if on_llm_response_callback is None:

        def on_llm_response_callback(
            resp: LLMResponse, model_name: str, session_cost: float
        ) -> None:
            if resp.usage:
                prompt_tokens_str = (
                    str(resp.usage.prompt_tokens) if resp.usage.prompt_tokens is not None else "N/A"
                )
                completion_tokens_str = (
                    str(resp.usage.completion_tokens)
                    if resp.usage.completion_tokens is not None
                    else "N/A"
                )
                cost_str = f"${resp.usage.cost:.6f}" if resp.usage.cost is not None else "N/A"
                stats_str = (
                    f"STATS: Model: {model_name}, "
                    f"Prompt: {prompt_tokens_str}, "
                    f"Completion: {completion_tokens_str}, "
                    f"Cost: {cost_str}, "
                    f"Session Cost: ${session_cost:.6f}"
                )
            else:
                stats_str = (
                    f"STATS: Model: {model_name}, "
                    f"Prompt: N/A, Completion: N/A, Cost: N/A, Session Cost: ${session_cost:.6f}"
                )
            try:
                update_display_text_safely(stats_str, None, None)
            except Exception:
                print(stats_str)

    if responses_file is not None:
        cli_args.responses_file = responses_file

    # ... (rest of the function)

    if current_model_name == "mock":
        if not cli_args.responses_file:  # Use cli_args here
            raise ValueError("responses_file is required for mock model (via cli_args)")
        llm = MockLLM.from_file(cli_args.responses_file)
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            # Raising RuntimeError here, which will be caught by the generic Exception in main.
            # Could be a custom error or handled more specifically if desired.
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable not set, required for non-mock models."
            )
        llm = OpenRouterLLM(model=current_model_name, api_key=api_key, timeout=current_llm_timeout)

    # ...
    # The LLMWrapper protocol expects send_message to take temperature and max_tokens.
    # The DeveloperAgent's __init__ expects a send_message callable that matches
    # `Callable[[List[Dict[str, str]]], str]`.
    # The current LLM implementations `send_message` methods were updated to `Optional[str]`
    # and to accept optional temp/tokens.
    # DeveloperAgent needs to be updated to pass these if it's to use the full protocol,
    # or the LLM send_message methods need to align with what DeveloperAgent expects.
    # For now, assuming DeveloperAgent.send_message call signature is what llm.send_message provides.
    # The current DeveloperAgent.send_message(self.history) matches the basic signature.
    agent = DeveloperAgent(
        llm.send_message,
        cwd=current_cwd,  # Use cwd from cli_args
        cli_args=cli_args,  # Pass the whole namespace
        on_llm_response_callback=on_llm_response_callback,  # Pass callback to agent
        # matching_strictness and disable_git_auto_commits are already correctly sourced from cli_args if present
        # in the original code, so no change needed there.
    )
    result = agent.run_task(task)
    if return_history:
        return result, list(agent.history)
    return result


def main(
    argv: Optional[List[str]] = None,
) -> int:  # Changed return to int, was Coroutine[Any, Any, int]
    parser = argparse.ArgumentParser(description="CLI developer agent")
    parser.add_argument(  # Model arg
        "--model",
        default="mock",  # Default to mock
        help="Specify the LLM to use. 'mock' uses MockLLM (requires --responses-file). "
        "Other values (e.g., 'anthropic/claude-3-opus') use OpenRouterLLM (requires OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--responses-file",
        help="Path to JSON file with mock LLM responses (required for 'mock' model).",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto approve commands (legacy, granular flags take precedence)",
    )
    parser.add_argument("--cwd", default=".", help="Working directory")
    parser.add_argument(
        "--allow-read-files",
        action="store_true",
        default=False,
        help="Automatically approve file reads (includes listing files/directories).",
    )
    parser.add_argument(
        "--allow-edit-files",
        action="store_true",
        default=False,
        help="Automatically approve file edits (includes creating new files).",
    )
    parser.add_argument(
        "--allow-execute-safe-commands",
        action="store_true",
        default=False,
        help="Automatically approve commands marked as 'safe' by the LLM.",
    )
    parser.add_argument(
        "--allow-execute-all-commands",
        action="store_true",
        default=False,
        help="Automatically approve ALL commands, including those not marked as 'safe'. Implies --allow-execute-safe-commands.",
    )
    parser.add_argument(
        "--allow-use-browser",
        action="store_true",
        default=False,
        help="Automatically approve browser usage.",
    )
    parser.add_argument(
        "--allow-use-mcp",
        action="store_true",
        default=False,
        help="Automatically approve MCP (Multi-Capability Plugin) usage.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for LLM API calls.",
    )
    parser.add_argument(
        "--matching-strictness",
        type=int,
        default=100,
        help="Set the string matching strictness for file edits (0-100, 100 is exact match).",
    )
    parser.add_argument(
        "--disable-git-auto-commits",
        action="store_true",
        default=False,
        help="Disable automatic git commits after file modifications.",
    )
    args = parser.parse_args(argv)

    # --- Slash Command Setup ---
    slash_command_registry = SlashCommandRegistry()
    # The agent_context will be more fully fleshed out as needed.
    # For now, it needs cli_args and a way to update the UI.
    # We'll define _ui_update_func later, so use a placeholder for now or None.
    # This will be properly initialized after app and display_control exist.
    agent_cli_context = None  # Will be initialized later

    # --- FileCache Initialization ---
    # Instantiate FileCache early, using args.cwd
    # It will perform an initial scan.
    # Error handling for FileCache instantiation (e.g., if cwd is invalid) can be added if necessary.
    try:
        file_cache = FileCache(root_dir=args.cwd, initial_scan=True)
        logging.info(
            f"FileCache initialized. Found {len(file_cache.get_paths())} files in initial scan of {args.cwd}."
        )
    except Exception as e:
        logging.error(
            f"Failed to initialize FileCache with root_dir {args.cwd}: {e}", exc_info=True
        )
        # Depending on severity, might want to exit or continue without cache functionality.
        # For now, log and continue; slash command using it should handle its absence if file_cache is None.
        file_cache = None  # Ensure file_cache is defined even if init fails.

    # Register commands
    try:
        slash_command_registry.register(ModelCommand())
        slash_command_registry.register(SetTimeoutCommand())
        slash_command_registry.register(PlanModeCommand())
        slash_command_registry.register(ActModeCommand())
        slash_command_registry.register(UndoCommand())  # Register UndoCommand
        slash_command_registry.register(UndoAllCommand())  # Register UndoAllCommand
        # Register HelpCommand
        help_command = HelpCommand(slash_command_registry)
        slash_command_registry.register(help_command)
        # RefreshCommand will be registered after AgentCliContext is fully defined
        # and can hold the file_cache instance.

    except ValueError as e:
        # This is early in startup, so print to stderr and exit.
        print(f"Error registering slash commands: {e}", file=sys.stderr)
        sys.exit(1)

    if not (0 <= args.matching_strictness <= 100):
        # This will call sys.exit, so the app won't start.
        parser.error("--matching-strictness must be between 0 and 100.")

    if args.model == "mock" and not args.responses_file:
        parser.error("--responses-file is required when using the mock model.")

    # --- prompt_toolkit UI Setup ---
    display_text_fragments.append(
        "Welcome to the CLI Agent. Type your task below and press Enter.\nUse Ctrl-C or Ctrl-D to exit.\n"
    )
    display_control = FormattedTextControl(
        text="".join(display_text_fragments), focusable=False, focus_on_click=False
    )
    display_window = Window(
        content=display_control,
        wrap_lines=True,
        # Make it scrollable from the bottom if content overflows
        # scroll_offsets=ScrollOffsets(bottom=0, top=0), # Might need custom scroll handler
        dont_extend_height=False,
    )

    input_field = TextArea(
        height=3,  # Allow multi-line input, but keep it modest
        prompt="Task: ",
        multiline=True,  # Use True for Enter key to submit, False for Shift+Enter for newline
        wrap_lines=True,
        search_field=SearchToolbar(),  # Optional: for multi-line search
    )

    # --- prompt_toolkit UI Setup ---
    # Initial message for the display
    display_text_fragments.append(
        "Welcome to the CLI Agent. Type your task, or /stop, /exit.\nCtrl-C or Ctrl-D also exits.\n"
    )
    display_control = FormattedTextControl(
        text="".join(display_text_fragments), focusable=False, focus_on_click=False
    )
    display_window = Window(content=display_control, wrap_lines=True)

    input_field = TextArea(
        height=3,
        prompt="Task / Command: ",
        multiline=True,
        wrap_lines=True,
        # search_field=SearchToolbar(), # Optional
    )

    # Main container for UI
    container = HSplit(
        [
            display_window,
            Window(height=1, char="-", style="class:line"),
            input_field,
        ]
    )

    # KeyBindings: Ctrl-C/Ctrl-D for exit
    kb = KeyBindings()

    @kb.add("c-c", eager=True)
    @kb.add("c-d", eager=True)
    def _(event):
        update_display_text_safely(
            "Exit requested via Ctrl-C/D. Signaling agent to stop...", event.app, display_control
        )
        stop_agent_event.set()
        event.app.exit(result=0)

    # Application instance
    app = Application(
        layout=Layout(container, focused_element=input_field),
        key_bindings=kb,
        full_screen=True,
        style=Style.from_dict(
            {"line": "#888888", "textarea": "bg:#1e1e1e #ansidefault"}
        ),  # Basic styling
        mouse_support=True,
    )

    # Now that `app` and `display_control` are created, setup logging with UI handler
    setup_logging(
        log_file="agent_cli.log", app_for_ui_logging=app, dc_for_ui_logging=display_control
    )
    logging.info("CLI Agent UI Initialized. Model: %s", args.model)  # This will go to file and UI

    # Properly initialize agent_cli_context now that app and display_control are available
    # and _ui_update_func can be created.
    # We need to pass the _ui_update_func to the context as well.
    # The _ui_update_func here is the one defined inside accept_input_handler,
    # which might be problematic if commands need to update UI outside of handler response.
    # Let's make a more general one.

    # Create a general UI update function for the context
    # This is important for commands that might want to give feedback directly.
    context_ui_update_func = partial(
        update_display_text_safely, app_ref=app, dc_ref=display_control
    )

    agent_cli_context = AgentCliContext(  # AgentCliContext class definition moved to top level
        cli_args_namespace=args,
        display_update_func=context_ui_update_func,
        file_cache_instance=file_cache,  # Pass the initialized file_cache
    )
    # Now that agent_cli_context is created with file_cache, other commands needing it can be registered
    try:
        # Register RefreshCommand now that agent_cli_context (and its file_cache) is available
        refresh_cmd = RefreshCommand()
        slash_command_registry.register(refresh_cmd)
        logging.info("RefreshCommand registered.")
    except ValueError as e:  # Catch potential registration errors for this specific command
        logging.error(f"Error registering RefreshCommand: {e}", exc_info=True)
        # Optionally, inform the user via UI if this is critical
        context_ui_update_func(f"Error: Could not register RefreshCommand: {e}")

    # --- Autocompleter Setup ---
    # Instantiate the completer with the file_cache from the context
    # Ensure file_cache in agent_cli_context is not None, or handle it in AtMentionCompleter
    if agent_cli_context.file_cache:
        at_mention_completer = AtMentionCompleter(file_cache=agent_cli_context.file_cache)
        input_field.completer = at_mention_completer
        logging.info("AtMentionCompleter initialized and attached to input field.")
    else:
        logging.warning(
            "FileCache not available in AgentCliContext. AtMentionCompleter not attached."
        )

    # Input handler for the TextArea
    def accept_input_handler():  # Removed unused 'buff: Buffer'
        command_or_task_full = input_field.text.strip()
        input_field.text = ""  # Clear input field

        # Use the context_ui_update_func for general UI updates from this handler
        _ui_update_func = agent_cli_context.display_update_func

        if not command_or_task_full:
            _ui_update_func("No task or command entered.")
            return True  # Keep the buffer (though text is cleared)

        if command_or_task_full.startswith("/"):
            parts = command_or_task_full[1:].split()
            command_name = parts[0]
            command_args = parts[1:]

            if command_name == "exit":  # Keep direct handling for /exit
                _ui_update_func("User command: /exit. Application will close.")
                stop_agent_event.set()  # Signal any running agent
                app.exit(result=0)
                return True

            if command_name == "stop":  # Keep direct handling for /stop
                if not stop_agent_event.is_set():
                    _ui_update_func("User command: /stop. Signaling current agent task to stop.")
                    stop_agent_event.set()
                else:
                    _ui_update_func("Stop signal already active.")
                return True

            # Process other slash commands through the registry
            # Pass the agent_cli_context to the command
            result_message = slash_command_registry.execute_command(
                command_name, command_args, agent_cli_context
            )
            if result_message:
                _ui_update_func(f"{result_message}")

            # After executing a slash command, we typically don't start an agent task.
            # Refresh the prompt or wait for next input.
            # If a command needs to trigger an agent run, it should do so explicitly
            # or set up state that the next (empty) input submission triggers.

        elif command_or_task_full:  # It's a task for the agent
            _ui_update_func(f"New task received: {command_or_task_full}")
            stop_agent_event.clear()  # Clear event for the new task

            # Ensure agent_cli_context.agent is updated if a new agent is created,
            # or ensure the agent uses the updated cli_args from agent_cli_context.
            # For now, run_agent_and_update_display uses the global 'args'.
            # This needs to be harmonized: run_agent should probably get its config
            # from agent_cli_context.cli_args.

            app.loop.call_soon_threadsafe(
                app.call_from_executor,  # Runs the function in a separate thread
                run_agent_and_update_display,  # The function to run
                command_or_task_full,  # task
                agent_cli_context.cli_args,  # Pass potentially modified args
                app,  # app_ref for UI updates from thread
                display_control,  # dc_ref for UI updates from thread
                stop_agent_event,  # stop_event
            )
        return True  # Keep the buffer (text was cleared)

    input_field.accept_handler = accept_input_handler

    # Run the application
    exit_code = app.run()

    logging.info(f"Application exited with code {exit_code}.")  # Goes to file log
    return exit_code if isinstance(exit_code, int) else 0


if __name__ == "__main__":
    # Standard main entry point
    try:
        return_code = main()
        sys.exit(return_code)
    except Exception as e:
        # Fallback for errors not caught within main's try/except for app.run()
        print(f"Critical unhandled exception: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

# if __name__ == "__main__": block is already present above and correctly calls main()
# The SyntaxError was due to code outside any function, specifically the return statement.
# The lines above this comment (the second if __name__ == "__main__" and the preceding UI setup)
# were identified as incorrectly placed or duplicated from within the main() function.
# The main() function, as defined from line 278, correctly contains its UI setup and return.
# This deletion removes the misplaced global code that caused the SyntaxError.
