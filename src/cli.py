from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Callable
import traceback # For detailed error logging in the UI
import threading # For stop_agent_event

import os
from functools import partial

from io import StringIO # For capturing stdout
# prompt_toolkit imports
from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window, VSplit
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.widgets import TextArea, SearchToolbar, Label
from prompt_toolkit.document import Document
from prompt_toolkit.styles import Style

from .agent import DeveloperAgent
from .llm import MockLLM, OpenRouterLLM

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
    app_ref.loop.call_soon_threadsafe(setattr, dc_ref, 'text', new_text)


class UIUpdateLogHandler(logging.Handler):
    def __init__(self, app_ref: Application, dc_ref: FormattedTextControl):
        super().__init__()
        self.app_ref = app_ref
        self.dc_ref = dc_ref
        self.setFormatter(logging.Formatter("[LOG] %(message)s")) # Simple format for UI logs

    def emit(self, record):
        log_entry = self.format(record)
        # The update_display_text_safely function handles call_soon_threadsafe
        update_display_text_safely(log_entry, self.app_ref, self.dc_ref)


def setup_logging(
    log_file: str = "agent.log",
    app_for_ui_logging: Optional[Application] = None,
    dc_for_ui_logging: Optional[FormattedTextControl] = None
) -> None:
    handlers_list = []

    # File Handler (always add)
    file_handler = logging.FileHandler(log_file, mode='a') # Append mode
    file_handler.setLevel(logging.INFO) # Set level for this handler
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    handlers_list.append(file_handler)

    # UI Handler (if app and display control are provided)
    if app_for_ui_logging and dc_for_ui_logging:
        ui_handler = UIUpdateLogHandler(app_for_ui_logging, dc_for_ui_logging)
        ui_handler.setLevel(logging.INFO) # Set level for UI logs
        handlers_list.append(ui_handler)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO, # Overall minimum level; handlers can be more restrictive
        handlers=handlers_list,
    )
    # If basicConfig was already called, this might not reconfigure root logger as expected.
    # It's safer to get the root logger and manipulate its handlers.
    # For simplicity here, assuming this is the first and only call to basicConfig,
    # or that it successfully replaces handlers.


def run_agent_and_update_display(
    task: str,
    args: argparse.Namespace,
    app_ref: Application,
    dc_ref: FormattedTextControl,
    stop_event: threading.Event
):
    """
    Runs the agent in a separate thread and updates the display control.
    """

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
            responses_file=args.responses_file,
            auto_approve=args.auto_approve,
            allow_read_files=args.allow_read_files,
            allow_edit_files=args.allow_edit_files,
            allow_execute_safe_commands=args.allow_execute_safe_commands,
            allow_execute_all_commands=args.allow_execute_all_commands,
            allow_use_browser=args.allow_use_browser,
            allow_use_mcp=args.allow_use_mcp,
            disable_git_auto_commits=args.disable_git_auto_commits,
            cwd=args.cwd,
            model=args.model,
            llm_timeout=args.llm_timeout,
            matching_strictness=args.matching_strictness,
            cli_args=args
        )

        sys.stdout = original_stdout # Restore stdout
        agent_stdout_output = captured_stdout.getvalue()

        if stop_event.is_set():
            _update_ui("Agent task stopped by user during execution.")
            if agent_stdout_output:
                _update_ui(f"Partial STDOUT:\n{agent_stdout_output}")
        else:
            if agent_stdout_output:
                _update_ui(f"Agent STDOUT:\n{agent_stdout_output}")
            _update_ui(f"Agent Result:\n{result}")
            logging.info("Agent task completed successfully.") # Goes to all handlers

    except Exception as e:
        if captured_stdout: # Ensure stdout is restored if it was changed
             sys.stdout = original_stdout
        logging.error(f"Agent task failed: {e}", exc_info=True) # Goes to all handlers
        # Also send specific error to UI directly if logging handler for UI is restrictive
        _update_ui(f"Error during agent execution: {e}\n{traceback.format_exc()}")
    finally:
        if captured_stdout: # Ensure stdout is restored in all cases
             sys.stdout = original_stdout

        final_message = "Task processing finished."
        if stop_event.is_set():
            final_message = "Task processing stopped due to user signal."
        _update_ui(final_message)


def run_agent( # Keep the original run_agent signature and logic
    task: str,
    responses_file: str | None = None, # Keep this arg
    *,
    auto_approve: bool = False,
    allow_read_files: bool = False,
    allow_edit_files: bool = False,
    allow_execute_safe_commands: bool = False,
    allow_execute_all_commands: bool = False,
    allow_use_browser: bool = False,
    allow_use_mcp: bool = False,
    disable_git_auto_commits: bool = False,
    cwd: str = ".",
    model: str = "mock",
    return_history: bool = False,
    llm_timeout: Optional[float] = None,
    matching_strictness: int = 100, # Kept as it's not purely an approval flag
    cli_args: Optional[argparse.Namespace] = None # Added cli_args
    # allow_read_files: bool = False, # Removed
    # allow_edit_files: bool = False, # Removed
    # allow_execute_safe_commands: bool = False, # Removed
    # allow_execute_all_commands: bool = False, # Removed
    # allow_use_browser: bool = False, # Removed
    # allow_use_mcp: bool = False, # Removed
) -> str | tuple[str, list[dict[str, str]]]:
    if cli_args is None:  # Provide default if not passed, though main() should always pass it.
        cli_args = argparse.Namespace(
            auto_approve=auto_approve,
            allow_read_files=allow_read_files,
            allow_edit_files=allow_edit_files,
            allow_execute_safe_commands=allow_execute_safe_commands,
            allow_execute_all_commands=allow_execute_all_commands,
            allow_use_browser=allow_use_browser,
            allow_use_mcp=allow_use_mcp,
        )
    else:
        cli_args.auto_approve = auto_approve
        cli_args.allow_read_files = allow_read_files
        cli_args.allow_edit_files = allow_edit_files
        cli_args.allow_execute_safe_commands = allow_execute_safe_commands
        cli_args.allow_execute_all_commands = allow_execute_all_commands
        cli_args.allow_use_browser = allow_use_browser
        cli_args.allow_use_mcp = allow_use_mcp
        cli_args.disable_git_auto_commits = disable_git_auto_commits

    if model == "mock":
        if not responses_file:
            raise ValueError("responses_file is required for mock model")
        # Ensure responses_file exists if model is mock (FileNotFoundError will be caught by main)
        llm = MockLLM.from_file(responses_file)
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            # Raising RuntimeError here, which will be caught by the generic Exception in main.
            # Could be a custom error or handled more specifically if desired.
            raise RuntimeError("OPENROUTER_API_KEY environment variable not set, required for non-mock models.")
        llm = OpenRouterLLM(model=model, api_key=api_key, timeout=llm_timeout)

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
        cwd=cwd,
        cli_args=cli_args, # Pass the whole namespace
        matching_strictness=matching_strictness
        ,disable_git_auto_commits=disable_git_auto_commits
        # Removed individual approval flags, they are now in cli_args
    )
    result = agent.run_task(task)
    if return_history:
        return result, list(agent.history)
    return result


def main(argv: Optional[List[str]] = None) -> int: # Changed return to int, was Coroutine[Any, Any, int]
    parser = argparse.ArgumentParser(description="CLI developer agent")
    parser.add_argument( # Model arg
        "--model",
        default="mock", # Default to mock
        help="Specify the LLM to use. 'mock' uses MockLLM (requires --responses-file). "
             "Other values (e.g., 'anthropic/claude-3-opus') use OpenRouterLLM (requires OPENROUTER_API_KEY)."
    )
    parser.add_argument(
        "--responses-file",
        help="Path to JSON file with mock LLM responses (required for 'mock' model).",
    )
    parser.add_argument("--auto-approve", action="store_true", help="Auto approve commands (legacy, granular flags take precedence)")
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

    if not (0 <= args.matching_strictness <= 100):
        # This will call sys.exit, so the app won't start.
        parser.error("--matching-strictness must be between 0 and 100.")

    if args.model == "mock" and not args.responses_file:
        parser.error("--responses-file is required when using the mock model.")

    # --- prompt_toolkit UI Setup ---
    display_text_fragments.append("Welcome to the CLI Agent. Type your task below and press Enter.\nUse Ctrl-C or Ctrl-D to exit.\n")
    display_control = FormattedTextControl(text="".join(display_text_fragments), focusable=False, focus_on_click=False)
    display_window = Window(
        content=display_control,
        wrap_lines=True,
        # Make it scrollable from the bottom if content overflows
        # scroll_offsets=ScrollOffsets(bottom=0, top=0), # Might need custom scroll handler
        dont_extend_height=False
    )

    input_field = TextArea(
        height=3, # Allow multi-line input, but keep it modest
        prompt="Task: ",
        multiline=True, # Use True for Enter key to submit, False for Shift+Enter for newline
        wrap_lines=True,
        search_field=SearchToolbar(), # Optional: for multi-line search
    )

    # --- prompt_toolkit UI Setup ---
    # Initial message for the display
    display_text_fragments.append(
        "Welcome to the CLI Agent. Type your task, or /stop, /exit.\n"
        "Ctrl-C or Ctrl-D also exits.\n"
    )
    display_control = FormattedTextControl(
        text="".join(display_text_fragments),
        focusable=False,
        focus_on_click=False
    )
    display_window = Window(content=display_control, wrap_lines=True)

    input_field = TextArea(
        height=3, prompt="Task / Command: ", multiline=True, wrap_lines=True,
        # search_field=SearchToolbar(), # Optional
    )

    # Main container for UI
    container = HSplit([
        display_window,
        Window(height=1, char='-', style='class:line'),
        input_field,
    ])

    # KeyBindings: Ctrl-C/Ctrl-D for exit
    kb = KeyBindings()
    @kb.add('c-c', eager=True)
    @kb.add('c-d', eager=True)
    def _(event):
        update_display_text_safely("Exit requested via Ctrl-C/D. Signaling agent to stop...", event.app, display_control)
        stop_agent_event.set()
        event.app.exit(result=0)

    # Application instance
    app = Application(
        layout=Layout(container, focused_element=input_field),
        key_bindings=kb,
        full_screen=True,
        style=Style.from_dict({'line': '#888888', 'textarea': 'bg:#1e1e1e #ansidefault'}), # Basic styling
        mouse_support=True,
    )

    # Now that `app` and `display_control` are created, setup logging with UI handler
    setup_logging(
        log_file="agent_cli.log",
        app_for_ui_logging=app,
        dc_for_ui_logging=display_control
    )
    logging.info("CLI Agent UI Initialized. Model: %s", args.model) # This will go to file and UI

    # Input handler for the TextArea
    def accept_input_handler(buff: Buffer):
        command_or_task = input_field.text.strip()
        input_field.text = ""  # Clear input field

        _ui_update_func = partial(update_display_text_safely, app_ref=app, dc_ref=display_control)

        if command_or_task == "/exit":
            _ui_update_func("User command: /exit. Application will close.")
            stop_agent_event.set() # Signal any running agent
            app.exit(result=0)
            return True

        if command_or_task == "/stop":
            if not stop_agent_event.is_set():
                _ui_update_func("User command: /stop. Signaling current agent task to stop.")
                stop_agent_event.set()
            else:
                _ui_update_func("Stop signal already active.")
            return True

        if command_or_task: # It's a task
            _ui_update_func(f"New task received: {command_or_task}")
            stop_agent_event.clear() # Clear event for the new task

            app.loop.call_soon_threadsafe(
                app.call_from_executor, # Runs the function in a separate thread
                run_agent_and_update_display, # The function to run
                command_or_task,    # task
                args,               # cli_args
                app,                # app_ref for UI updates from thread
                display_control,    # dc_ref for UI updates from thread
                stop_agent_event    # stop_event
            )
        else: # Empty input
            _ui_update_func("No task or command entered.")
        return True # Keep the buffer (though text is cleared)

    input_field.accept_handler = accept_input_handler

    # Run the application
    exit_code = app.run()

    logging.info(f"Application exited with code {exit_code}.") # Goes to file log
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
