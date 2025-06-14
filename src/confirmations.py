"""
Handles user confirmation prompts for CLI interactions.
"""

from __future__ import annotations

import logging


def request_user_confirmation(prompt_message: str) -> bool:
    """
    Requests user confirmation with a y/n prompt.

    Args:
        prompt_message: The message to display to the user (e.g., "Allow action? (y/n)").

    Returns:
        True if the user confirms with 'y' (case-insensitive), False otherwise.
    """
    print(prompt_message, end=" ", flush=True)
    try:
        user_input = input()
        return user_input.strip().lower() == "y"
    except KeyboardInterrupt:
        logging.info("\nConfirmation cancelled by user.")
        return False
    except EOFError:  # Handle cases where stdin is closed unexpectedly
        logging.info("\nConfirmation input stream closed.")
        return False
