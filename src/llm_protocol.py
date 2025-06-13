from __future__ import annotations

from typing import List, Dict, Optional, Protocol, Tuple, Any

class LLMWrapper(Protocol):
    """
    Protocol for a Large Language Model wrapper that can send messages and receive responses.
    """

    def send_message(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7, # Default value, can be overridden by implementations
        max_tokens: int = 1024     # Default value, can be overridden by implementations
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Sends a list of messages to the LLM and returns the response and usage info.

        Args:
            messages: A list of message dictionaries, where each dictionary
                      has "role" (e.g., "user", "assistant", "system")
                      and "content" (the message text).
            temperature: The sampling temperature for the LLM.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            A tuple containing:
                - An Optional[str] representing the LLM's response content,
                  or None if no response is generated.
                - An Optional[Dict[str, Any]] containing usage information like
                  token counts and cost, or None if not available.
        """
        ...
