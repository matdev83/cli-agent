from __future__ import annotations

from typing import List, Dict, Optional, Protocol
from dataclasses import dataclass

@dataclass
class LLMUsageInfo:
    """Stores usage information for an LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0

@dataclass
class LLMResponse:
    """Stores the response from an LLM call, including content and usage."""
    content: Optional[str] = None
    usage: Optional[LLMUsageInfo] = None

class LLMWrapper(Protocol):
    """
    Protocol for a Large Language Model wrapper that can send messages and receive responses.
    """

    def send_message(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7, # Default value, can be overridden by implementations
        max_tokens: int = 1024     # Default value, can be overridden by implementations
    ) -> Optional[LLMResponse]: # Return type changed
        """
        Sends a list of messages to the LLM and returns the response.

        Args:
            messages: A list of message dictionaries, where each dictionary
                      has "role" (e.g., "user", "assistant", "system")
                      and "content" (the message text).
            temperature: The sampling temperature for the LLM.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            An LLMResponse object containing the LLM's response and usage info,
            or None if a significant error occurred.
        """
        ...
