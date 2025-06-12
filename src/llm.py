from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional # Added Optional

from openai import OpenAI

from src.llm_protocol import LLMWrapper # Import the protocol

class MockLLM(LLMWrapper): # Indicate conformance to the protocol
    """Simple mock LLM that returns predefined responses."""

    def __init__(self, responses: List[str]):
        self._responses = list(responses)
        self._index = 0

    @classmethod
    def from_file(cls, path: str) -> "MockLLM":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("responses file must contain a JSON list")
        return cls([str(item) for item in data])

    def send_message(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7, # Added to match protocol
        max_tokens: int = 1024     # Added to match protocol
    ) -> Optional[str]: # Changed return type to Optional[str]
        # temperature and max_tokens are ignored in this mock implementation
        if self._index >= len(self._responses):
            return None # Return None if no more responses, fits Optional[str]
        resp = self._responses[self._index]
        self._index += 1
        return resp # This is a string, fits Optional[str]


class OpenRouterLLM(LLMWrapper): # Indicate conformance to the protocol
    """LLM client using OpenRouter.ai API via the OpenAI SDK."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://cline.bot", # Example referrer
                "X-Title": "CLI Agent",              # Example title
            },
        )

    def send_message(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,  # Added to match protocol
        max_tokens: int = 1024      # Added to match protocol
    ) -> Optional[str]: # Changed return type to Optional[str]

        # Prepare request parameters, including temperature and max_tokens if model supports them
        request_params = {
            "model": self.model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        }
        # Only add temperature and max_tokens if they are not default, or if API expects them.
        # For OpenAI compatible APIs, these are common.
        if temperature != 0.7: # Or some other way to check if it should be included
            request_params["temperature"] = temperature
        if max_tokens != 1024: # Or some other way to check
            request_params["max_tokens"] = max_tokens

        try:
            response = self._client.chat.completions.create(**request_params)

            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                return content # content can be string or None
            return None # No response or empty choice
        except Exception as e:
            # Log error e
            print(f"Error calling OpenRouter API: {e}") # Simple error logging
            return None # Return None on API error
