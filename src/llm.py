from __future__ import annotations

import json
import logging # Added import
from pathlib import Path
from typing import List, Dict, Optional # Added Optional
import time # Added for retry delays

from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError # Added specific exceptions

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

    def __init__(self, model: str, api_key: str, timeout: Optional[float] = None):
        self.model = model
        self.timeout = timeout
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

        max_retries = 3
        base_delay = 1.0  # seconds

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(**request_params, timeout=self.timeout)

                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                    return content # content can be string or None
                # If response is valid but no choices/message, consider it a success with no content
                logging.info("OpenRouter API returned a response with no content.")
                return None
            except RateLimitError as e:
                logging.warning(f"Rate limit error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt + 1 >= max_retries:
                    logging.error("Max retries reached for rate limit error. Failing.")
                    return None

                retry_after_header = e.response.headers.get("Retry-After")
                delay = base_delay * (2 ** attempt) # Default exponential backoff
                if retry_after_header:
                    try:
                        delay = float(retry_after_header)
                    except ValueError:
                        logging.warning(f"Could not parse Retry-After header value: {retry_after_header}. Using exponential backoff.")

                delay = min(delay, 60) # Cap delay at 60 seconds
                logging.info(f"Waiting {delay:.2f} seconds before retrying.")
                time.sleep(delay)
            except APIStatusError as e:
                logging.warning(f"API status error on attempt {attempt + 1}/{max_retries}: {e.status_code} - {e.message}")
                if e.status_code >= 500 or e.status_code == 429: # Server-side errors or (redundant) rate limit
                    if attempt + 1 >= max_retries:
                        logging.error(f"Max retries reached for API status error {e.status_code}. Failing.")
                        return None
                    delay = min(base_delay * (2 ** attempt), 60)
                    logging.info(f"Waiting {delay:.2f} seconds before retrying for status {e.status_code}.")
                    time.sleep(delay)
                else: # Client-side errors (4xx other than 429)
                    logging.error(f"Client-side API error {e.status_code}. Failing without further retries.")
                    return None # Do not retry for client errors like 401, 403, 400
            except APIConnectionError as e:
                logging.warning(f"API connection error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt + 1 >= max_retries:
                    logging.error("Max retries reached for API connection error. Failing.")
                    return None
                delay = min(base_delay * (2 ** attempt), 60)
                logging.info(f"Waiting {delay:.2f} seconds before retrying for connection error.")
                time.sleep(delay)
            except Exception as e: # Catch-all for other unexpected errors during the API call
                logging.error(f"Unexpected error calling OpenRouter API on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt + 1 >= max_retries:
                    logging.error("Max retries reached for unexpected error. Failing.")
                    return None
                delay = min(base_delay * (2 ** attempt), 30) # Shorter cap for general errors
                logging.info(f"Waiting {delay:.2f} seconds before retrying for unexpected error.")
                time.sleep(delay)

        logging.error("All retries failed after multiple attempts.")
        return None # Return None if all retries fail
