from __future__ import annotations

import json
import logging  # Added import
from pathlib import Path
from typing import List, Dict, Optional, Any  # Added Any
import time  # Added for retry delays

from openai import (
    OpenAI,
    APIStatusError,
    APIConnectionError,
    RateLimitError,
)  # Added specific exceptions

from src.llm_protocol import LLMWrapper, LLMResponse, LLMUsageInfo  # Import the protocol


class MockLLM(LLMWrapper):  # Indicate conformance to the protocol
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
        temperature: float = 0.7,  # Added to match protocol
        max_tokens: int = 1024,  # Added to match protocol
    ) -> Optional[LLMResponse]:  # Return type changed
        # temperature and max_tokens are ignored in this mock implementation

        dummy_usage = LLMUsageInfo(prompt_tokens=10, completion_tokens=20, cost=0.0)

        if self._index >= len(self._responses):
            # Consistent with OpenRouterLLM's error return, provide a default structure
            return LLMResponse(content=None, usage=dummy_usage)

        resp_content = self._responses[self._index]
        self._index += 1

        return LLMResponse(content=resp_content, usage=dummy_usage)


class OpenRouterLLM(LLMWrapper):  # Indicate conformance to the protocol
    """LLM client using OpenRouter.ai API via the OpenAI SDK."""

    def __init__(self, model: str, api_key: str, timeout: Optional[float] = None):
        self.model = model
        self.timeout = timeout
        # TODO: Consider adding a staticmethod for default_empty_llm_response if not in protocol
        self._default_empty_response = LLMResponse(content=None, usage=LLMUsageInfo(prompt_tokens=0, completion_tokens=0, cost=0.0))
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://cline.bot",  # Example referrer
                "X-Title": "CLI Agent",  # Example title
            },
        )

    def send_message(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,  # Added to match protocol
        max_tokens: int = 1024,  # Added to match protocol
    ) -> Optional[LLMResponse]:  # Changed return type to Optional[LLMResponse]
        # Prepare request parameters, including temperature and max_tokens if model supports them
        request_params = {
            "model": self.model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        }
        # Only add temperature and max_tokens if they are not default, or if API expects them.
        # For OpenAI compatible APIs, these are common.
        if temperature != 0.7:  # Or some other way to check if it should be included
            request_params["temperature"] = temperature
        if max_tokens != 1024:  # Or some other way to check
            request_params["max_tokens"] = max_tokens

        max_retries = 3
        base_delay = 1.0  # seconds

        for attempt in range(max_retries):
            should_retry = False
            current_delay = base_delay * (2**attempt)  # Default exponential backoff for this attempt
            log_error_type = "Unknown error" # Placeholder for specific error type in logs

            try:
                response = self._client.chat.completions.create(
                    **request_params, timeout=self.timeout
                )

                content: Optional[str] = None
                usage_info: Optional[LLMUsageInfo] = None

                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content

                usage_info = self._parse_usage_info(response)
                return LLMResponse(content=content, usage=usage_info)

            except RateLimitError as e:
                log_error_type = "Rate limit error"
                logging.warning(f"{log_error_type} on attempt {attempt + 1}/{max_retries}: {e}")
                should_retry = True
                retry_after_header = e.response.headers.get("Retry-After")
                if retry_after_header:
                    try:
                        current_delay = float(retry_after_header)
                    except ValueError:
                        logging.warning(
                            f"Could not parse Retry-After header value: {retry_after_header}. Using exponential backoff."
                        )
                current_delay = min(current_delay, 60)  # Cap delay
            except APIStatusError as e:
                log_error_type = f"API status error ({e.status_code})"
                logging.warning(f"{log_error_type} on attempt {attempt + 1}/{max_retries}: {e.message}")
                # Do not retry client-side errors (4xx) other than 429 (RateLimitError)
                if 400 <= e.status_code < 500 and e.status_code != 429:
                    logging.error(f"Client-side {log_error_type} not retried. Failing.")
                    # Assuming LLMResponse.default_empty() is available or will be added
                    return LLMResponse(content=None, usage=LLMUsageInfo(prompt_tokens=0, completion_tokens=0, cost=0.0))
                should_retry = True
                current_delay = min(current_delay, 60)  # Cap delay
            except APIConnectionError as e:
                log_error_type = "API connection error"
                logging.warning(f"{log_error_type} on attempt {attempt + 1}/{max_retries}: {e}")
                should_retry = True
                current_delay = min(current_delay, 60)  # Cap delay
            except Exception as e:
                log_error_type = "Unexpected error"
                # Log with exc_info=True for unexpected errors to get traceback
                logging.error(
                    f"{log_error_type} calling OpenRouter API on attempt {attempt + 1}/{max_retries}: {e}",
                    exc_info=True
                )
                should_retry = True # Decide if all unexpected errors are retryable
                current_delay = min(current_delay, 30)  # Shorter cap for general errors

            if should_retry and attempt + 1 < max_retries:
                logging.info(f"Waiting {current_delay:.2f} seconds before retrying for {log_error_type}.")
                time.sleep(current_delay)
            elif should_retry: # Last attempt failed
                logging.error(f"Max retries reached for {log_error_type}. Failing.")
                # Assuming LLMResponse.default_empty() is available or will be added
                return LLMResponse(content=None, usage=LLMUsageInfo(prompt_tokens=0, completion_tokens=0, cost=0.0))
            # If should_retry is False, the error was non-retryable and already handled.
            # Loop will exit or specific error block returned. This path implies non-retryable error handled inside except.

        # Fallback if loop completes (e.g. if should_retry was false on last attempt in a way not returning directly)
        # This typically indicates all retries failed or a non-retryable error occurred.
        logging.error("All retries failed or a non-retryable error occurred after multiple attempts.")
        return self._default_empty_response

    @staticmethod
    def _parse_usage_info(response_obj: Any) -> LLMUsageInfo:
        """Parses usage information from an OpenRouter API response object."""
        # Ensure response_obj and response_obj.usage are not None
        if not hasattr(response_obj, "usage") or not response_obj.usage:
            return LLMUsageInfo(prompt_tokens=0, completion_tokens=0, cost=0.0)

        usage_data = response_obj.usage
        parsed_cost = 0.0

        # Potential sources of cost information, in order of preference
        cost_sources_values = [
            getattr(usage_data, "cost", None),
            getattr(usage_data, "total_cost", None),
        ]
        # Fallback: check cost directly on the response object itself if not found in usage_data
        # This is less standard for OpenAI SDK but included for robustness from original logic.
        if not any(cs is not None for cs in cost_sources_values): # only if not found in usage_data
             cost_sources_values.append(getattr(response_obj, "cost", None))


        for cost_val in cost_sources_values:
            if cost_val is not None and isinstance(cost_val, (int, float, str)):
                try:
                    parsed_cost = float(cost_val)
                    if parsed_cost >= 0:  # Found a valid, non-negative cost
                        break
                except (TypeError, ValueError):
                    continue  # Try next source if current one is not a valid float

        def _to_int_internal(val: Any) -> int:
            if val is None:
                return 0
            if not isinstance(val, (int, float, str)):
                return 0
            try:
                return int(val)
            except (TypeError, ValueError):
                return 0

        return LLMUsageInfo(
            prompt_tokens=_to_int_internal(getattr(usage_data, "prompt_tokens", 0)),
            completion_tokens=_to_int_internal(getattr(usage_data, "completion_tokens", 0)),
            cost=parsed_cost,
        )
