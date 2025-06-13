import pytest
import time
import unittest # For self.assertEqual etc.
from unittest.mock import patch, MagicMock, ANY, call

from openai import OpenAI, APITimeoutError, RateLimitError, APIStatusError, APIConnectionError
# Simplify mocking - avoiding direct openai.types imports for response objects
# from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionChoice
# from openai.types import CompletionUsage
# from openai.core import Headers # Removed, as it might be causing issues and requests.Response has headers
from requests import Response # For creating a mock HTTP response for RateLimitError

from src.llm import OpenRouterLLM, MockLLM

@pytest.fixture
def mock_openai_client_constructor():
    with patch('src.llm.OpenAI', autospec=True) as mock_constructor:
        # mock_client_instance = mock_constructor.return_value # This is the client instance
        # mock_client_instance.chat = MagicMock()
        # mock_client_instance.chat.completions = MagicMock()
        yield mock_constructor # Yield the constructor mock itself for verifying __init__ calls

@pytest.fixture
def mock_openai_client_instance(mock_openai_client_constructor):
    """Provides a mocked OpenAI client instance, separate from constructor."""
    mock_client_instance = mock_openai_client_constructor.return_value
    mock_client_instance.chat = MagicMock()
    mock_client_instance.chat.completions = MagicMock()
    return mock_client_instance


# --- Timeout Tests ---
def test_openrouter_llm_init_with_timeout(mock_openai_client_constructor):
    """Test that OpenRouterLLM initializes the OpenAI client with the timeout."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key", timeout=30.0)
    assert llm.timeout == 30.0
    # mock_openai_client_constructor.assert_called_once() # Already called by fixture setup
    # Check args if needed, but timeout is not passed to OpenAI() constructor directly in current code

def test_openrouter_llm_send_message_uses_timeout(mock_openai_client_instance):
    """Test that send_message passes the timeout to completions.create."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key", timeout=45.0)
    # The llm instance uses the mock_openai_client_instance via the fixture chain

    mock_choice = MagicMock()
    mock_choice.message = MagicMock()
    mock_choice.message.content = "Test response"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = None # For this test, usage is not the focus
    mock_openai_client_instance.chat.completions.create.return_value = mock_completion

    messages = [{"role": "user", "content": "Hello"}]
    llm.send_message(messages)

    mock_openai_client_instance.chat.completions.create.assert_called_once_with(
        model="test/model",
        messages=[{"role": "user", "content": "Hello"}],
        timeout=45.0
    )

def test_openrouter_llm_send_message_handles_timeout_error(mock_openai_client_instance):
    """Test that send_message returns None, None when APITimeoutError is raised and retries occur."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key", timeout=0.1)

    mock_openai_client_instance.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())

    messages = [{"role": "user", "content": "Hello"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        content, usage_info = llm.send_message(messages)

    assert content is None
    assert usage_info is None
    assert mock_openai_client_instance.chat.completions.create.call_count == 3
    assert mock_sleep.call_count == 2

def test_openrouter_llm_send_message_no_timeout_set(mock_openai_client_instance):
    """Test that send_message works correctly when no timeout is set on LLM."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key")

    mock_choice = MagicMock()
    mock_choice.message = MagicMock()
    mock_choice.message.content = "Test response"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = None # For this test, usage is not the focus
    mock_openai_client_instance.chat.completions.create.return_value = mock_completion

    messages = [{"role": "user", "content": "Hello"}]
    llm.send_message(messages)

    mock_openai_client_instance.chat.completions.create.assert_called_once_with(
        model="test/model",
        messages=[{"role": "user", "content": "Hello"}],
        timeout=None
    )

# --- HTTP Error Retry Tests ---

def test_openrouter_llm_retries_on_ratelimiterror(mock_openai_client_instance):
    """Test retries on RateLimitError and respects Retry-After header."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key")

    # Mock a requests.Response object for the RateLimitError
    mock_http_response = Response()
    mock_http_response.status_code = 429
    mock_http_response.headers["Retry-After"] = "0.1" # Retry after 0.1 seconds

    mock_successful_completion = MagicMock()
    mock_successful_choice = MagicMock()
    mock_successful_choice.message = MagicMock()
    mock_successful_choice.message.content = "Success after retries"
    mock_successful_completion.choices = [mock_successful_choice]
    mock_successful_completion.usage = None # For this test, usage is not the focus

    mock_openai_client_instance.chat.completions.create.side_effect = [
        RateLimitError("Rate limited", response=mock_http_response, body=None),
        RateLimitError("Rate limited again", response=mock_http_response, body=None),
        mock_successful_completion
    ]

    messages = [{"role": "user", "content": "Test"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        response_content, usage_info = llm.send_message(messages)

    assert response_content == "Success after retries"
    assert usage_info is None # As mock_successful_completion.usage is None
    assert mock_openai_client_instance.chat.completions.create.call_count == 3
    # Check that sleep was called with the Retry-After value
    mock_sleep.assert_any_call(0.1)
    assert mock_sleep.call_count == 2

def test_openrouter_llm_retries_on_ratelimiterror_no_retry_after(mock_openai_client_instance):
    """Test retries on RateLimitError with exponential backoff if no Retry-After."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key")

    mock_http_response_no_header = Response()
    mock_http_response_no_header.status_code = 429
    # No Retry-After header

    mock_successful_completion_one_retry = MagicMock()
    mock_successful_choice_one_retry = MagicMock()
    mock_successful_choice_one_retry.message = MagicMock()
    mock_successful_choice_one_retry.message.content = "Success after one retry"
    mock_successful_completion_one_retry.choices = [mock_successful_choice_one_retry]
    mock_successful_completion_one_retry.usage = None # For this test, usage is not the focus

    mock_openai_client_instance.chat.completions.create.side_effect = [
        RateLimitError("Rate limited", response=mock_http_response_no_header, body=None),
        mock_successful_completion_one_retry
    ]

    messages = [{"role": "user", "content": "Test"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        response_content, _ = llm.send_message(messages)

    assert response_content == "Success after one retry"
    assert mock_openai_client_instance.chat.completions.create.call_count == 2
    mock_sleep.assert_called_once_with(1.0) # base_delay * (2**0)

def test_openrouter_llm_retries_on_apistatuserror_5xx(mock_openai_client_instance):
    """Test retries on APIStatusError with 5xx status code."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key")

    mock_http_response_500 = Response()
    mock_http_response_500.status_code = 500 # Server error

    mock_successful_completion_5xx = MagicMock()
    mock_successful_choice_5xx = MagicMock()
    mock_successful_choice_5xx.message = MagicMock()
    mock_successful_choice_5xx.message.content = "Success after 5xx retry"
    mock_successful_completion_5xx.choices = [mock_successful_choice_5xx]
    mock_successful_completion_5xx.usage = None

    mock_openai_client_instance.chat.completions.create.side_effect = [
        APIStatusError("Server error", response=mock_http_response_500, body=None),
        mock_successful_completion_5xx
    ]

    messages = [{"role": "user", "content": "Test"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        response_content, _ = llm.send_message(messages)

    assert response_content == "Success after 5xx retry"
    assert mock_openai_client_instance.chat.completions.create.call_count == 2
    mock_sleep.assert_called_once_with(1.0) # Exponential backoff

def test_openrouter_llm_no_retry_on_apistatuserror_4xx_client_error(mock_openai_client_instance):
    """Test no extensive retries for client-side 4xx errors (e.g., 401, 403)."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key")

    mock_http_response_401 = Response()
    mock_http_response_401.status_code = 401 # Unauthorized

    mock_openai_client_instance.chat.completions.create.side_effect = APIStatusError(
        "Unauthorized", response=mock_http_response_401, body=None
    )

    messages = [{"role": "user", "content": "Test"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        response_content, usage_info = llm.send_message(messages)

    assert response_content is None
    assert usage_info is None
    assert mock_openai_client_instance.chat.completions.create.call_count == 1 # Fails on first attempt
    assert mock_sleep.call_count == 0 # No sleep, no retry

def test_openrouter_llm_all_retries_exhausted(mock_openai_client_instance):
    """Test returns None, None if all retries are exhausted for a retriable error."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key")

    mock_http_response_503 = Response()
    mock_http_response_503.status_code = 503 # Service unavailable

    mock_openai_client_instance.chat.completions.create.side_effect = APIStatusError(
        "Service unavailable", response=mock_http_response_503, body=None
    ) # Will raise this for all 3 attempts

    messages = [{"role": "user", "content": "Test"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        response_content, usage_info = llm.send_message(messages)

    assert response_content is None
    assert usage_info is None
    assert mock_openai_client_instance.chat.completions.create.call_count == 3
    assert mock_sleep.call_count == 2 # Sleeps between 3 attempts

def test_openrouter_llm_retries_on_apiconnectionerror(mock_openai_client_instance):
    """Test retries on APIConnectionError."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key")

    mock_successful_completion_conn_err = MagicMock()
    mock_successful_choice_conn_err = MagicMock()
    mock_successful_choice_conn_err.message = MagicMock()
    mock_successful_choice_conn_err.message.content = "Success after connection error"
    mock_successful_completion_conn_err.choices = [mock_successful_choice_conn_err]
    mock_successful_completion_conn_err.usage = None

    mock_openai_client_instance.chat.completions.create.side_effect = [
        APIConnectionError(request=MagicMock()), # Simulate connection error
        mock_successful_completion_conn_err
    ]

    messages = [{"role": "user", "content": "Test"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        response_content, _ = llm.send_message(messages)

    assert response_content == "Success after connection error"
    assert mock_openai_client_instance.chat.completions.create.call_count == 2
    mock_sleep.assert_called_once_with(1.0) # Exponential backoff


class TestLLMResponses(unittest.TestCase):
    @patch('src.llm.OpenAI') # Patch the constructor
    def test_openrouterllm_send_message_with_usage(self, mock_openai_constructor):
        mock_client_instance = mock_openai_constructor.return_value
        mock_create_method = mock_client_instance.chat.completions.create

        llm = OpenRouterLLM(model="test_model", api_key="fake_key")

        mock_completion_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Test LLM response"
        mock_choice.message = mock_message
        mock_completion_response.choices = [mock_choice]

        mock_usage_obj = MagicMock()
        mock_usage_obj.prompt_tokens = 50
        mock_usage_obj.completion_tokens = 100
        mock_usage_obj.cost = 0.000123 # USD
        mock_completion_response.usage = mock_usage_obj

        mock_create_method.return_value = mock_completion_response

        messages = [{"role": "user", "content": "Test message"}]
        content, usage_info = llm.send_message(messages)

        mock_create_method.assert_called_once()
        # Check default_headers passed to OpenAI constructor
        constructor_args, constructor_kwargs = mock_openai_constructor.call_args
        self.assertIn("default_headers", constructor_kwargs)
        self.assertEqual(
            constructor_kwargs["default_headers"]["X-OpenRouter-Settings"],
            '{"return_usage": true}'
        )

        self.assertEqual(content, "Test LLM response")
        self.assertIsNotNone(usage_info)
        self.assertEqual(usage_info['model_name'], 'test_model')
        self.assertEqual(usage_info['prompt_tokens'], 50)
        self.assertEqual(usage_info['completion_tokens'], 100)
        self.assertEqual(usage_info['cost'], 0.000123)

    def test_mock_llm_return_type_and_exhaustion(self):
        llm = MockLLM(responses=["Hello", "World"])

        content1, usage1 = llm.send_message([])
        self.assertEqual(content1, "Hello")
        self.assertIsNone(usage1)

        content2, usage2 = llm.send_message([])
        self.assertEqual(content2, "World")
        self.assertIsNone(usage2)

        # Test exhaustion
        content_exhausted, usage_exhausted = llm.send_message([])
        self.assertIsNone(content_exhausted)
        self.assertIsNone(usage_exhausted)

# To run this file directly for testing:
# pytest tests/test_llm.py
# or python -m unittest tests.test_llm (if using unittest structure primarily)
