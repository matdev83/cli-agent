import pytest
import time
from unittest.mock import patch, MagicMock, ANY, call

from openai import OpenAI, APITimeoutError, RateLimitError, APIStatusError, APIConnectionError
# Simplify mocking - avoiding direct openai.types imports for response objects
# from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionChoice
# from openai.types import CompletionUsage
# from openai.core import Headers # Removed, as it might be causing issues and requests.Response has headers
from requests import Response # For creating a mock HTTP response for RateLimitError

from src.llm import OpenRouterLLM

@pytest.fixture
def mock_openai_client():
    with patch('src.llm.OpenAI', autospec=True) as mock_constructor:
        mock_client_instance = mock_constructor.return_value
        mock_client_instance.chat = MagicMock()
        mock_client_instance.chat.completions = MagicMock()
        yield mock_client_instance

# --- Timeout Tests ---
def test_openrouter_llm_init_with_timeout(mock_openai_client):
    """Test that OpenRouterLLM initializes the OpenAI client with the timeout."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key", timeout=30.0)
    assert llm.timeout == 30.0

def test_openrouter_llm_send_message_uses_timeout(mock_openai_client):
    """Test that send_message passes the timeout to completions.create."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key", timeout=45.0)

    mock_choice = MagicMock()
    mock_choice.message = MagicMock()
    mock_choice.message.content = "Test response"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_completion

    messages = [{"role": "user", "content": "Hello"}]
    llm.send_message(messages)

    mock_openai_client.chat.completions.create.assert_called_once_with(
        model="test/model",
        messages=[{"role": "user", "content": "Hello"}],
        timeout=45.0
    )

def test_openrouter_llm_send_message_handles_timeout_error(mock_openai_client):
    """Test that send_message returns None when APITimeoutError is raised and retries occur."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key", timeout=0.1)

    mock_openai_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())

    messages = [{"role": "user", "content": "Hello"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        response = llm.send_message(messages)

    assert response is None
    assert mock_openai_client.chat.completions.create.call_count == 3
    assert mock_sleep.call_count == 2

def test_openrouter_llm_send_message_no_timeout_set(mock_openai_client):
    """Test that send_message works correctly when no timeout is set on LLM."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key")

    mock_choice = MagicMock()
    mock_choice.message = MagicMock()
    mock_choice.message.content = "Test response"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_completion

    messages = [{"role": "user", "content": "Hello"}]
    llm.send_message(messages)

    mock_openai_client.chat.completions.create.assert_called_once_with(
        model="test/model",
        messages=[{"role": "user", "content": "Hello"}],
        timeout=None
    )

# --- HTTP Error Retry Tests ---

def test_openrouter_llm_retries_on_ratelimiterror(mock_openai_client):
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

    mock_openai_client.chat.completions.create.side_effect = [
        RateLimitError("Rate limited", response=mock_http_response, body=None),
        RateLimitError("Rate limited again", response=mock_http_response, body=None),
        mock_successful_completion
    ]

    messages = [{"role": "user", "content": "Test"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        response_content = llm.send_message(messages)

    assert response_content == "Success after retries"
    assert mock_openai_client.chat.completions.create.call_count == 3
    # Check that sleep was called with the Retry-After value
    mock_sleep.assert_any_call(0.1)
    assert mock_sleep.call_count == 2

def test_openrouter_llm_retries_on_ratelimiterror_no_retry_after(mock_openai_client):
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

    mock_openai_client.chat.completions.create.side_effect = [
        RateLimitError("Rate limited", response=mock_http_response_no_header, body=None),
        mock_successful_completion_one_retry
    ]

    messages = [{"role": "user", "content": "Test"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        response_content = llm.send_message(messages)

    assert response_content == "Success after one retry"
    assert mock_openai_client.chat.completions.create.call_count == 2
    mock_sleep.assert_called_once_with(1.0) # base_delay * (2**0)

def test_openrouter_llm_retries_on_apistatuserror_5xx(mock_openai_client):
    """Test retries on APIStatusError with 5xx status code."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key")

    mock_http_response_500 = Response()
    mock_http_response_500.status_code = 500 # Server error

    mock_successful_completion_5xx = MagicMock()
    mock_successful_choice_5xx = MagicMock()
    mock_successful_choice_5xx.message = MagicMock()
    mock_successful_choice_5xx.message.content = "Success after 5xx retry"
    mock_successful_completion_5xx.choices = [mock_successful_choice_5xx]

    mock_openai_client.chat.completions.create.side_effect = [
        APIStatusError("Server error", response=mock_http_response_500, body=None),
        mock_successful_completion_5xx
    ]

    messages = [{"role": "user", "content": "Test"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        response_content = llm.send_message(messages)

    assert response_content == "Success after 5xx retry"
    assert mock_openai_client.chat.completions.create.call_count == 2
    mock_sleep.assert_called_once_with(1.0) # Exponential backoff

def test_openrouter_llm_no_retry_on_apistatuserror_4xx_client_error(mock_openai_client):
    """Test no extensive retries for client-side 4xx errors (e.g., 401, 403)."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key")

    mock_http_response_401 = Response()
    mock_http_response_401.status_code = 401 # Unauthorized

    mock_openai_client.chat.completions.create.side_effect = APIStatusError(
        "Unauthorized", response=mock_http_response_401, body=None
    )

    messages = [{"role": "user", "content": "Test"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        response_content = llm.send_message(messages)

    assert response_content is None
    assert mock_openai_client.chat.completions.create.call_count == 1 # Fails on first attempt
    assert mock_sleep.call_count == 0 # No sleep, no retry

def test_openrouter_llm_all_retries_exhausted(mock_openai_client):
    """Test returns None if all retries are exhausted for a retriable error."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key")

    mock_http_response_503 = Response()
    mock_http_response_503.status_code = 503 # Service unavailable

    mock_openai_client.chat.completions.create.side_effect = APIStatusError(
        "Service unavailable", response=mock_http_response_503, body=None
    ) # Will raise this for all 3 attempts

    messages = [{"role": "user", "content": "Test"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        response_content = llm.send_message(messages)

    assert response_content is None
    assert mock_openai_client.chat.completions.create.call_count == 3
    assert mock_sleep.call_count == 2 # Sleeps between 3 attempts

def test_openrouter_llm_retries_on_apiconnectionerror(mock_openai_client):
    """Test retries on APIConnectionError."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key")

    mock_successful_completion_conn_err = MagicMock()
    mock_successful_choice_conn_err = MagicMock()
    mock_successful_choice_conn_err.message = MagicMock()
    mock_successful_choice_conn_err.message.content = "Success after connection error"
    mock_successful_completion_conn_err.choices = [mock_successful_choice_conn_err]

    mock_openai_client.chat.completions.create.side_effect = [
        APIConnectionError(request=MagicMock()), # Simulate connection error
        mock_successful_completion_conn_err
    ]

    messages = [{"role": "user", "content": "Test"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        response_content = llm.send_message(messages)

    assert response_content == "Success after connection error"
    assert mock_openai_client.chat.completions.create.call_count == 2
    mock_sleep.assert_called_once_with(1.0) # Exponential backoff

# To run this file directly for testing:
# pytest tests/test_llm.py
