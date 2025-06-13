import pytest
import time
from unittest.mock import patch, MagicMock, ANY, call

from openai import OpenAI, APITimeoutError, RateLimitError, APIStatusError, APIConnectionError
# Simplify mocking - avoiding direct openai.types imports for response objects
# from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionChoice
# from openai.types import CompletionUsage
# from openai.core import Headers # Removed, as it might be causing issues and requests.Response has headers
from requests import Response # For creating a mock HTTP response for RateLimitError

from src.llm import OpenRouterLLM, MockLLM # Added MockLLM
from src.llm_protocol import LLMResponse, LLMUsageInfo # Added LLMResponse and LLMUsageInfo
import json # For MockLLM test file

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
    # No usage attribute in this mock, so default usage is expected
    mock_openai_client.chat.completions.create.return_value = mock_completion

    messages = [{"role": "user", "content": "Hello"}]
    result = llm.send_message(messages)

    assert isinstance(result, LLMResponse)
    assert result.content == "Test response"
    assert isinstance(result.usage, LLMUsageInfo)
    assert result.usage.prompt_tokens == 0 # Default due to no usage in mock
    assert result.usage.completion_tokens == 0 # Default due to no usage in mock

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
        result = llm.send_message(messages)

    assert isinstance(result, LLMResponse)
    assert result.content is None
    assert isinstance(result.usage, LLMUsageInfo)
    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0
    assert result.usage.cost == 0.0
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
    # No usage attribute in this mock
    mock_openai_client.chat.completions.create.return_value = mock_completion

    messages = [{"role": "user", "content": "Hello"}]
    result = llm.send_message(messages)

    assert isinstance(result, LLMResponse)
    assert result.content == "Test response"
    assert isinstance(result.usage, LLMUsageInfo)
    assert result.usage.prompt_tokens == 0 # Default
    assert result.usage.completion_tokens == 0 # Default


    mock_openai_client.chat.completions.create.assert_called_once_with(
        model="test/model",
        messages=[{"role": "user", "content": "Hello"}],
        timeout=None # Ensure this is not mistaken for LLMResponse content
    )

# --- MockLLM Tests ---

class TestMockLLM:
    @pytest.fixture
    def mock_responses_file(self, tmp_path):
        file_path = tmp_path / "mock_responses.json"
        responses = ["Test response 1", "Test response 2"]
        with open(file_path, "w") as f:
            json.dump(responses, f)
        return file_path

    def test_mock_llm_from_file(self, mock_responses_file):
        llm = MockLLM.from_file(str(mock_responses_file))
        assert isinstance(llm, MockLLM)
        # First message
        response1 = llm.send_message([{"role": "user", "content": "Hello 1"}])
        assert isinstance(response1, LLMResponse)
        assert response1.content == "Test response 1"
        assert isinstance(response1.usage, LLMUsageInfo)
        assert response1.usage.prompt_tokens == 10
        assert response1.usage.completion_tokens == 20
        assert response1.usage.cost == 0.0
        # Second message
        response2 = llm.send_message([{"role": "user", "content": "Hello 2"}])
        assert isinstance(response2, LLMResponse)
        assert response2.content == "Test response 2"
        assert isinstance(response2.usage, LLMUsageInfo) # Check usage again for second response
        assert response2.usage.prompt_tokens == 10
        assert response2.usage.completion_tokens == 20
        assert response2.usage.cost == 0.0


    def test_mock_llm_send_message(self):
        responses = ["Response A", "Response B"]
        llm = MockLLM(responses)

        # First message
        response1 = llm.send_message([{"role": "user", "content": "First message"}])
        assert isinstance(response1, LLMResponse)
        assert response1.content == "Response A"
        assert isinstance(response1.usage, LLMUsageInfo)
        assert response1.usage.prompt_tokens == 10
        assert response1.usage.completion_tokens == 20
        assert response1.usage.cost == 0.0

        # Second message
        response2 = llm.send_message([{"role": "user", "content": "Second message"}])
        assert isinstance(response2, LLMResponse)
        assert response2.content == "Response B"
        assert isinstance(response2.usage, LLMUsageInfo)
        assert response2.usage.prompt_tokens == 10
        assert response2.usage.completion_tokens == 20
        assert response2.usage.cost == 0.0

    def test_mock_llm_send_message_exhausted(self):
        responses = ["Single response"]
        llm = MockLLM(responses)

        # First message - consumes the only response
        llm.send_message([{"role": "user", "content": "Consume response"}])

        # Second message - responses exhausted
        exhausted_response = llm.send_message([{"role": "user", "content": "Exhausted"}])
        assert isinstance(exhausted_response, LLMResponse)
        assert exhausted_response.content is None
        assert isinstance(exhausted_response.usage, LLMUsageInfo)
        assert exhausted_response.usage.prompt_tokens == 10 # Dummy usage still provided
        assert exhausted_response.usage.completion_tokens == 20
        assert exhausted_response.usage.cost == 0.0

# --- OpenRouterLLM Tests --- (Keeping separate class for clarity if OpenRouter tests grow)
# Note: Many existing OpenRouterLLM tests already use mock_openai_client fixture.
# We will adapt them below.

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
    # No usage in this mock success response
    mock_openai_client.chat.completions.create.side_effect = [
        RateLimitError("Rate limited", response=mock_http_response, body=None),
        RateLimitError("Rate limited again", response=mock_http_response, body=None),
        mock_successful_completion
    ]

    messages = [{"role": "user", "content": "Test"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        result = llm.send_message(messages)

    assert isinstance(result, LLMResponse)
    assert result.content == "Success after retries"
    assert isinstance(result.usage, LLMUsageInfo) # Should have default usage
    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0
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
    # No usage in this mock success response
    mock_openai_client.chat.completions.create.side_effect = [
        RateLimitError("Rate limited", response=mock_http_response_no_header, body=None),
        mock_successful_completion_one_retry
    ]

    messages = [{"role": "user", "content": "Test"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        result = llm.send_message(messages)

    assert isinstance(result, LLMResponse)
    assert result.content == "Success after one retry"
    assert isinstance(result.usage, LLMUsageInfo) # Default usage
    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0
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
    # No usage in this mock success response
    mock_openai_client.chat.completions.create.side_effect = [
        APIStatusError("Server error", response=mock_http_response_500, body=None),
        mock_successful_completion_5xx
    ]

    messages = [{"role": "user", "content": "Test"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        result = llm.send_message(messages)

    assert isinstance(result, LLMResponse)
    assert result.content == "Success after 5xx retry"
    assert isinstance(result.usage, LLMUsageInfo) # Default usage
    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0
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
        result = llm.send_message(messages)

    assert isinstance(result, LLMResponse)
    assert result.content is None
    assert isinstance(result.usage, LLMUsageInfo)
    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0
    assert result.usage.cost == 0.0
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
        result = llm.send_message(messages)

    assert isinstance(result, LLMResponse)
    assert result.content is None
    assert isinstance(result.usage, LLMUsageInfo)
    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0
    assert result.usage.cost == 0.0
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
    # No usage in this mock success response
    mock_openai_client.chat.completions.create.side_effect = [
        APIConnectionError(request=MagicMock()), # Simulate connection error
        mock_successful_completion_conn_err
    ]

    messages = [{"role": "user", "content": "Test"}]
    with patch('time.sleep', return_value=None) as mock_sleep:
        result = llm.send_message(messages)

    assert isinstance(result, LLMResponse)
    assert result.content == "Success after connection error"
    assert isinstance(result.usage, LLMUsageInfo) # Default usage
    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0
    assert mock_openai_client.chat.completions.create.call_count == 2
    mock_sleep.assert_called_once_with(1.0) # Exponential backoff

def test_openrouter_llm_send_message_success_with_usage(mock_openai_client):
    """Test successful message sending and includes usage info."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key")

    mock_choice = MagicMock()
    mock_choice.message = MagicMock()
    mock_choice.message.content = "Successful response with usage"

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = MagicMock() # Simulate usage attribute
    mock_completion.usage.prompt_tokens = 70
    mock_completion.usage.completion_tokens = 80

    mock_openai_client.chat.completions.create.return_value = mock_completion

    messages = [{"role": "user", "content": "Hello with usage"}]
    result = llm.send_message(messages)

    assert isinstance(result, LLMResponse)
    assert result.content == "Successful response with usage"
    assert isinstance(result.usage, LLMUsageInfo)
    assert result.usage.prompt_tokens == 70
    assert result.usage.completion_tokens == 80
    assert result.usage.cost == 0.0 # Cost is not calculated from API response yet

def test_openrouter_llm_send_message_no_content(mock_openai_client):
    """Test message sending where API returns no content but provides usage."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key")

    mock_choice_no_content = MagicMock()
    mock_choice_no_content.message = MagicMock()
    mock_choice_no_content.message.content = None # No content

    mock_completion_no_content = MagicMock()
    mock_completion_no_content.choices = [mock_choice_no_content]
    mock_completion_no_content.usage = MagicMock()
    mock_completion_no_content.usage.prompt_tokens = 15
    mock_completion_no_content.usage.completion_tokens = 5 # e.g. filter/stop

    mock_openai_client.chat.completions.create.return_value = mock_completion_no_content

    messages = [{"role": "user", "content": "Trigger no content"}]
    result = llm.send_message(messages)

    assert isinstance(result, LLMResponse)
    assert result.content is None
    assert isinstance(result.usage, LLMUsageInfo)
    assert result.usage.prompt_tokens == 15
    assert result.usage.completion_tokens == 5
    assert result.usage.cost == 0.0

def test_openrouter_llm_send_message_no_usage_from_api(mock_openai_client):
    """Test message sending where API returns content but no usage info."""
    llm = OpenRouterLLM(model="test/model", api_key="test_key")

    mock_choice_content_only = MagicMock()
    mock_choice_content_only.message = MagicMock()
    mock_choice_content_only.message.content = "Content but no usage"

    mock_completion_no_usage = MagicMock()
    mock_completion_no_usage.choices = [mock_choice_content_only]
    # Simulate response object where 'usage' attribute might be missing or None
    # One way: mock_completion_no_usage.usage = None
    # Another way: ensure hasattr(mock_completion_no_usage, 'usage') is false
    # For this test, setting it to None is sufficient as OpenRouterLLM checks `response.usage` (truthiness)
    mock_completion_no_usage.usage = None

    mock_openai_client.chat.completions.create.return_value = mock_completion_no_usage

    messages = [{"role": "user", "content": "Trigger no usage info"}]
    result = llm.send_message(messages)

    assert isinstance(result, LLMResponse)
    assert result.content == "Content but no usage"
    assert isinstance(result.usage, LLMUsageInfo) # Should fall back to default
    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0
    assert result.usage.cost == 0.0

# To run this file directly for testing:
# pytest tests/test_llm.py
