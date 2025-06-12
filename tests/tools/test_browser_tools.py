import pytest
import json
from unittest.mock import patch, MagicMock

from src.tools.browser import BrowserActionTool

# Dummy AgentMemory class for testing
class MockAgentMemory:
    def __init__(self, cwd: str = "/app"):
        self.cwd = cwd
        self.playwright_sync_instance = None
        self.playwright_browser_context = None
        self.playwright_page = None

# --- BrowserActionTool Tests ---

def test_browser_action_tool_instantiation():
    tool = BrowserActionTool()
    assert tool.name == "browser_action"
    assert "Perform an action in a web browser" in tool.description
    assert len(tool.parameters) == 4 # action, url, coordinate, text

def test_browser_action_missing_action_parameter():
    tool = BrowserActionTool()
    agent_memory = MockAgentMemory()
    result_str = tool.execute({}, agent_memory=agent_memory)
    result = json.loads(result_str)
    assert result["status"] == "error"
    assert "Missing required parameter 'action'" in result["message"]

# --- Test 'launch' action ---
@patch('src.tools.browser.sync_playwright') # Patched where it's looked up
def test_browser_action_launch_success(mock_sync_playwright, tmp_path):
    agent_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = BrowserActionTool()

    # Mock Playwright's internal structure
    mock_browser_instance = MagicMock(name="BrowserInstance")
    mock_context_instance = MagicMock(name="ContextInstance")
    mock_page_instance = MagicMock(name="PageInstance")

    mock_ps_instance = MagicMock(name="PlaywrightInstance")
    # Let chromium be an auto-created MagicMock by mock_ps_instance
    # And set the return_value of its launch method
    mock_ps_instance.chromium.launch.return_value = mock_browser_instance

    # sync_playwright() returns a PlaywrightContextManager
    # PlaywrightContextManager.start() returns a Playwright instance (mock_ps_instance)
    mock_playwright_context_manager = MagicMock(name="PlaywrightContextManager")
    mock_sync_playwright.return_value = mock_playwright_context_manager
    mock_playwright_context_manager.start.return_value = mock_ps_instance

    mock_browser_instance.new_context.return_value = mock_context_instance
    mock_context_instance.new_page.return_value = mock_page_instance

    test_url = "http://example.com"
    params = {"action": "launch", "url": test_url}
    result_str = tool.execute(params, agent_memory=agent_memory)
    result = json.loads(result_str)

    assert result["status"] == "success"
    assert f"Browser launched at {test_url}" in result["message"]
    mock_ps_instance.chromium.launch.assert_called_once_with(headless=True) # Assert on the auto-mocked launch
    mock_browser_instance.new_context.assert_called_once_with(
        viewport={'width': BrowserActionTool.VIEWPORT_WIDTH, 'height': BrowserActionTool.VIEWPORT_HEIGHT}
    )
    mock_context_instance.new_page.assert_called_once()
    mock_page_instance.goto.assert_called_once_with(test_url)

    assert agent_memory.playwright_sync_instance == mock_ps_instance
    assert agent_memory.playwright_browser_context == mock_context_instance
    assert agent_memory.playwright_page == mock_page_instance

def test_browser_action_launch_missing_url():
    tool = BrowserActionTool()
    agent_memory = MockAgentMemory()
    params = {"action": "launch"}
    result_str = tool.execute(params, agent_memory=agent_memory)
    result = json.loads(result_str)
    assert result["status"] == "error"
    assert "Missing 'url' for 'launch' action" in result["message"]

# --- Test 'close' action ---
@patch('src.tools.browser.sync_playwright') # Patched where it's looked up (though not strictly needed for close if launch is mocked)
def test_browser_action_close_success(mock_sync_playwright_unused, tmp_path):
    agent_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = BrowserActionTool()

    # Setup mock objects in agent_memory to simulate an open browser
    mock_context_to_close = MagicMock(name="ContextToClose")
    mock_sync_instance_to_stop = MagicMock(name="SyncInstanceToStop")
    agent_memory.playwright_browser_context = mock_context_to_close
    agent_memory.playwright_sync_instance = mock_sync_instance_to_stop
    agent_memory.playwright_page = MagicMock() # Page isn't directly used by close, but good to have

    params = {"action": "close"}
    result_str = tool.execute(params, agent_memory=agent_memory)
    result = json.loads(result_str)

    assert result["status"] == "success"
    assert "Browser closed" in result["message"]

    mock_context_to_close.close.assert_called_once()
    mock_sync_instance_to_stop.stop.assert_called_once()

    assert agent_memory.playwright_sync_instance is None
    assert agent_memory.playwright_browser_context is None
    assert agent_memory.playwright_page is None


def test_browser_action_close_when_nothing_open():
    tool = BrowserActionTool()
    agent_memory = MockAgentMemory() # No browser state
    params = {"action": "close"}
    result_str = tool.execute(params, agent_memory=agent_memory)
    result = json.loads(result_str)
    assert result["status"] == "success" # Should still succeed, just does nothing
    assert "Browser closed" in result["message"]


# --- Tests for actions requiring a page ---
def test_actions_require_page_not_launched():
    tool = BrowserActionTool()
    agent_memory = MockAgentMemory() # No page set
    actions_to_test = ["click", "type", "scroll_down", "scroll_up"]

    for action in actions_to_test:
        params = {"action": action}
        # Add dummy required params if any, to pass initial checks for those
        if action == "click":
            params["coordinate"] = "1,1"
        elif action == "type":
            params["text"] = "test"

        result_str = tool.execute(params, agent_memory=agent_memory)
        result = json.loads(result_str)
        assert result["status"] == "error"
        assert "Browser not launched or page not available" in result["message"]

@patch('src.tools.browser.sync_playwright') # Patched where it's looked up (though less critical here if page is directly mocked)
def test_browser_action_click_success(mock_sync_playwright_unused):
    tool = BrowserActionTool()
    agent_memory = MockAgentMemory()
    mock_page = MagicMock()
    agent_memory.playwright_page = mock_page

    params = {"action": "click", "coordinate": "123,456"}
    result_str = tool.execute(params, agent_memory=agent_memory)
    result = json.loads(result_str)

    assert result["status"] == "success"
    assert "Clicked at 123,456" in result["message"]
    mock_page.mouse.click.assert_called_once_with(123, 456)

def test_browser_action_click_missing_coordinate():
    tool = BrowserActionTool()
    agent_memory = MockAgentMemory()
    agent_memory.playwright_page = MagicMock() # Page exists
    params = {"action": "click"}
    result_str = tool.execute(params, agent_memory=agent_memory)
    result = json.loads(result_str)
    assert result["status"] == "error"
    assert "Missing 'coordinate' for 'click' action" in result["message"]

def test_browser_action_click_invalid_coordinate():
    tool = BrowserActionTool()
    agent_memory = MockAgentMemory()
    agent_memory.playwright_page = MagicMock()
    params = {"action": "click", "coordinate": "123;456"} # Invalid format
    result_str = tool.execute(params, agent_memory=agent_memory)
    result = json.loads(result_str)
    assert result["status"] == "error"
    assert "Invalid coordinate format" in result["message"]


@patch('src.tools.browser.sync_playwright') # Patched where it's looked up
def test_browser_action_type_success(mock_sync_playwright_unused):
    tool = BrowserActionTool()
    agent_memory = MockAgentMemory()
    mock_page = MagicMock()
    agent_memory.playwright_page = mock_page

    test_text = "Hello, world!"
    params = {"action": "type", "text": test_text}
    result_str = tool.execute(params, agent_memory=agent_memory)
    result = json.loads(result_str)

    assert result["status"] == "success"
    assert f"Typed text: '{test_text}'" in result["message"]
    mock_page.keyboard.type.assert_called_once_with(test_text)

def test_browser_action_type_missing_text():
    tool = BrowserActionTool()
    agent_memory = MockAgentMemory()
    agent_memory.playwright_page = MagicMock()
    params = {"action": "type"}
    result_str = tool.execute(params, agent_memory=agent_memory)
    result = json.loads(result_str)
    assert result["status"] == "error"
    assert "Missing 'text' for 'type' action" in result["message"]

@patch('src.tools.browser.sync_playwright') # Patched where it's looked up
def test_browser_action_scroll_down_success(mock_sync_playwright_unused):
    tool = BrowserActionTool()
    agent_memory = MockAgentMemory()
    mock_page = MagicMock()
    agent_memory.playwright_page = mock_page

    params = {"action": "scroll_down"}
    result_str = tool.execute(params, agent_memory=agent_memory)
    result = json.loads(result_str)

    assert result["status"] == "success"
    assert "Scrolled down" in result["message"]
    mock_page.evaluate.assert_called_once_with("window.scrollBy(0, window.innerHeight)")

@patch('src.tools.browser.sync_playwright') # Patched where it's looked up
def test_browser_action_scroll_up_success(mock_sync_playwright_unused):
    tool = BrowserActionTool()
    agent_memory = MockAgentMemory()
    mock_page = MagicMock()
    agent_memory.playwright_page = mock_page

    params = {"action": "scroll_up"}
    result_str = tool.execute(params, agent_memory=agent_memory)
    result = json.loads(result_str)

    assert result["status"] == "success"
    assert "Scrolled up" in result["message"]
    mock_page.evaluate.assert_called_once_with("window.scrollBy(0, -window.innerHeight)")

def test_browser_action_unknown_action():
    tool = BrowserActionTool()
    agent_memory = MockAgentMemory()
    agent_memory.playwright_page = MagicMock() # Page exists
    params = {"action": "fly_to_moon"}
    result_str = tool.execute(params, agent_memory=agent_memory)
    result = json.loads(result_str)
    assert result["status"] == "error"
    assert "Unknown browser action: 'fly_to_moon'" in result["message"]

# Test for launch cleaning up old state
@patch('src.tools.browser.sync_playwright') # Patched where it's looked up
def test_browser_action_launch_cleans_up_old_state(mock_sync_playwright, tmp_path):
    agent_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = BrowserActionTool()

    # Pre-populate agent_memory with old mocks
    old_mock_ps_instance = MagicMock(name="old_ps_instance")
    old_mock_context_instance = MagicMock(name="old_context_instance")
    agent_memory.playwright_sync_instance = old_mock_ps_instance
    agent_memory.playwright_browser_context = old_mock_context_instance
    agent_memory.playwright_page = MagicMock(name="old_page_instance")


    # New mocks for the new launch
    mock_browser_instance = MagicMock(name="new_browser_instance")
    mock_context_instance = MagicMock(name="new_context_instance")
    mock_page_instance = MagicMock(name="new_page_instance")

    mock_ps_instance = MagicMock(name="new_ps_instance")
    # Let chromium be an auto-created MagicMock by mock_ps_instance for the new launch
    # And set the return_value of its launch method
    mock_ps_instance.chromium.launch.return_value = mock_browser_instance

    # Corrected mocking for the new launch
    mock_playwright_context_manager = MagicMock(name="PlaywrightContextManager")
    mock_sync_playwright.return_value = mock_playwright_context_manager
    mock_playwright_context_manager.start.return_value = mock_ps_instance

    mock_browser_instance.new_context.return_value = mock_context_instance
    mock_context_instance.new_page.return_value = mock_page_instance

    test_url = "http://example.new.com"
    params = {"action": "launch", "url": test_url}
    result_str = tool.execute(params, agent_memory=agent_memory)
    result = json.loads(result_str)

    assert result["status"] == "success"

    # Assert old context and sync instance were closed
    old_mock_context_instance.close.assert_called_once()
    old_mock_ps_instance.stop.assert_called_once()

    # Assert new instances are stored
    assert agent_memory.playwright_sync_instance == mock_ps_instance
    assert agent_memory.playwright_browser_context == mock_context_instance
    assert agent_memory.playwright_page == mock_page_instance
    mock_ps_instance.chromium.launch.assert_called_once_with(headless=True) # Check launch on the auto-mock
    mock_page_instance.goto.assert_called_once_with(test_url)

# Test general error handling in execute
@patch.object(BrowserActionTool, '_get_playwright_page', side_effect=Exception("Generic Playwright Error"))
def test_browser_action_generic_error_handling(mock_get_page):
    tool = BrowserActionTool()
    agent_memory = MockAgentMemory()
    # Simulate that a page exists to get past the initial check
    # The error will be raised when _get_playwright_page is called internally by an action like 'click'
    # However, the current structure calls _get_playwright_page before dispatching.
    # Let's test an action that needs a page, like 'click'
    # To make it simpler, let's assume the error happens during an action like page.mouse.click

    # We need to mock the actual playwright call that fails
    # Let's mock the 'click' action's page call specifically
    # The _get_playwright_page is mocked to raise an exception.
    # This exception should be caught by the main try-except in execute.

    # No need to set agent_memory.playwright_page if _get_playwright_page itself is mocked to fail
    # agent_memory.playwright_page = mock_page_for_error

    # Store the mocks for context and sync_instance that might be on agent_memory
    # for the cleanup check, even if _get_playwright_page fails early.
    # The tool's execute() method's except block will try to clean these up.
    mock_context_on_agent = MagicMock(name="ContextOnAgentForCleanup")
    mock_sync_on_agent = MagicMock(name="SyncInstanceOnAgentForCleanup")
    agent_memory.playwright_browser_context = mock_context_on_agent
    agent_memory.playwright_sync_instance = mock_sync_on_agent


    params = {"action": "click", "coordinate": "1,1"} # Action that uses the page
    result_str = tool.execute(params, agent_memory=agent_memory)
    result = json.loads(result_str)

    assert result["status"] == "error"
    assert "An error occurred: Generic Playwright Error" in result["message"] # Error from the mocked _get_playwright_page

    # Check if cleanup was attempted on the (potentially existing) context and sync instance
    mock_context_on_agent.close.assert_called_once()
    mock_sync_on_agent.stop.assert_called_once()
    assert agent_memory.playwright_page is None # State should be cleared by the handler
    assert agent_memory.playwright_browser_context is None
    assert agent_memory.playwright_sync_instance is None

    # Restore the original method if needed for other tests, though pytest isolates tests.
    # BrowserActionTool._get_playwright_page = _get_playwright_page_original (if saved)
    # This is not needed due to pytest's test isolation.
