import os
import platform
from src.prompts.system import get_system_prompt
from src.tools.browser import BrowserActionTool # Import BrowserActionTool


def test_placeholder_replacement(monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(os.path, "expanduser", lambda x: "/home/test")
    prompt = get_system_prompt(tools=[], cwd="/work") # Added tools=[]
    assert "/work" in prompt
    assert "Darwin" in prompt
    assert "/bin/zsh" in prompt
    assert "/home/test" in prompt


def test_browser_sections(monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/sh")
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    # Create a dummy BrowserActionTool instance for testing its description rendering
    # The viewport size in the description comes from the tool's class attributes.
    dummy_browser_tool = BrowserActionTool()
    # The tool's description contains: f"The browser viewport will be {self.VIEWPORT_WIDTH}x{self.VIEWPORT_HEIGHT}."
    # which are 1280x1024 by default in the tool.
    # The browser_settings passed to get_system_prompt are not directly used for this phrase,
    # but the tool's own description (which generate_tools_documentation uses) is.

    prompt_no = get_system_prompt(tools=[], cwd="/work", supports_browser_use=False)
    # For prompt_yes, include the dummy_browser_tool to test its description rendering
    prompt_yes = get_system_prompt(tools=[dummy_browser_tool], cwd="/work", supports_browser_use=True, browser_settings={"viewport": {"width": 1280, "height": 1024}})

    # Test the generic phrase about browser usage enabled by supports_browser_use
    generic_browser_phrase = "You can use the browser_action tool to interact with websites."
    assert generic_browser_phrase not in prompt_no
    assert generic_browser_phrase in prompt_yes

    # Test that the viewport size from BrowserActionTool's description is in the prompt
    # when the tool is included and supports_browser_use is True.
    # The default viewport in BrowserActionTool is 1280x1024.
    viewport_phrase = f"{BrowserActionTool.VIEWPORT_WIDTH}x{BrowserActionTool.VIEWPORT_HEIGHT}"
    assert viewport_phrase not in prompt_no # Should not be there if tool not included or support_browser_use=False
    assert viewport_phrase in prompt_yes # Should be there due to tool's description

