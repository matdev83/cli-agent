import pytest
import platform
import os
from typing import List, Dict, Any

from src.prompts.system import get_system_prompt, generate_tools_documentation
from src.tools.tool_protocol import Tool

# Define a reusable MockTool class for these tests
class MockPromptTool(Tool):
    def __init__(self, name: str, description: str, parameters: List[Dict[str, str]]):
        self._name = name
        self._description = description
        self._parameters = parameters

    @property
    def name(self) -> str: return self._name
    @property
    def description(self) -> str: return self._description
    @property
    def parameters(self) -> List[Dict[str, str]]: return self._parameters

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        # Not called in prompt generation tests
        raise NotImplementedError

# --- Tests for generate_tools_documentation ---

def test_generate_tools_documentation_empty():
    docs = generate_tools_documentation([], "/test/cwd")
    assert docs.strip() == "" # Empty list of tools should produce an empty string

def test_generate_tools_documentation_single_tool_no_params():
    tool1 = MockPromptTool(name="tool_one", description="Tool one does stuff.", parameters=[])
    docs = generate_tools_documentation([tool1], "/test/cwd")
    assert "## tool_one" in docs
    assert "Description: Tool one does stuff." in docs
    assert "Parameters: None" in docs
    assert "<tool_one>\n\n</tool_one>" in docs # Usage block for no params

def test_generate_tools_documentation_single_tool_with_params_cwd_replacement():
    # Test CWD replacement for old style placeholders in descriptions, as current generate_tools_documentation handles it.
    tool1 = MockPromptTool(
        name="tool_two",
        description="Tool two with params from ${cwd}.", # Old style placeholder
        parameters=[
            {"name": "param1", "description": "First param.", "type": "string", "required": True},
            {"name": "param2", "description": "Second param from ${cwd.toPosix()}.", "type": "int", "required": False} # Old style
        ]
    )
    test_cwd_val = "/test/cwd"
    docs = generate_tools_documentation([tool1], test_cwd_val)

    assert "## tool_two" in docs
    assert f"Description: Tool two with params from {test_cwd_val}." in docs # CWD replaced by generate_tools_documentation
    assert "Parameters:" in docs
    assert "- param1: (string, (required)) First param." in docs
    # CWD replaced by generate_tools_documentation for param description
    assert f"- param2: (int, (optional)) Second param from {test_cwd_val}." in docs
    assert "<tool_two>" in docs
    assert "<param1>string</param1>" in docs
    assert "<param2>int</param2>" in docs
    assert "</tool_two>" in docs

def test_generate_tools_documentation_parameter_formatting():
    tool_params = [
        {"name": "p_string", "description": "A string.", "type": "string", "required": True},
        {"name": "p_int_opt", "description": "An int.", "type": "int"}, # required defaults to False
        {"name": "p_bool", "description": "A bool.", "type": "boolean", "required": True},
        {"name": "p_no_type", "description": "No type specified.", "required": False},
    ]
    tool = MockPromptTool(name="complex_tool", description="Desc.", parameters=tool_params)
    docs = generate_tools_documentation([tool], "/cwd")

    assert "- p_string: (string, (required)) A string." in docs
    assert "- p_int_opt: (int, (optional)) An int." in docs
    assert "- p_bool: (boolean, (required)) A bool." in docs
    assert "- p_no_type: (string, (optional)) No type specified." in docs # Defaults to string type if missing

    assert "<p_string>string</p_string>" in docs
    assert "<p_int_opt>int</p_int_opt>" in docs
    assert "<p_bool>boolean</p_bool>" in docs
    assert "<p_no_type>string</p_no_type>" in docs # Defaults to string for usage example type

# --- Tests for get_system_prompt (using Jinja2) ---

def test_get_system_prompt_basic_substitutions():
    mock_tools = [MockPromptTool("fake_tool", "desc for {{ cwd }}", [])] # Tool desc with Jinja style CWD
    test_cwd = "/projects/my_agent"
    prompt = get_system_prompt(tools=mock_tools, cwd=test_cwd)

    # Check Jinja substitutions
    assert f"Your current working directory is: {test_cwd}" in prompt
    assert f"Operating System: {platform.system()}" in prompt
    assert f"Default Shell: {os.environ.get('SHELL', 'sh')}" in prompt
    assert f"Home Directory: {os.path.expanduser('~')}" in prompt

    # Check tool documentation inclusion (Jinja variable in tool description)
    assert "## fake_tool" in prompt
    assert f"desc for {test_cwd}" in prompt # Jinja should render {{ cwd }} within the tool_documentation block

def test_get_system_prompt_browser_support_conditional():
    mock_tools = []
    # Test when supports_browser_use is False
    prompt_no_browser = get_system_prompt(tools=mock_tools, cwd="/cwd", supports_browser_use=False)
    # Check that the specific capability text for browser is NOT there
    assert "You can use the browser_action tool to interact with websites." not in prompt_no_browser
    # Check that browser-specific rules section is NOT there (if distinct in template)
    # The Jinja template uses `{% if supports_browser_use %}` blocks.

    # Test when supports_browser_use is True
    prompt_with_browser = get_system_prompt(tools=mock_tools, cwd="/cwd", supports_browser_use=True)
    assert "You can use the browser_action tool to interact with websites." in prompt_with_browser

def test_get_system_prompt_mcp_servers_documentation():
    mock_tools = []
    custom_mcp_docs = "### Server XYZ\n- Status: connected."
    prompt = get_system_prompt(tools=mock_tools, cwd="/cwd", mcp_servers_documentation=custom_mcp_docs)
    assert custom_mcp_docs in prompt

    default_mcp_docs = "(No MCP servers currently connected)"
    prompt_default = get_system_prompt(tools=mock_tools, cwd="/cwd") # Uses default mcp_servers_documentation
    assert default_mcp_docs in prompt_default

def test_get_system_prompt_with_multiple_tools():
    tool1 = MockPromptTool("tool_alpha", "Alpha tool.", [])
    tool2 = MockPromptTool("tool_beta", "Beta tool with param {{ cwd }}.", [{"name": "p1", "type":"str", "description":"P1", "required":True}])
    mock_tools = [tool1, tool2]
    test_cwd = "/beta/test"
    prompt = get_system_prompt(tools=mock_tools, cwd=test_cwd)

    assert "## tool_alpha" in prompt
    assert "Alpha tool." in prompt
    assert "## tool_beta" in prompt
    assert f"Beta tool with param {test_cwd}." in prompt # Jinja replacement for {{ cwd }}
    assert "- p1: (str, (required)) P1" in prompt
```
