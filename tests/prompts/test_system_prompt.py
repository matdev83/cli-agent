import platform
import os
from typing import Dict, Any

from src.prompts.system import get_system_prompt, generate_tools_documentation
from src.tools.tool_protocol import Tool

# Define a reusable MockTool class for these tests
class MockPromptTool(Tool):
    def __init__(self, name: str, description: str, parameters_schema: Dict[str, str]): # Updated
        self._name = name
        self._description = description
        self._parameters_schema = parameters_schema # Updated

    @property
    def name(self) -> str: return self._name
    @property
    def description(self) -> str: return self._description
    @property
    def parameters_schema(self) -> Dict[str, str]: return self._parameters_schema # Updated

    def execute(self, params: Dict[str, Any], agent_tools_instance: Any) -> str: # Updated
        # Not called in prompt generation tests
        raise NotImplementedError

# --- Tests for generate_tools_documentation ---

def test_generate_tools_documentation_empty():
    docs = generate_tools_documentation([], "/test/cwd")
    assert docs.strip() == "" # Empty list of tools should produce an empty string

def test_generate_tools_documentation_single_tool_no_params():
    tool1 = MockPromptTool(name="tool_one", description="Tool one does stuff.", parameters_schema={}) # Updated
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
        parameters_schema={ # Updated
            "param1": "First param.",
            "param2": "Second param from ${cwd.toPosix()}."
        }
    )
    test_cwd_val = "/test/cwd"
    docs = generate_tools_documentation([tool1], test_cwd_val)

    assert "## tool_two" in docs
    assert f"Description: Tool two with params from {test_cwd_val}." in docs # CWD replaced by generate_tools_documentation
    assert "Parameters:" in docs
    # Assertions updated to reflect default type "string" and "(optional)" from _format_parameter
    assert "- param1: (string, (optional)) First param." in docs
    # CWD replaced by generate_tools_documentation for param description
    assert f"- param2: (string, (optional)) Second param from {test_cwd_val}." in docs
    assert "<tool_two>" in docs
    assert "<param1>string</param1>" in docs # Usage part defaults to string
    assert "<param2>string</param2>" in docs # Usage part defaults to string
    assert "</tool_two>" in docs

def test_generate_tools_documentation_parameter_formatting():
    tool_params_schema = { # Updated
        "p_string": "A string.",
        "p_int_opt": "An int.",
        "p_bool": "A bool.",
        "p_no_type": "No type specified."
    }
    # Note: 'type' and 'required' are no longer part of parameters_schema directly.
    # _format_parameter defaults them to "string" and "(optional)".
    tool = MockPromptTool(name="complex_tool", description="Desc.", parameters_schema=tool_params_schema) # Updated
    docs = generate_tools_documentation([tool], "/cwd")

    assert "- p_string: (string, (optional)) A string." in docs
    assert "- p_int_opt: (string, (optional)) An int." in docs
    assert "- p_bool: (string, (optional)) A bool." in docs
    assert "- p_no_type: (string, (optional)) No type specified." in docs

    assert "<p_string>string</p_string>" in docs
    assert "<p_int_opt>string</p_int_opt>" in docs # Type in usage defaults to string
    assert "<p_bool>string</p_bool>" in docs     # Type in usage defaults to string
    assert "<p_no_type>string</p_no_type>" in docs

# --- Tests for get_system_prompt (using Jinja2) ---

def test_get_system_prompt_basic_substitutions():
    mock_tools = [MockPromptTool("fake_tool", "desc for {{ cwd }}", {})] # Updated & Tool desc with Jinja style CWD
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
    mock_tools = [] # No tools needed, just testing conditional sections
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
    mock_tools = [] # No tools needed
    custom_mcp_docs = "### Server XYZ\n- Status: connected."
    prompt = get_system_prompt(tools=mock_tools, cwd="/cwd", mcp_servers_documentation=custom_mcp_docs)
    assert custom_mcp_docs in prompt

    default_mcp_docs = "(No MCP servers currently connected)"
    prompt_default = get_system_prompt(tools=mock_tools, cwd="/cwd") # Uses default mcp_servers_documentation
    assert default_mcp_docs in prompt_default

def test_get_system_prompt_with_multiple_tools():
    tool1 = MockPromptTool("tool_alpha", "Alpha tool.", {}) # Updated
    tool2 = MockPromptTool("tool_beta", "Beta tool with param {{ cwd }}.", {"p1": "P1"}) # Updated
    mock_tools = [tool1, tool2]
    test_cwd = "/beta/test"
    prompt = get_system_prompt(tools=mock_tools, cwd=test_cwd)

    assert "## tool_alpha" in prompt
    assert "Alpha tool." in prompt
    assert "## tool_beta" in prompt
    assert f"Beta tool with param {test_cwd}." in prompt # Jinja replacement for {{ cwd }}
    assert "- p1: (string, (optional)) P1" in prompt # Updated assertion
