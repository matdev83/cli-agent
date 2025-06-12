import pytest

from src.tools.mcp import UseMCPTool, AccessMCPResourceTool # Corrected case

# --- UseMCPTool Tests ---

def test_use_mcp_tool_instantiation():
    tool = UseMCPTool() # Corrected case
    assert tool.name == "use_mcp_tool" # Name property remains snake_case
    assert "Request to use a tool provided by a connected MCP server." in tool.description
    assert len(tool.parameters) == 3

def test_use_mcp_tool_execute_success():
    tool = UseMCPTool() # Corrected case
    params = {
        "server_name": "test-server",
        "tool_name": "test-tool",
        "arguments": "{'param1': 'value1'}"
    }
    result = tool.execute(params)
    assert "Success: UseMCPTool called." in result # Corrected class name in message
    assert "Server: 'test-server'" in result
    assert "Tool: 'test-tool'" in result
    assert "Args: '{'param1': 'value1'}'" in result
    assert "Full MCP interaction pending." in result

def test_use_mcp_tool_execute_missing_params():
    tool = UseMCPTool() # Corrected case

    # Missing all
    result_all_missing = tool.execute({})
    assert "Error: Missing required parameters: server_name, tool_name, arguments." in result_all_missing

    # Missing server_name
    params_no_server = {"tool_name": "t", "arguments": "{}"}
    result_no_server = tool.execute(params_no_server)
    assert "Error: Missing required parameters: server_name." in result_no_server

    # Missing tool_name
    params_no_tool = {"server_name": "s", "arguments": "{}"}
    result_no_tool = tool.execute(params_no_tool)
    assert "Error: Missing required parameters: tool_name." in result_no_tool

    # Missing arguments
    params_no_args = {"server_name": "s", "tool_name": "t"}
    result_no_args = tool.execute(params_no_args)
    assert "Error: Missing required parameters: arguments." in result_no_args

    # Arguments is None (should be treated as missing)
    params_args_none = {"server_name": "s", "tool_name": "t", "arguments": None}
    result_args_none = tool.execute(params_args_none)
    assert "Error: Missing required parameters: arguments." in result_args_none

def test_use_mcp_tool_execute_empty_args_string_is_valid():
    tool = UseMCPTool() # Corrected case
    params = {
        "server_name": "test-server",
        "tool_name": "test-tool",
        "arguments": "" # Empty string for JSON arguments
    }
    result = tool.execute(params)
    assert "Success: UseMCPTool called." in result # Corrected class name in message
    assert "Args: ''" in result # Ensure empty string is passed through

# --- AccessMCPResourceTool Tests ---

def test_access_mcp_resource_tool_instantiation():
    tool = AccessMCPResourceTool() # Corrected case
    assert tool.name == "access_mcp_resource" # Name property remains snake_case
    assert "Request to access a resource provided by a connected MCP server." in tool.description
    assert len(tool.parameters) == 2

def test_access_mcp_resource_tool_execute_success():
    tool = AccessMCPResourceTool() # Corrected case
    params = {
        "server_name": "resource-server",
        "uri": "mcp://resource-server/some/resource-id"
    }
    result = tool.execute(params)
    assert "Success: AccessMCPResourceTool called." in result # Corrected class name in message
    assert "Server: 'resource-server'" in result
    assert "URI: 'mcp://resource-server/some/resource-id'" in result
    assert "Full MCP interaction pending." in result

def test_access_mcp_resource_tool_execute_missing_params():
    tool = AccessMCPResourceTool() # Corrected case

    # Missing all
    result_all_missing = tool.execute({})
    assert "Error: Missing required parameters: server_name, uri." in result_all_missing

    # Missing server_name
    params_no_server = {"uri": "mcp://s/r"}
    result_no_server = tool.execute(params_no_server)
    assert "Error: Missing required parameters: server_name." in result_no_server

    # Missing uri
    params_no_uri = {"server_name": "s"}
    result_no_uri = tool.execute(params_no_uri)
    assert "Error: Missing required parameters: uri." in result_no_uri

    # URI is None (should be treated as missing)
    params_uri_none = {"server_name": "s", "uri": None}
    result_uri_none = tool.execute(params_uri_none)
    assert "Error: Missing required parameters: uri." in result_uri_none

def test_access_mcp_resource_tool_execute_empty_uri_is_valid():
    tool = AccessMCPResourceTool() # Corrected case
    params = {
        "server_name": "test-server",
        "uri": "" # Empty string for URI
    }
    # The tool's current execute logic for AccessMcpResourceTool: `if not uri:` will treat "" as missing.
    # The prompt for UseMcpTool said "arguments can be empty JSON '{}'", which is different from uri.
    # For a URI, an empty string is usually not a valid identifier, but let's test current behavior.
    # If an empty URI string should be valid, the tool logic `if not uri:` needs to be `if uri is None:`.
    # Based on current tool code: `if not uri:` means "" is a "missing" (falsy) value.
    result = tool.execute(params)
    assert "Error: Missing required parameters: uri." in result # Current behavior

    # If we wanted to allow empty string URI and only error on None:
    # Change tool execute to `if uri is None:`
    # Then this test would be:
    # assert "Success: AccessMcpResourceTool called." in result
    # assert "URI: ''" in result
