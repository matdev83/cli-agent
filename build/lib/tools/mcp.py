from __future__ import annotations

from typing import Dict, Any, List
from .tool_protocol import Tool

class UseMCPTool(Tool): # Renamed to UseMCPTool
    """A tool to request usage of a tool provided by a connected MCP server."""

    @property
    def name(self) -> str:
        return "use_mcp_tool" # Tool name for LLM remains snake_case

    @property
    def description(self) -> str:
        return (
            "Request to use a tool provided by a connected MCP server. "
            "The agent should specify the server name, the tool name, and the arguments for the tool as a JSON object string."
        )

    @property
    def parameters_schema(self) -> Dict[str, str]:
        return {
            "server_name": "The name of the MCP server providing the tool.",
            "tool_name": "The name of the tool to execute.",
            "arguments": "A JSON object string containing the tool's input parameters."
        }

    def execute(self, params: Dict[str, Any], agent_tools_instance: Any) -> str:
        """
        Executes the tool stub.
        Expects 'server_name', 'tool_name', and 'arguments' in params.
        Full MCP interaction is pending.
        """
        server_name = params.get("server_name")
        tool_name = params.get("tool_name")
        arguments = params.get("arguments") # arguments is a JSON string

        missing = []
        if not server_name:
            missing.append("server_name")
        if not tool_name:
            missing.append("tool_name")
        if arguments is None: # Explicitly check for None, as an empty JSON string "" or "{}" is valid.
            missing.append("arguments")

        if missing:
            return f"Error: Missing required parameters: {', '.join(missing)}."

        return f"Success: UseMCPTool called. Server: '{server_name}', Tool: '{tool_name}', Args: '{arguments}'. Full MCP interaction pending."

class AccessMCPResourceTool(Tool): # Renamed to AccessMCPResourceTool
    """A tool to request access to a resource provided by a connected MCP server."""

    @property
    def name(self) -> str:
        return "access_mcp_resource" # Tool name for LLM remains snake_case

    @property
    def description(self) -> str:
        return (
            "Request to access a resource provided by a connected MCP server. "
            "The agent should specify the server name and the URI of the resource."
        )

    @property
    def parameters_schema(self) -> Dict[str, str]:
        return {
            "server_name": "The name of the MCP server providing the resource.",
            "uri": "The URI identifying the specific resource to access."
        }

    def execute(self, params: Dict[str, Any], agent_tools_instance: Any) -> str:
        """
        Executes the tool stub.
        Expects 'server_name' and 'uri' in params.
        Full MCP interaction is pending.
        """
        server_name = params.get("server_name")
        uri = params.get("uri")

        missing = []
        if not server_name:
            missing.append("server_name")
        if not uri: # URI can be an empty string, but None means it's missing
            missing.append("uri")

        if missing:
            return f"Error: Missing required parameters: {', '.join(missing)}."

        return f"Success: AccessMCPResourceTool called. Server: '{server_name}', URI: '{uri}'. Full MCP interaction pending."
