from __future__ import annotations

import ast
import json # For returning structured output
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from .tool_protocol import Tool

# --- Helper function for path resolution (similar to one in file.py) ---
def _resolve_path(path_str: str, agent_memory: Any = None) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()

    base_dir = Path(os.getcwd()) # Default base
    if agent_memory and hasattr(agent_memory, 'cwd') and agent_memory.cwd:
        base_dir = Path(agent_memory.cwd)

    return (base_dir / p).resolve()

# --- Helper function for ListCodeDefinitionsTool ---
def _python_top_level_defs(file_path: Path) -> List[str]:
    """Extracts top-level function and class definition lines from a Python file."""
    try:
        source = file_path.read_text(encoding="utf-8")
        module = ast.parse(source)
    except SyntaxError:
        return [f"Error: Syntax error in {file_path.name}"]
    except Exception: # Catch other read errors, e.g. if it's not a text file
        return [f"Error: Could not read or parse {file_path.name}"]

    lines = source.splitlines()
    defs = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Get the line of the definition. AST lineno is 1-indexed.
            try:
                # Ensure node.lineno is valid
                if 0 < node.lineno <= len(lines):
                    line = lines[node.lineno - 1].strip()
                    # Add a prefix like "def " or "class " if not present, common in some outputs
                    # However, the original code just took the line, so we stick to that.
                    defs.append(f"| {line}") # Using a slightly different marker for clarity
                else: # Should not happen with valid AST and source
                    defs.append(f"| Error: Invalid line number {node.lineno} for a definition in {file_path.name}")

            except IndexError: # Should not happen if node.lineno is valid
                 defs.append(f"| Error: Could not retrieve line for a definition in {file_path.name}")
    return defs

class ListCodeDefinitionsTool(Tool):
    @property
    def name(self) -> str:
        return "list_code_definitions"

    @property
    def description(self) -> str:
        return ("Lists top-level function and class definition lines from Python files "
                "within a specified directory. Returns a JSON string list.")

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {
                "name": "directory_path",
                "description": "The relative or absolute path to the directory to scan for Python files.",
                "type": "string",
                "required": True
            }
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        dir_path_str = params.get("directory_path")
        if not dir_path_str:
            return json.dumps({"error": "Missing required parameter 'directory_path'."})

        try:
            abs_dir_path = _resolve_path(dir_path_str, agent_memory)

            if not abs_dir_path.exists():
                return json.dumps({"error": f"Directory not found at {str(abs_dir_path)}"})
            if not abs_dir_path.is_dir():
                return json.dumps({"error": f"Path {str(abs_dir_path)} is not a directory."})

            output_structure = [] # List of dicts: {"file": "name", "definitions": []} or similar

            found_defs = False
            # Track if any Python files were found at all
            found_py_files = False
            for entry in sorted(abs_dir_path.iterdir()):
                if entry.is_file() and entry.suffix == ".py":
                    found_py_files = True
                    defs = _python_top_level_defs(entry)
                    if defs: # Only add if there are definitions or errors from parsing
                        output_structure.append({
                            "file": entry.name,
                            "definitions": defs # These already include error messages if any
                        })
                        # Consider it found_defs if at least one definition is not an error
                        if any(not d.startswith("Error:") and not d.startswith("| Error:") for d in defs):
                            found_defs = True

            if not found_py_files: # No .py files found
                 return json.dumps({"message": "No Python files found in the directory."}) # More specific message

            if not output_structure and found_py_files: # .py files found, but no definitions extracted (e.g. all empty or no classes/funcs)
                 return json.dumps({"message": "No definitions found in Python files."})


            if not found_defs and any("Error:" in d for item in output_structure for d in item["definitions"]):
                 # Only errors were found during parsing of some files
                 return json.dumps({"results": output_structure, "message": "Found Python files, but encountered errors while parsing definitions."})

            # If found_defs is true, or if there are no parsing errors in any file.
            return json.dumps({"results": output_structure, "message": "Successfully listed code definitions." if found_defs else "No definitions found in Python files."})

        except Exception as e:
            return json.dumps({"error": f"Error listing code definitions in {dir_path_str}: {e}"})


class BrowserActionTool(Tool):
    @property
    def name(self) -> str:
        return "browser_action"

    @property
    def description(self) -> str:
        return "STUB - Performs an action in a web browser. (Not Implemented)"

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {"name": "action", "description": "The browser action to perform (e.g., 'open_url', 'search').", "type": "string", "required": True},
            {"name": "action_params", "description": "Parameters for the specified browser action.", "type": "dict", "required": False}
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        raise NotImplementedError(f"{self.name} is not implemented yet.")


class UseMCPTool(Tool):
    @property
    def name(self) -> str:
        return "use_mcp_tool"

    @property
    def description(self) -> str:
        return "STUB - Interacts with an MCP tool. (Not Implemented)"

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {"name": "tool_name", "description": "The name of the MCP tool to use.", "type": "string", "required": True},
            {"name": "tool_inputs", "description": "Inputs for the MCP tool.", "type": "dict", "required": False}
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        raise NotImplementedError(f"{self.name} is not implemented yet.")


class AccessMCPResourceTool(Tool):
    @property
    def name(self) -> str:
        return "access_mcp_resource"

    @property
    def description(self) -> str:
        return "STUB - Accesses an MCP resource. (Not Implemented)"

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {"name": "resource_id", "description": "The ID of the MCP resource to access.", "type": "string", "required": True},
            {"name": "access_params", "description": "Parameters for accessing the resource.", "type": "dict", "required": False}
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        raise NotImplementedError(f"{self.name} is not implemented yet.")

# --- Wrapper function for old tests ---
def list_code_definition_names(directory_path: str, agent_memory: Any = None) -> str:
    tool = ListCodeDefinitionsTool()
    result_str = tool.execute({"directory_path": directory_path}, agent_memory=agent_memory)
    data = json.loads(result_str)

    if "error" in data:
        # Consider how to represent this; the old test might not expect an error for "not found"
        # but the tool gives one. For "No source code definitions found.", the test is specific.
        if "Directory not found" in data["error"]:
             # This case might map to "No source code definitions found." if the dir simply doesn't exist.
             return "No source code definitions found."
        raise RuntimeError(f"Tool error: {data['error']}")

    if "message" in data and (
        data["message"] == "No Python files found in the directory." or
        data["message"] == "No definitions found in Python files." or # Covers case where .py files are empty/no defs
        data["message"] == "No Python files with definitions found or directory is empty." # Original tool message
    ):
        return "No source code definitions found."

    if "results" in data and not data["results"]: # Empty results list
        return "No source code definitions found."

    if "results" in data:
        output_lines = []
        for file_info in data["results"]:
            output_lines.append(file_info["file"])
            output_lines.append("|----") # Test expects this separator
            for definition in file_info["definitions"]:
                # The tool already formats definitions with "| "
                output_lines.append(definition)
            output_lines.append("|----") # Test expects this separator at the end of each file block
        return "\n".join(output_lines)

    # Fallback or if the JSON structure is unexpected
    return "Error: Could not parse tool output as expected."
