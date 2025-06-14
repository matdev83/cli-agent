from __future__ import annotations

import ast
import json  # For returning structured output
import os
import re
from pathlib import Path
from typing import List, Dict, Any

from .tool_protocol import Tool

# --- Helper function for path resolution (similar to one in file.py) ---
def _resolve_path(path_str: str, agent_tools_instance: Any = None) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()

    base_dir = Path(os.getcwd()) # Default base
    if agent_tools_instance and hasattr(agent_tools_instance, 'cwd') and agent_tools_instance.cwd:
        base_dir = Path(agent_tools_instance.cwd)

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

# --- JavaScript/TypeScript support ---
def _javascript_top_level_defs(file_path: Path) -> List[str]:
    """Extracts top-level function and class definitions from a JS/TS file."""
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return [f"Error: Could not read {file_path.name}"]

    defs = []
    pattern_func = re.compile(r"^\s*(export\s+)?(async\s+)?function\s+\w+")
    pattern_class = re.compile(r"^\s*(export\s+)?class\s+\w+")
    pattern_arrow = re.compile(r"^\s*(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s*)?\(?")

    for line in lines:
        stripped = line.rstrip()
        if pattern_func.match(stripped) or pattern_class.match(stripped) or pattern_arrow.match(stripped):
            defs.append(f"| {stripped.strip()}")

    return defs

class ListCodeDefinitionNamesTool(Tool):
    """A tool to list top-level definitions (functions and classes) from source files."""
    @property
    def name(self) -> str:
        return "list_code_definition_names"
    @property
    def description(self) -> str:
        return (
            "Lists top-level function and class definition lines from Python and JavaScript/TypeScript "
            "source files in the specified directory. Returns a JSON string detailing definitions per file."
        )
    @property
    def parameters_schema(self) -> Dict[str, str]:
        return {
            "path": "The path of the directory to list top level source code definitions for."
        }

    def execute(self, params: Dict[str, Any], agent_tools_instance: Any) -> str:
        """Executes the tool to find top-level code definitions. Expects 'path' in params."""
        path_str = params.get("path")
        if not path_str:
            return json.dumps({"error": "Missing required parameter 'path'."})

        try:
            abs_dir_path = _resolve_path(path_str, agent_tools_instance)

            if not abs_dir_path.exists():
                return json.dumps({"error": f"Directory not found at {str(abs_dir_path)}"})
            if not abs_dir_path.is_dir():
                return json.dumps({"error": f"Path {str(abs_dir_path)} is not a directory."})

            output_structure = []

            found_defs = False
            found_supported_files = False

            suffix_map = {
                ".py": _python_top_level_defs,
                ".js": _javascript_top_level_defs,
                ".ts": _javascript_top_level_defs,
            }

            for entry in sorted(abs_dir_path.iterdir()):
                if entry.is_file() and entry.suffix in suffix_map:
                    found_supported_files = True
                    defs = suffix_map[entry.suffix](entry)
                    if defs:
                        output_structure.append({
                            "file": entry.name,
                            "definitions": defs,
                        })
                        if any(not d.startswith("Error:") and not d.startswith("| Error:") for d in defs):
                            found_defs = True

            if not found_supported_files:
                return json.dumps({"message": "No supported source files found in the directory."})

            if not output_structure:
                return json.dumps({"message": "No definitions found in source files."})


            if not found_defs and any("Error:" in d for item in output_structure for d in item["definitions"]):
                 # Only errors were found during parsing of some files
                 return json.dumps({"results": output_structure, "message": "Found source files, but encountered errors while parsing definitions."})

            # If found_defs is true, or if there are no parsing errors in any file.
            return json.dumps({"results": output_structure, "message": "Successfully listed code definitions." if found_defs else "No definitions found in source files."})

        except Exception as e:
            return json.dumps({"error": f"Error listing code definitions in {path_str}: {e}"})

# --- Wrapper function for old tests ---
def list_code_definition_names(directory_path: str, agent_tools_instance: Any = None) -> str: # Renamed parameter
    tool = ListCodeDefinitionNamesTool()
    result_str = tool.execute({"path": directory_path}, agent_tools_instance=agent_tools_instance) # Pass updated param
    data = json.loads(result_str)

    if "error" in data:
        # Consider how to represent this; the old test might not expect an error for "not found"
        # but the tool gives one. For "No source code definitions found.", the test is specific.
        if "Directory not found" in data["error"]:
             # This case might map to "No source code definitions found." if the dir simply doesn't exist.
             return "No source code definitions found."
        raise RuntimeError(f"Tool error: {data['error']}")

    if "message" in data and (
        data["message"] == "No supported source files found in the directory." or
        data["message"] == "No definitions found in source files."
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
