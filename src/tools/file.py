from __future__ import annotations

import os
import re
import fnmatch
import json # For ListFilesTool and SearchFilesTool output
from pathlib import Path
from typing import List, Dict, Any, Optional
from .tool_protocol import Tool

# --- Helper function for ReplaceInFileTool ---
def _parse_diff_blocks(diff: str) -> List[tuple[str, str]]:
    """
    Parses a diff string into search and replace blocks.
    The expected format is <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE.
    """
    pattern = re.compile(
        r"<<<<<<< SEARCH\n(?P<search>.*?)\n=======\n(?P<replace>.*?)\n>>>>>>> REPLACE",
        re.DOTALL,
    )
    blocks = []
    for m in pattern.finditer(diff):
        blocks.append((m.group("search"), m.group("replace")))

    # If the diff string is not empty but no blocks were found, it's a format error.
    if not blocks and diff.strip():
        raise ValueError("No valid diff blocks found. Ensure the format is: <<<<<<< SEARCH...=======...>>>>>>> REPLACE")
    return blocks

# --- Utility for path resolution ---
def _resolve_path(path_str: str, agent_memory: Any = None) -> Path:
    """
    Resolves a path string to an absolute Path object.
    If agent_memory.cwd is available, it's used as the base for relative paths.
    Otherwise, os.getcwd() is used.
    Absolute paths are resolved directly.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()

    base_dir = Path(os.getcwd()) # Default base directory
    if agent_memory and hasattr(agent_memory, 'cwd') and agent_memory.cwd:
        base_dir = Path(agent_memory.cwd)

    return (base_dir / p).resolve()

# --- Tool Implementations ---

class ReadFileTool(Tool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Reads the entire content of a specified file and returns it as a string."

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {
                "name": "path",
                "description": "The relative or absolute path to the file to be read.",
                "type": "string",
                "required": True
            }
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        file_path_str = params.get("path")
        if not file_path_str:
            return "Error: Missing required parameter 'path'."

        try:
            abs_file_path = _resolve_path(file_path_str, agent_memory)

            if not abs_file_path.exists():
                return f"Error: File not found at {str(abs_file_path)}"
            if not abs_file_path.is_file():
                return f"Error: Path {str(abs_file_path)} is a directory, not a file."

            with abs_file_path.open("r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading file {file_path_str}: {e}"


class WriteToFileTool(Tool):
    @property
    def name(self) -> str:
        return "write_to_file"

    @property
    def description(self) -> str:
        return "Writes the given content to a specified file. Creates parent directories if they don't exist."

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {
                "name": "path",
                "description": "The relative or absolute path to the file to be written.",
                "type": "string",
                "required": True
            },
            {
                "name": "content",
                "description": "The content to write to the file.",
                "type": "string",
                "required": True
            }
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        file_path_str = params.get("path")
        content = params.get("content")

        if not file_path_str:
            return "Error: Missing required parameter 'path'."
        if content is None:
            return "Error: Missing required parameter 'content'."

        try:
            abs_file_path = _resolve_path(file_path_str, agent_memory)

            abs_file_path.parent.mkdir(parents=True, exist_ok=True)
            with abs_file_path.open("w", encoding="utf-8") as f:
                f.write(content)
            return f"File written successfully to {str(abs_file_path)}"
        except Exception as e:
            return f"Error writing to file {file_path_str}: {e}"


class ReplaceInFileTool(Tool):
    @property
    def name(self) -> str:
        return "replace_in_file"

    @property
    def description(self) -> str:
        return ("Applies one or more SEARCH/REPLACE diff blocks to a file. "
                "Each block must follow the format: <<<<<<< SEARCH\\n(lines to search)\\n=======\\n(lines to replace)\\n>>>>>>> REPLACE")

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {
                "name": "path",
                "description": "The relative or absolute path to the file to be modified.",
                "type": "string",
                "required": True
            },
            {
                "name": "diff_blocks",
                "description": "A string containing one or more diff blocks in the specified format.",
                "type": "string",
                "required": True
            }
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        file_path_str = params.get("path")
        diff_str = params.get("diff_blocks")

        if not file_path_str:
            return "Error: Missing required parameter 'path'."
        if diff_str is None: # diff_blocks can be an empty string (no-op)
            return "Error: Missing required parameter 'diff_blocks'."

        try:
            abs_file_path = _resolve_path(file_path_str, agent_memory)

            if not abs_file_path.exists():
                return f"Error: File not found at {str(abs_file_path)}"
            if not abs_file_path.is_file():
                return f"Error: Path {str(abs_file_path)} is a directory, not a file."

            original_text = abs_file_path.read_text(encoding="utf-8")

            if not diff_str.strip(): # If diff_blocks is empty or whitespace, it's a no-op
                return f"Warning: 'diff_blocks' was empty. No changes made to file {str(abs_file_path)}."

            blocks = _parse_diff_blocks(diff_str) # Helper function handles format errors

            modified_text = original_text
            for i, (search, replace) in enumerate(blocks):
                if search not in modified_text:
                    # Be more specific about which block failed.
                    max_preview = 30
                    search_preview = search[:max_preview].replace('\n', '\\n') + ('...' if len(search) > max_preview else '')
                    return (f"Error: Search block {i+1} (starting with '{search_preview}') "
                            f"not found in the current state of the file {str(abs_file_path)}. "
                            "Ensure blocks are ordered correctly and match the file content.")
                modified_text = modified_text.replace(search, replace, 1)

            abs_file_path.write_text(modified_text, encoding="utf-8")
            return f"File {str(abs_file_path)} modified successfully with {len(blocks)} block(s)."
        except ValueError as ve: # Catch errors from _parse_diff_blocks
            return f"Error processing diff_blocks for {file_path_str}: {ve}"
        except Exception as e:
            return f"Error replacing in file {file_path_str}: {e}"


class ListFilesTool(Tool):
    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        return "Lists files and directories within a specified path. Can list recursively. Returns a JSON list of strings."

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {
                "name": "path", # Renamed from dir_path for consistency with other tools
                "description": "The relative or absolute path to the directory to list.",
                "type": "string",
                "required": True
            },
            {
                "name": "recursive",
                "description": "Whether to list files recursively. Defaults to False.",
                "type": "boolean",
                "required": False
            }
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        dir_path_str = params.get("path")
        recursive = params.get("recursive", False)

        if not dir_path_str:
            return "Error: Missing required parameter 'path'."

        try:
            abs_dir_path = _resolve_path(dir_path_str, agent_memory)

            if not abs_dir_path.exists():
                return f"Error: Directory not found at {str(abs_dir_path)}"
            if not abs_dir_path.is_dir():
                return f"Error: Path {str(abs_dir_path)} is not a directory."

            results: List[str] = []
            if not recursive:
                for entry in abs_dir_path.iterdir():
                    results.append(entry.name + ("/" if entry.is_dir() else ""))
            else:
                for root, dirs, files in os.walk(abs_dir_path):
                    current_root = Path(root)
                    # Add files relative to abs_dir_path
                    for name in files:
                        rel_path = (current_root / name).relative_to(abs_dir_path)
                        results.append(rel_path.as_posix())
                    # Add directories relative to abs_dir_path, ensuring they end with /
                    for name in dirs:
                        rel_path = (current_root / name).relative_to(abs_dir_path)
                        results.append(rel_path.as_posix() + "/")

            return json.dumps(sorted(results))
        except Exception as e:
            return f"Error listing files in {dir_path_str}: {e}"


class SearchFilesTool(Tool):
    @property
    def name(self) -> str:
        return "search_files"

    @property
    def description(self) -> str:
        return ("Searches for a regex pattern in files within a directory. "
                "Can filter by file pattern (glob). Returns a JSON list of match objects.")

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {
                "name": "directory",
                "description": "The relative or absolute path to the directory to search within.",
                "type": "string",
                "required": True
            },
            {
                "name": "regex_pattern",
                "description": "The Python regex pattern to search for within file lines.",
                "type": "string",
                "required": True
            },
            {
                "name": "file_pattern",
                "description": "Optional glob pattern to filter files (e.g., '*.py', 'test_*.py'). Defaults to all files ('*').",
                "type": "string",
                "required": False
            }
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        directory_str = params.get("directory")
        regex_str = params.get("regex_pattern")
        file_glob_pattern = params.get("file_pattern", "*")

        if not directory_str:
            return "Error: Missing required parameter 'directory'."
        if not regex_str:
            return "Error: Missing required parameter 'regex_pattern'."

        try:
            abs_dir_path = _resolve_path(directory_str, agent_memory)

            if not abs_dir_path.exists():
                return f"Error: Directory not found at {str(abs_dir_path)}"
            if not abs_dir_path.is_dir():
                return f"Error: Path {str(abs_dir_path)} is not a directory."

            try:
                pattern = re.compile(regex_str)
            except re.error as e:
                return f"Error: Invalid regex pattern '{regex_str}': {e}"

            matches: List[Dict[str, Any]] = []
            for dirpath_str, _, filenames in os.walk(abs_dir_path):
                current_walk_dir = Path(dirpath_str)
                for filename in filenames:
                    if not fnmatch.fnmatch(filename, file_glob_pattern):
                        continue

                    file_path = current_walk_dir / filename
                    try:
                        # Ensure we only try to read files
                        if not file_path.is_file():
                            continue
                        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                            for lineno, line in enumerate(f, start=1):
                                if pattern.search(line):
                                    matches.append(
                                        {
                                            "file": (file_path.relative_to(abs_dir_path)).as_posix(),
                                            "line": lineno,
                                            "content": line.rstrip("\n"),
                                        }
                                    )
                    except Exception: # Skip files that cannot be opened or read
                        pass

            return json.dumps(matches)
        except Exception as e:
            return f"Error searching files in {directory_str}: {e}"

# --- Wrapper functions for old tests ---
# Ensure json is imported if not already (it is used by ListFilesTool and SearchFilesTool's execute, but wrappers parse it)
# import json # Already imported at the top

def read_file(path: str, agent_memory: Any = None) -> str:
    tool = ReadFileTool()
    result = tool.execute({"path": path}, agent_memory=agent_memory)
    if result.startswith("Error:"):
        # Old tests might expect FileNotFoundError or similar.
        # This is a simple bridge; more sophisticated error mapping could be done.
        # Example: if "File not found" in result: raise FileNotFoundError(result)
        raise ValueError(result)
    return result

def write_to_file(path: str, content: str, agent_memory: Any = None) -> None:
    tool = WriteToFileTool()
    result = tool.execute({"path": path, "content": content}, agent_memory=agent_memory)
    if result.startswith("Error:"):
        raise ValueError(result)
    # Original function had no return, so None is fine.

def replace_in_file(path: str, diff_blocks: str, agent_memory: Any = None) -> None:
    tool = ReplaceInFileTool()
    # The test 'test_replace_in_file' uses a diff format "------- SEARCH...+++++++ REPLACE"
    # The tool's _parse_diff_blocks method expects "<<<<<<< SEARCH...>>>>>>> REPLACE"
    # This wrapper will adapt the test's format to the tool's expected format.
    # Note: The test's actual diff string is "------- SEARCH\nfoo\n=======\nbar\n+++++++ REPLACE"
    # The tool's internal parser is strict. The adaptation below handles this.
    adapted_diff = diff_blocks.replace("------- SEARCH", "<<<<<<< SEARCH") \
                              .replace("+++++++ REPLACE", ">>>>>>> REPLACE")

    result = tool.execute({"path": path, "diff_blocks": adapted_diff}, agent_memory=agent_memory)

    # The test 'test_replace_in_file_not_found' expects ValueError if search block not found.
    # The tool's execute method returns a string like "Error: Search block ... not found..."
    if result.startswith("Error:") and "Search block" in result and "not found" in result:
        raise ValueError(result) # Make it a ValueError for the test
    elif result.startswith("Error:"): # Other errors reported by the tool
        raise RuntimeError(result) # Or a more specific custom error

def list_files(path: str, recursive: bool = False, agent_memory: Any = None) -> List[str]:
    tool = ListFilesTool()
    result_str = tool.execute({"path": path, "recursive": recursive}, agent_memory=agent_memory)
    if result_str.startswith("Error:"):
        raise ValueError(result_str)
    # The test 'test_list_files' expects a list of strings.
    # The tool's execute method returns a JSON string representation of a list.
    loaded_json = json.loads(result_str)
    return loaded_json

def search_files(directory: str, regex_pattern: str, file_pattern: str = "*", agent_memory: Any = None) -> List[Dict[str, Any]]:
    tool = SearchFilesTool()
    params = {"directory": directory, "regex_pattern": regex_pattern}
    # The tool's execute method defaults file_pattern to '*' internally if not provided.
    # So, only add file_pattern to params if it's not the default "*" to avoid redundancy.
    if file_pattern != "*": # This check ensures we don't override the tool's internal default unless specified.
        params["file_pattern"] = file_pattern

    result_str = tool.execute(params, agent_memory=agent_memory)
    if result_str.startswith("Error:"):
        raise ValueError(result_str)
    # The test 'test_search_files' expects a list of dictionaries.
    # The tool's execute method returns a JSON string.
    return json.loads(result_str)
