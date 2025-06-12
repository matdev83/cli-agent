from __future__ import annotations

from pathlib import Path
import ast
from typing import List


def _python_top_level_defs(file_path: Path) -> List[str]:
    source = file_path.read_text(encoding="utf-8")
    try:
        module = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    defs = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            line = lines[node.lineno - 1].rstrip()
            defs.append("|" + line)
    return defs


def list_code_definition_names(path: str) -> str:
    """List top level definition names for supported source files in directory."""
    p = Path(path)
    if not p.is_dir():
        raise ValueError("path must be a directory")

    result_lines: List[str] = []
    for entry in sorted(p.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix == ".py":
            defs = _python_top_level_defs(entry)
        else:
            defs = []
        if defs:
            result_lines.append(entry.name)
            result_lines.append("|----")
            result_lines.extend(defs)
            result_lines.append("|----")
            result_lines.append("")
    if not result_lines:
        return "No source code definitions found."
    return "\n".join(result_lines).rstrip()


def browser_action(*args, **kwargs):
    """Stub for browser_action tool."""
    raise NotImplementedError("browser_action is not implemented yet")


def use_mcp_tool(*args, **kwargs):
    """Stub for use_mcp_tool tool."""
    raise NotImplementedError("use_mcp_tool is not implemented yet")


def access_mcp_resource(*args, **kwargs):
    """Stub for access_mcp_resource tool."""
    raise NotImplementedError("access_mcp_resource is not implemented yet")
