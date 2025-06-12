from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import xml.etree.ElementTree as ET

TOOL_USE_NAMES = [
    "execute_command",
    "read_file",
    "write_to_file",
    "replace_in_file",
    "search_files",
    "list_files",
    "list_code_definition_names",
    "browser_action",
    "use_mcp_tool",
    "access_mcp_resource",
    "ask_followup_question",
    "plan_mode_respond",
    "load_mcp_documentation",
    "attempt_completion",
    "new_task",
    "condense",
    "report_bug",
    "new_rule",
    "web_fetch",
]

TOOL_PARAM_NAMES = [
    "command",
    "requires_approval",
    "path",
    "content",
    "diff",
    "regex",
    "file_pattern",
    "recursive",
    "action",
    "url",
    "coordinate",
    "text",
    "server_name",
    "tool_name",
    "arguments",
    "uri",
    "question",
    "options",
    "response",
    "result",
    "context",
    "title",
    "what_happened",
    "steps_to_reproduce",
    "api_request_output",
    "additional_context",
]

@dataclass
class TextContent:
    type: str
    content: str
    partial: bool = False

@dataclass
class ToolUse:
    type: str
    name: str
    params: Dict[str, str]
    partial: bool = False

AssistantMessageContent = Union[TextContent, ToolUse]

def _extract_param(block: str, param: str) -> Optional[str]:
    """Extract a parameter value from an XML block using ElementTree."""
    try:
        root = ET.fromstring(f"<root>{block}</root>")
    except ET.ParseError:
        start_tag = f"<{param}>"
        start = block.find(start_tag)
        if start == -1:
            return None
        start += len(start_tag)
        end_tag = f"</{param}>"
        end = block.find(end_tag, start)
        if end != -1:
            return block[start:end].strip()
        next_tag = block.find("<", start)
        if next_tag == -1:
            return block[start:].strip()
        return block[start:next_tag].strip()
    elem = root.find(param)
    if elem is None or elem.text is None:
        return None
    return elem.text.strip()

def parse_assistant_message(message: str) -> List[AssistantMessageContent]:
    """Parse assistant message containing optional tool calls."""
    result: List[AssistantMessageContent] = []
    i = 0
    length = len(message)
    while i < length:
        lt = message.find("<", i)
        if lt == -1:
            text = message[i:].strip()
            if text:
                result.append(TextContent(type="text", content=text))
            break
        if lt > i:
            text = message[i:lt]
            if text.strip():
                result.append(TextContent(type="text", content=text.strip()))
        gt = message.find(">", lt)
        if gt == -1:
            text = message[lt:]
            if text.strip():
                result.append(TextContent(type="text", content=text.strip(), partial=True))
            break
        tag = message[lt + 1 : gt]
        if tag.startswith("/"):
            i = gt + 1
            continue
        closing = f"</{tag}>"
        close_pos = message.find(closing, gt + 1)
        if tag in TOOL_USE_NAMES:
            if close_pos == -1:
                inner = message[gt + 1 :]
                params = {
                    p: _extract_param(inner, p) for p in TOOL_PARAM_NAMES if f"<{p}>" in inner
                }
                result.append(ToolUse(type="tool_use", name=tag, params=params, partial=True))
                break
            inner = message[gt + 1 : close_pos]
            params = {
                p: _extract_param(inner, p) for p in TOOL_PARAM_NAMES if f"<{p}>" in inner
            }
            params = {k: v for k, v in params.items() if v is not None}
            result.append(ToolUse(type="tool_use", name=tag, params=params, partial=False))
            i = close_pos + len(closing)
        else:
            if close_pos == -1:
                content = message[gt + 1 :]
                if content.strip():
                    result.append(TextContent(type="text", content=content.strip(), partial=True))
                break
            content = message[gt + 1 : close_pos]
            if content.strip():
                result.append(TextContent(type="text", content=content.strip()))
            i = close_pos + len(closing)
    return result
