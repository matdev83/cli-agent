from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional

TOOL_USE_NAMES = [
    "execute_command",
    "read_file",
    "write_to_file",
    "replace_in_file",
    "search_files",
    "list_files",
    "list_code_definitions",
    "browser_action",
    "use_mcp_tool",
    "access_mcp_resource",
    "ask_followup_question",
    "plan_mode_respond",
    "load_mcp_documentation",
    "attempt_completion",
    "new_task",
    "condense",        # Added
    "report_bug",      # Added
    "new_rule",        # Added
    # "web_fetch", # Was commented out, keeping it so
]

@dataclass
class TextContent:
    type: str = "text"
    content: str
    partial: bool = False

@dataclass
class ToolUse:
    type: str = "tool_use"
    name: str
    params: Dict[str, str] = field(default_factory=dict)
    partial: bool = False

AssistantMessageContent = Union[TextContent, ToolUse]

def parse_assistant_message(message: str) -> List[AssistantMessageContent]:
    """
    Parse assistant message which might contain text and tool use requests.
    Uses xml.etree.ElementTree for parsing tool calls.
    """
    result: List[AssistantMessageContent] = []

    wrapped_message = f"<root_message_wrapper>{message}</root_message_wrapper>"

    try:
        root = ET.fromstring(wrapped_message)
    except ET.ParseError:
        if message.strip():
            result.append(TextContent(content=message.strip(), partial=False))
        return result

    if root.text and root.text.strip():
        result.append(TextContent(content=root.text.strip()))

    for element in root:
        if element.tag in TOOL_USE_NAMES:
            parameters: Dict[str, str] = {}
            for child in element:
                param_name = child.tag
                param_value = (child.text or "").strip()
                parameters[param_name] = param_value
            result.append(ToolUse(name=element.tag, params=parameters))
        else:
            # Serialize unknown tags back to string to preserve them as text
            result.append(TextContent(content=ET.tostring(element, encoding='unicode', method='xml')))

        if element.tail and element.tail.strip():
            result.append(TextContent(content=element.tail.strip()))

    if not result: return []

    merged_result: List[AssistantMessageContent] = [result[0]]
    for current_block in result[1:]:
        last_block = merged_result[-1]
        if isinstance(last_block, TextContent) and isinstance(current_block, TextContent):
            last_block.content += "\n" + current_block.content
        else:
            merged_result.append(current_block)

    return merged_result
