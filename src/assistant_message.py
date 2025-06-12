from __future__ import annotations

import xml.etree.ElementTree as ET
import copy # Add copy import
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
    content: str
    type: str = "text"
    partial: bool = False

@dataclass
class ToolUse:
    name: str
    type: str = "tool_use"
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

    for element in root: # element is a direct child of <root_message_wrapper>
        if element.tag in TOOL_USE_NAMES: # Handles direct tool calls like <read_file>...</read_file>
            actual_tool_name = element.tag
            parameters: Dict[str, str] = {}
            for param_child in element: # Parameters are direct children
                param_name = param_child.tag
                param_value = (param_child.text or "").strip()
                parameters[param_name] = param_value
            result.append(ToolUse(name=actual_tool_name, params=parameters))

        elif element.tag == "tool_use": # Handles wrapped tool calls like <tool_use><tool_name>...</tool_name>...</tool_use>
            tool_name_element = element.find("tool_name")
            if tool_name_element is None or not (tool_name_element.text and tool_name_element.text.strip()):
                # Malformed wrapped tool_use (missing tool_name), treat as text
                result.append(TextContent(content=ET.tostring(element, encoding='unicode', method='xml')))
            else:
                actual_tool_name = tool_name_element.text.strip()
                parameters: Dict[str, str] = {}
                params_element = element.find("params")
                if params_element is not None:
                    for param_child in params_element: # Parameters are children of <params>
                        param_name = param_child.tag
                        param_value = (param_child.text or "").strip()
                        parameters[param_name] = param_value
                result.append(ToolUse(name=actual_tool_name, params=parameters))

        elif element.tag == "text_content":
            text_val = (element.text or "").strip()
            if text_val:
                result.append(TextContent(content=text_val))
        else: # Fallback for other unknown top-level tags
            element_copy = copy.deepcopy(element) # Create a deep copy to avoid modifying the original element
            element_copy.tail = None # Clear the tail on the copy
            # Serialize the copy. method='xml' preserves more XML structure like self-closing tags if appropriate.
            element_str = ET.tostring(element_copy, encoding='unicode', method='xml').strip()
            if element_str: # Add if not empty after stripping
                result.append(TextContent(content=element_str))

        if element.tail and element.tail.strip(): # Text after the current element
            result.append(TextContent(content=element.tail.strip()))

    if not result: return []

    # Return the list of parsed blocks directly.
    # The previous merging logic was too aggressive for some test cases
    # and could combine text that should remain separate due to intervening XML.
    return result
