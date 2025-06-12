from __future__ import annotations

from typing import Dict, Any, List
from .tool_protocol import Tool

class NewTaskTool(Tool):
    @property
    def name(self) -> str:
        return "new_task"

    @property
    def description(self) -> str:
        return "Creates context for a new task based on the current conversation. (Full implementation pending)"

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {
                "name": "context",
                "description": "Detailed summary of the conversation and work so far to preload the new task.",
                "type": "string",
                "required": True
            }
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        context = params.get("context", "No context provided.")
        return f"Success: NewTaskTool called with context: '{context}'. Full implementation of new task creation is pending."

class CondenseTool(Tool):
    @property
    def name(self) -> str:
        return "condense"

    @property
    def description(self) -> str:
        return "Creates a detailed summary of the conversation so far to compact the context window. (Full implementation pending)"

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {
                "name": "context", # Parameter name kept as "context" as per prompt, though "summary" might also fit
                "description": "Detailed summary of the conversation to be condensed.",
                "type": "string",
                "required": True
            }
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        context = params.get("context", "No context provided.")
        return f"Success: CondenseTool called with context: '{context}'. Full implementation of context condensation is pending."

class ReportBugTool(Tool):
    @property
    def name(self) -> str:
        return "report_bug"

    @property
    def description(self) -> str:
        return "Collects information to submit a bug report. (Full implementation pending)"

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {
                "name": "title",
                "description": "Concise description of the issue.",
                "type": "string",
                "required": True
            },
            {
                "name": "what_happened",
                "description": "What happened and what was expected.",
                "type": "string",
                "required": True
            },
            {
                "name": "steps_to_reproduce",
                "description": "Steps to reproduce the bug.",
                "type": "string",
                "required": True
            },
            {
                "name": "api_request_output",
                "description": "Relevant API request output, if any.",
                "type": "string",
                "required": False # Explicitly False
            },
            {
                "name": "additional_context",
                "description": "Other issue details or relevant context.",
                "type": "string",
                "required": False # Explicitly False
            }
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        # For a stub, just acknowledging the parameters is useful.
        title = params.get('title', 'N/A')
        what_happened = params.get('what_happened', 'N/A')
        # Add more if desired for the stub message.
        return (f"Success: ReportBugTool called. Title: '{title}', What Happened: '{what_happened}'. "
                f"Received params: {params}. Full implementation of bug reporting is pending.")

class NewRuleTool(Tool):
    @property
    def name(self) -> str:
        return "new_rule"

    @property
    def description(self) -> str:
        return "Creates a new Cline rule file in .clinerules directory. (Full implementation pending)"

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {
                "name": "path",
                "description": "Path for the new rule file (e.g., .clinerules/my-rule.md). Should be relative to project root.",
                "type": "string",
                "required": True
            },
            {
                "name": "content",
                "description": "Content of the new rule file (Markdown format).",
                "type": "string",
                "required": True
            }
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        path = params.get("path", "No path provided.")
        content_preview = params.get("content", "No content provided.")[:50] + "..."
        # In a real tool, this would use file writing tools, considering agent_memory.cwd
        return (f"Success: NewRuleTool called. Path: '{path}', Content (preview): '{content_preview}'. "
                f"Full implementation of rule file creation is pending.")
