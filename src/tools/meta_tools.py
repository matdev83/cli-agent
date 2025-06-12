from __future__ import annotations

from typing import Dict, Any, List
from .tool_protocol import Tool

class NewTaskTool(Tool):
    """A tool to prepare context for creating a new task based on the current session."""
    @property
    def name(self) -> str:
        return "new_task"

    @property
    def description(self) -> str:
        return ("Request to create a new task with preloaded context covering the conversation with the user "
                "up to this point and key information for continuing with the new task. The context should include: "
                "1. Current Work, 2. Key Technical Concepts, 3. Relevant Files and Code, 4. Problem Solving, "
                "5. Pending Tasks and Next Steps.")

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
        """Executes the tool. Expects 'context' in params. Full task creation logic is pending."""
        context = params.get("context", "No context provided.")
        return f"Success: NewTaskTool called with context: '{context}'. Full implementation of new task creation is pending."

class AskFollowupQuestionTool(Tool):
    """A tool to ask the user a followup question to gather more information."""
    @property
    def name(self) -> str:
        return "ask_followup_question"

    @property
    def description(self) -> str:
        return ("Ask the user a question to gather additional information needed to complete the task. "
                "This is useful when the user's request is ambiguous or lacks critical details. "
                "The question should be clear and specific. If providing options, they should be concise and distinct.")

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {"name": "question", "description": "The question to ask the user.", "type": "string", "required": True},
            {"name": "options", "description": "Optional JSON string array of 2-5 options for the user.", "type": "string", "required": False}
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        """Executes the tool. Expects 'question' and optionally 'options' in params. Full implementation pending."""
        question = params.get("question", "No question provided.")
        options = params.get("options", "No options provided.")
        return f"Success: AskFollowupQuestionTool called. Question: '{question}', Options: '{options}'. Full implementation pending."

class AttemptCompletionTool(Tool):
    """A tool to present the result of work to the user after a tool use cycle."""
    @property
    def name(self) -> str:
        return "attempt_completion"

    @property
    def description(self) -> str:
        return ("After each tool use, the user will respond with the result of your work. "
                "Based on the result, you may decide to use another tool or present the result of your work to the user. "
                "Use this tool to present the result of your work. This tool use means you are done with the task for now.")

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {"name": "result", "description": "The result of your work to present to the user.", "type": "string", "required": True},
            {"name": "command", "description": "The command that was executed to achieve this result, if applicable.", "type": "string", "required": False}
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        """Executes the tool. Expects 'result' and optionally 'command' in params. Full implementation pending."""
        result = params.get("result", "No result provided.")
        command = params.get("command", "No command provided.")
        return f"Success: AttemptCompletionTool called. Result: '{result}', Command: '{command}'. Full implementation pending."

class PlanModeRespondTool(Tool):
    """A tool for responding to the user in planning mode."""
    @property
    def name(self) -> str:
        return "plan_mode_respond"

    @property
    def description(self) -> str:
        return ("Respond to the user's inquiry in an effort to plan a solution to their problem. "
                "Use this tool if you are in 'planning mode' and need to communicate your plan or ask clarifying "
                "questions before attempting to execute the plan. This is not for asking for help or saying you are stuck.")

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [
            {"name": "response", "description": "The response to send to the user for planning purposes.", "type": "string", "required": True}
        ]

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        """Executes the tool. Expects 'response' in params. Full implementation pending."""
        response = params.get("response", "No response provided.")
        return f"Success: PlanModeRespondTool called. Response: '{response}'. Full implementation pending."

class LoadMcpDocumentationTool(Tool):
    """A tool to load documentation about creating MCP servers."""
    @property
    def name(self) -> str:
        return "load_mcp_documentation"

    @property
    def description(self) -> str:
        return ("Load documentation about creating MCP servers, including an architectural overview, "
                "instructions for setting up the development environment, and tutorials for creating basic MCP servers. "
                "This tool helps in understanding and starting MCP server development.")

    @property
    def parameters(self) -> List[Dict[str, str]]:
        return [] # No parameters as per prompt

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        """Executes the tool. Expects no parameters. Full implementation pending."""
        # No parameters expected, so params dictionary might be empty or not contain specific keys.
        return f"Success: LoadMcpDocumentationTool called. Full implementation pending."

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
