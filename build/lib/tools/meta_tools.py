from __future__ import annotations

from typing import Dict, Any
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
    def parameters_schema(self) -> Dict[str, str]:
        return {
            "context": "The context to preload the new task with."
        }

    def execute(self, params: Dict[str, Any], agent_tools_instance: Any) -> str:
        """Executes the tool. Expects 'context' in params. Full task creation logic is pending."""
        context = params.get("context")
        if context is None: # Check if None, empty string might be valid context
            return "Error: Missing required parameter 'context' for tool 'new_task'."
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
    def parameters_schema(self) -> Dict[str, str]:
        return {
            "question": "The question to ask the user.",
            "options": "Optional array of 2-5 options for the user (JSON string or list)."
        }

    def execute(self, params: Dict[str, Any], agent_tools_instance: Any) -> str:
        """Executes the tool. Expects 'question' and optionally 'options' in params. Full implementation pending."""
        question = params.get("question")
        if question is None: # Check if None, empty string might be a (bad) question
            return "Error: Missing required parameter 'question' for tool 'ask_followup_question'."
        options = params.get("options", "No options provided.") # Optional, so default is fine
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
    def parameters_schema(self) -> Dict[str, str]:
        return {
            "result": "The result of the task.",
            "command": "Optional CLI command to demonstrate the result."
        }

    def execute(self, params: Dict[str, Any], agent_tools_instance: Any) -> str:
        """Executes the tool. Expects 'result' and optionally 'command' in params. Full implementation pending."""
        result = params.get("result")
        if result is None: # Check if None
            return "Error: Missing required parameter 'result' for tool 'attempt_completion'."
        command = params.get("command", "No command provided.") # Optional
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
    def parameters_schema(self) -> Dict[str, str]:
        return {
            "response": "The response to provide to the user."
        }

    def execute(self, params: Dict[str, Any], agent_tools_instance: Any) -> str:
        """Executes the tool. Expects 'response' in params. Full implementation pending."""
        response = params.get("response")
        if response is None: # Check if None
            return "Error: Missing required parameter 'response' for tool 'plan_mode_respond'."
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
    def parameters_schema(self) -> Dict[str, str]:
        return {} # No parameters as per prompt

    def execute(self, params: Dict[str, Any], agent_tools_instance: Any) -> str:
        """Executes the tool. Expects no parameters. Full implementation pending."""
        # No parameters expected, so params dictionary might be empty or not contain specific keys.
        return "Success: LoadMcpDocumentationTool called. Full implementation pending."

class CondenseTool(Tool):
    @property
    def name(self) -> str:
        return "condense"

    @property
    def description(self) -> str:
        return "Creates a detailed summary of the conversation so far to compact the context window. (Full implementation pending)"

    @property
    def parameters_schema(self) -> Dict[str, str]:
        return {
            "context": "Detailed summary of the conversation to be condensed."
        }

    def execute(self, params: Dict[str, Any], agent_tools_instance: Any) -> str:
        context = params.get("context")
        if context is None: # Check if None
            return "Error: Missing required parameter 'context' for tool 'condense'."
        return f"Success: CondenseTool called with context: '{context}'. Full implementation of context condensation is pending."

class ReportBugTool(Tool):
    @property
    def name(self) -> str:
        return "report_bug"

    @property
    def description(self) -> str:
        return "Collects information to submit a bug report. (Full implementation pending)"

    @property
    def parameters_schema(self) -> Dict[str, str]:
        return {
            "title": "Concise description of the issue.",
            "what_happened": "What happened and what was expected.",
            "steps_to_reproduce": "Steps to reproduce the bug.",
            "api_request_output": "Relevant API request output, if any.",
            "additional_context": "Other issue details or relevant context."
        }

    def execute(self, params: Dict[str, Any], agent_tools_instance: Any) -> str:
        title = params.get('title')
        what_happened = params.get('what_happened')
        steps_to_reproduce = params.get('steps_to_reproduce')

        missing_required = []
        if title is None:
            missing_required.append('title')
        if what_happened is None:
            missing_required.append('what_happened')
        if steps_to_reproduce is None:
            missing_required.append('steps_to_reproduce')

        if missing_required:
            return f"Error: Missing required parameters for 'report_bug': {', '.join(missing_required)}."

        # Optional params can use .get with a default
        api_request_output = params.get('api_request_output', 'N/A')
        additional_context = params.get('additional_context', 'N/A')

        return (f"Success: ReportBugTool called. Title: '{title}', What Happened: '{what_happened}', "
                f"Steps: '{steps_to_reproduce}', API Output: '{api_request_output}', Additional: '{additional_context}'. "
                f"Full implementation of bug reporting is pending.")

class NewRuleTool(Tool):
    @property
    def name(self) -> str:
        return "new_rule"

    @property
    def description(self) -> str:
        return "Creates a new Cline rule file in .clinerules directory. (Full implementation pending)"

    @property
    def parameters_schema(self) -> Dict[str, str]:
        return {
            "path": "Path for the new rule file (e.g., .clinerules/my-rule.md). Should be relative to project root.",
            "content": "Content of the new rule file (Markdown format)."
        }

    def execute(self, params: Dict[str, Any], agent_tools_instance: Any) -> str:
        path = params.get("path")
        content = params.get("content") # Renamed for clarity, was content_preview

        missing_required = []
        if path is None:
            missing_required.append('path')
        if content is None:
            missing_required.append('content')

        if missing_required:
            return f"Error: Missing required parameters for 'new_rule': {', '.join(missing_required)}."

        content_preview = content[:50] + "..." if content else "No content provided."
        # In a real tool, this would use file writing tools, considering agent_memory.cwd
        return (f"Success: NewRuleTool called. Path: '{path}', Content (preview): '{content_preview}'. "
                f"Full implementation of rule file creation is pending.")
