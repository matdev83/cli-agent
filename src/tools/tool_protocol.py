from typing import Protocol, Dict, Any, List

class Tool(Protocol):
    """
    Protocol for a tool that the DeveloperAgent can execute.
    """

    @property
    def name(self) -> str:
        """The name of the tool, as it would be invoked by the LLM."""
        ...

    @property
    def description(self) -> str:
        """A brief description of what the tool does and its parameters."""
        ...

    @property
    def parameters_schema(self) -> Dict[str, str]:
        """
        A dictionary describing the tool's parameters, where keys are parameter
        names and values are their descriptions.
        Example: {"path": "The path to the file."}
        """
        ...

    def execute(self, params: Dict[str, Any], agent_tools_instance: Any) -> str:
        """
        Executes the tool with the given parameters.

        Args:
            params: A dictionary of parameters for the tool.
            agent_tools_instance: An instance of the agent or a dedicated tools
                                  handler class, providing access to shared
                                  functionality like user confirmations if needed.

        Returns:
            A string representing the result of the tool execution.
        """
        ...
