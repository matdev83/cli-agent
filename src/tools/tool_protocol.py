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
    def parameters(self) -> List[Dict[str, str]]:
        """
        A list of parameter definitions, where each definition is a dictionary
        containing 'name' and 'description' for the parameter.
        Example: [{'name': 'path', 'description': 'The path to the file.'}]
        """
        ...

    def execute(self, params: Dict[str, Any], agent_memory: Any = None) -> str:
        """
        Executes the tool with the given parameters.

        Args:
            params: A dictionary of parameters for the tool.
            agent_memory: An optional memory object for tools that need to interact with agent's memory.

        Returns:
            A string representing the result of the tool execution.
        """
        ...
