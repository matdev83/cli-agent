from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Memory:
    """Simple in-memory storage for conversation history and file context."""

    history: List[Dict[str, str]] = field(default_factory=list)
    file_context: Dict[str, str] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        self.history.append({"role": role, "content": content})

    def add_file_context(self, path: str, content: str) -> None:
        """Store the latest known content for *path* in memory."""
        self.file_context[path] = content

    def get_history(self) -> List[Dict[str, str]]:
        """Return a copy of the message history."""
        return list(self.history)

    def get_file_context(self) -> Dict[str, str]:
        """Return a copy of stored file contents."""
        return dict(self.file_context)
