from __future__ import annotations

from dataclasses import dataclass, field
import re
from textwrap import shorten
from typing import Dict, List, Iterable


@dataclass
class Memory:
    """Simple in-memory storage with basic summarisation and search."""

    history: List[Dict[str, str]] = field(default_factory=list)
    file_context: Dict[str, str] = field(default_factory=dict)
    max_messages: int = 50
    summary_char_limit: int = 200

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_messages:
            self._summarize_and_trim()

    def add_file_context(self, path: str, content: str) -> None:
        """Store the latest known content for *path* in memory."""
        self.file_context[path] = content

    def get_history(self) -> List[Dict[str, str]]:
        """Return a copy of the message history."""
        return list(self.history)

    def get_file_context(self) -> Dict[str, str]:
        """Return a copy of stored file contents."""
        return dict(self.file_context)

    # --- New functionality for summarisation and search ---

    def _summarize_and_trim(self) -> None:
        """Summarize older messages when history grows too large."""
        keep_last = self.max_messages // 2
        if len(self.history) <= keep_last + 1:
            return

        system_msg = self.history[0]
        to_summarise = self.history[1:-keep_last]
        summary_lines = []
        for msg in to_summarise:
            clean = re.sub(r"\s+", " ", msg["content"]).strip()
            snippet = shorten(clean, width=self.summary_char_limit, placeholder="...")
            summary_lines.append(f"{msg['role']}: {snippet}")
        summary_text = "Previous conversation summary:\n" + "\n".join(summary_lines)
        summary_message = {"role": "system", "content": summary_text}
        self.history = [system_msg, summary_message] + self.history[-keep_last:]

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Return messages most relevant to *query* using simple word overlap."""
        query_words = set(query.lower().split())
        scored: List[tuple[int, Dict[str, str]]] = []
        for msg in self.history:
            words = set(msg["content"].lower().split())
            score = len(words & query_words)
            if score:
                scored.append((score, msg))

        scored.sort(key=lambda s: s[0], reverse=True)
        return [m for _, m in scored[:top_k]]

