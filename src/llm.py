from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict


class MockLLM:
    """Simple mock LLM that returns predefined responses."""

    def __init__(self, responses: List[str]):
        self._responses = list(responses)
        self._index = 0

    @classmethod
    def from_file(cls, path: str) -> "MockLLM":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("responses file must contain a JSON list")
        return cls([str(item) for item in data])

    def send_message(self, history: List[Dict[str, str]]) -> str:
        if self._index >= len(self._responses):
            return ""
        resp = self._responses[self._index]
        self._index += 1
        return resp
