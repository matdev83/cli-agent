from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

from openai import OpenAI


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


class OpenRouterLLM:
    """LLM client using OpenRouter.ai API via the OpenAI SDK."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://cline.bot",
                "X-Title": "CLI Agent",
            },
        )

    def send_message(self, history: List[Dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": m["role"], "content": m["content"]} for m in history],
        )
        return response.choices[0].message.content
