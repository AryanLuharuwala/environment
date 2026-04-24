from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import anthropic


DEFAULT_MODEL = os.environ.get("ALE_MODEL", "claude-opus-4-7")


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def add(self, u: Any) -> None:
        self.input_tokens += getattr(u, "input_tokens", 0) or 0
        self.output_tokens += getattr(u, "output_tokens", 0) or 0
        self.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0


@dataclass
class Agent:
    """A persona with a stable system prompt that calls Claude.

    The system prompt is sent with `cache_control` so that repeated invocations
    (across rollouts of the same task) read from cache. Tools and the per-call
    user message are not cached.
    """

    name: str
    system_prompt: str
    model: str = DEFAULT_MODEL
    max_tokens: int = 4096
    client: anthropic.Anthropic = field(default_factory=anthropic.Anthropic)
    usage: Usage = field(default_factory=Usage)

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        response = self.client.messages.create(**kwargs)
        self.usage.add(response.usage)
        return response

    @staticmethod
    def text_of(response: Any) -> str:
        return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")

    @staticmethod
    def tool_uses(response: Any) -> list[Any]:
        return [b for b in response.content if getattr(b, "type", "") == "tool_use"]
