from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read += other.cache_read
        self.cache_write += other.cache_write

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
        }


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""


class Provider(ABC):
    """Backend-agnostic LLM interface.

    Providers own model selection, request formatting, and the per-turn message
    shape. The harness stays provider-neutral by routing through
    `complete` / `echo_assistant` / `tool_results`.
    """

    name: str = "provider"
    model: str = ""

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        max_tokens: int = 2048,
    ) -> LLMResponse: ...

    @abstractmethod
    def echo_assistant(self, response: LLMResponse) -> dict[str, Any]:
        """Assistant turn to append before tool results."""

    @abstractmethod
    def tool_results(self, results: list[tuple[str, str]]) -> list[dict[str, Any]]:
        """Message(s) to append carrying tool outputs. Anthropic returns one
        user message with tool_result blocks; OpenAI returns one role='tool'
        message per result."""
