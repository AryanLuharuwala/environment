from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..providers import LLMResponse, Provider, Usage, make_provider


@dataclass
class Agent:
    """A persona with a stable system prompt, bound to a Provider.

    Agents are provider-agnostic: the Provider handles caching (Claude),
    tool-schema conversion (OpenAI-format), and per-turn message shape. The
    Agent only supplies name + system prompt and a preferred `max_tokens`.
    """

    name: str
    system_prompt: str
    provider: Provider = field(default_factory=make_provider)
    max_tokens: int = 2048
    usage: Usage = field(default_factory=Usage)

    @property
    def model(self) -> str:
        return self.provider.model

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse:
        response = self.provider.complete(
            system=self.system_prompt,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=self.max_tokens,
        )
        self.usage.add(response.usage)
        return response
