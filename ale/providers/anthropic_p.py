from __future__ import annotations

import os
from typing import Any

import anthropic

from .base import LLMResponse, Provider, ToolCall, Usage

DEFAULT_MODEL = os.environ.get("ALE_ANTHROPIC_MODEL", "claude-opus-4-7")


class AnthropicProvider(Provider):
    """Claude backend. Sends the system prompt with ephemeral cache_control so
    repeated rollouts of the same agent read from cache."""

    name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.client = client or anthropic.Anthropic()

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        max_tokens: int = 2048,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = self._tool_choice(tool_choice)
        response = self.client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            t = getattr(block, "type", "")
            if t == "text":
                text_parts.append(block.text)
            elif t == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))

        u = response.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=response.stop_reason or "",
        )

    def echo_assistant(self, response: LLMResponse) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if response.text:
            content.append({"type": "text", "text": response.text})
        for tc in response.tool_calls:
            content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input})
        return {"role": "assistant", "content": content}

    def tool_results(self, results: list[tuple[str, str]]) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tid, "content": content}
                    for tid, content in results
                ],
            }
        ]

    @staticmethod
    def _tool_choice(choice: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(choice, dict):
            return choice
        if choice in ("auto", "any", "none"):
            return {"type": choice}
        return {"type": "tool", "name": choice}
