from __future__ import annotations

import json
import os
from typing import Any

from .base import LLMResponse, Provider, ToolCall, Usage


class OpenAICompatProvider(Provider):
    """Any OpenAI-format endpoint: OpenAI proper, vLLM, Ollama, Together,
    OpenRouter, Groq, LM Studio. Point `base_url` at the server and pick a
    `model` it exposes.

    No prompt caching: OpenAI-format APIs don't expose a portable cache-control
    knob. Use the Anthropic backend for the cached critic / judge role."""

    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAICompatProvider requires the 'openai' package. "
                "Install with: pip install 'ale[openai]'  (or: pip install openai)"
            ) from e
        self.model = model or os.environ.get("ALE_OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "not-needed"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        max_tokens: int = 2048,
    ) -> LLMResponse:
        oai_messages = [{"role": "system", "content": system}, *messages]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]
            kwargs["tool_choice"] = self._tool_choice(tool_choice)
        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tool_calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))

        u = response.usage
        usage = Usage(
            input_tokens=(u.prompt_tokens if u else 0) or 0,
            output_tokens=(u.completion_tokens if u else 0) or 0,
        )
        return LLMResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=response.choices[0].finish_reason or "",
        )

    def echo_assistant(self, response: LLMResponse) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": response.text or None}
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.input)},
                }
                for tc in response.tool_calls
            ]
        return msg

    def tool_results(self, results: list[tuple[str, str]]) -> list[dict[str, Any]]:
        return [
            {"role": "tool", "tool_call_id": tid, "content": content}
            for tid, content in results
        ]

    @staticmethod
    def _convert_tool(t: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }

    @staticmethod
    def _tool_choice(choice: str | dict[str, Any]) -> Any:
        if isinstance(choice, dict):
            if choice.get("type") == "tool":
                return {"type": "function", "function": {"name": choice.get("name", "")}}
            return choice
        if choice == "any":
            return "required"
        if choice in ("auto", "none"):
            return choice
        return {"type": "function", "function": {"name": choice}}
