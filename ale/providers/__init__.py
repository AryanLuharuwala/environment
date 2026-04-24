from __future__ import annotations

import os

from .anthropic_p import AnthropicProvider
from .base import LLMResponse, Provider, ToolCall, Usage


def make_provider(kind: str | None = None, model: str | None = None) -> Provider:
    """Build a provider by name. Defaults to $ALE_PROVIDER or 'anthropic'."""
    kind = kind or os.environ.get("ALE_PROVIDER", "anthropic")
    if kind == "anthropic":
        return AnthropicProvider(model=model)
    if kind in ("openai", "openai_compat"):
        from .openai_p import OpenAICompatProvider
        return OpenAICompatProvider(model=model)
    raise ValueError(f"Unknown provider: {kind!r}. Known: anthropic, openai")


__all__ = [
    "AnthropicProvider",
    "LLMResponse",
    "Provider",
    "ToolCall",
    "Usage",
    "make_provider",
]
