"""Provider tests. No network calls — both providers are tested against a
scripted fake client that returns canned responses."""
from __future__ import annotations

import json
from types import SimpleNamespace

from ale.providers import AnthropicProvider, LLMResponse
from ale.providers.base import ToolCall


# --- Anthropic -----------------------------------------------------------

class _FakeAnthropicMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response):
        self.messages = _FakeAnthropicMessages(response)


def _anthropic_response():
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="thinking..."),
            SimpleNamespace(
                type="tool_use",
                id="toolu_1",
                name="guess",
                input={"value": 50},
            ),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=0,
        ),
    )


def test_anthropic_sends_cache_control():
    p = AnthropicProvider(model="claude-opus-4-7", client=_FakeAnthropicClient(_anthropic_response()))
    p.complete(system="system prompt", messages=[{"role": "user", "content": "hi"}])
    sent = p.client.messages.last_kwargs
    assert isinstance(sent["system"], list)
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_parses_tool_use():
    p = AnthropicProvider(model="claude-opus-4-7", client=_FakeAnthropicClient(_anthropic_response()))
    r = p.complete(system="s", messages=[{"role": "user", "content": "hi"}], tools=[{"name": "guess", "description": "", "input_schema": {}}])
    assert r.text == "thinking..."
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "guess"
    assert r.tool_calls[0].input == {"value": 50}
    assert r.usage.cache_read == 100


def test_anthropic_echo_and_tool_results():
    p = AnthropicProvider(model="claude-opus-4-7", client=_FakeAnthropicClient(_anthropic_response()))
    resp = LLMResponse(text="t", tool_calls=[ToolCall(id="toolu_1", name="guess", input={"value": 50})])
    echoed = p.echo_assistant(resp)
    assert echoed["role"] == "assistant"
    types = [b["type"] for b in echoed["content"]]
    assert types == ["text", "tool_use"]
    results = p.tool_results([("toolu_1", "lower")])
    assert results[0]["role"] == "user"
    assert results[0]["content"][0] == {"type": "tool_result", "tool_use_id": "toolu_1", "content": "lower"}


# --- OpenAI-compat --------------------------------------------------------

class _FakeOpenAICompletions:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeOpenAIClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=_FakeOpenAICompletions(response))


def _openai_response():
    tc = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="guess", arguments=json.dumps({"value": 42})),
    )
    msg = SimpleNamespace(content="I'll guess.", tool_calls=[tc])
    choice = SimpleNamespace(message=msg, finish_reason="tool_calls")
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=7),
    )


def test_openai_converts_tools_and_parses_result():
    # Import inside test so skipping is easy if openai isn't installed in CI.
    try:
        from ale.providers.openai_p import OpenAICompatProvider
    except ImportError:
        import pytest
        pytest.skip("openai not installed")

    # Monkey-build: bypass __init__ that requires the openai SDK and install our fake.
    p = OpenAICompatProvider.__new__(OpenAICompatProvider)
    p.model = "test"
    p.client = _FakeOpenAIClient(_openai_response())

    tools = [{"name": "guess", "description": "Guess a number.", "input_schema": {"type": "object"}}]
    r = p.complete(system="sys", messages=[{"role": "user", "content": "go"}], tools=tools, tool_choice="any")
    sent = p.client.chat.completions.last_kwargs
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["tools"][0]["type"] == "function"
    assert sent["tools"][0]["function"]["name"] == "guess"
    assert sent["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert sent["tool_choice"] == "required"

    assert r.text == "I'll guess."
    assert r.tool_calls[0].input == {"value": 42}


def test_openai_echo_and_tool_results():
    try:
        from ale.providers.openai_p import OpenAICompatProvider
    except ImportError:
        import pytest
        pytest.skip("openai not installed")
    p = OpenAICompatProvider.__new__(OpenAICompatProvider)
    p.model = "test"
    p.client = None  # unused in these helpers
    resp = LLMResponse(text="t", tool_calls=[ToolCall(id="call_1", name="guess", input={"value": 1})])
    echoed = p.echo_assistant(resp)
    assert echoed["role"] == "assistant"
    assert echoed["tool_calls"][0]["function"]["name"] == "guess"
    results = p.tool_results([("call_1", "higher")])
    assert results == [{"role": "tool", "tool_call_id": "call_1", "content": "higher"}]
