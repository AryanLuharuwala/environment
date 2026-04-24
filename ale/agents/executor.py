from __future__ import annotations

from ..providers import Provider
from .base import Agent

EXECUTOR_SYSTEM = """You are the Executor in a multi-agent team. You act on a task by calling the provided tools.

Rules:
- Always call exactly one tool per turn until the task is solved.
- Read each observation carefully before deciding the next tool call.
- Be efficient. Wasted steps lower the reward.
- If a Planner has provided a plan, follow it but adapt when observations contradict it.
"""


def executor(provider: Provider | None = None, max_tokens: int = 2048) -> Agent:
    kwargs: dict = {
        "name": "executor",
        "system_prompt": EXECUTOR_SYSTEM,
        "max_tokens": max_tokens,
    }
    if provider is not None:
        kwargs["provider"] = provider
    return Agent(**kwargs)
