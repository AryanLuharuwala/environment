from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class Task:
    """A scorable, deterministic environment for agents to act in.

    Subclasses define a tool surface and step semantics. The agent perceives
    the world only through `reset()` and the observations returned by `step()`.
    A run ends when `done=True` or `max_steps` is exhausted; the final episode
    reward in [0, 1] is reported by the harness.
    """

    name: str = "task"
    description: str = ""
    max_steps: int = 20
    tools: list[dict[str, Any]] = []

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def reset(self) -> str:
        raise NotImplementedError

    def step(self, tool_name: str, tool_input: dict[str, Any]) -> StepResult:
        raise NotImplementedError

    def episode_reward(self) -> float:
        """Final scalar in [0, 1]. Default: best reward seen so far."""
        return getattr(self, "_best_reward", 0.0)
