from __future__ import annotations

import random
from typing import Any

from .base import StepResult, Task


class ToolMath(Task):
    name = "tool_math"
    description = (
        "Solve a multi-step arithmetic word problem using the available tools. "
        "Use `add`, `multiply`, and `subtract` for intermediate computations, then "
        "submit the final answer with `final_answer`. You may not compute in your head."
    )
    max_steps = 8
    tools = [
        {
            "name": "add",
            "description": "Return a + b.",
            "input_schema": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
        {
            "name": "subtract",
            "description": "Return a - b.",
            "input_schema": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
        {
            "name": "multiply",
            "description": "Return a * b.",
            "input_schema": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
        {
            "name": "final_answer",
            "description": "Submit the final numeric answer.",
            "input_schema": {
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
            },
        },
    ]

    def reset(self) -> str:
        rng = random.Random(self.seed)
        a, b, c, d = (rng.randint(2, 12) for _ in range(4))
        # Problem: a baskets of b apples; eat c, gain d more.
        self._answer = a * b - c + d
        self._best_reward = 0.0
        return (
            f"There are {a} baskets, each with {b} apples. "
            f"You eat {c} apples, then someone gives you {d} more. "
            f"How many apples do you have?"
        )

    def step(self, tool_name: str, tool_input: dict[str, Any]) -> StepResult:
        if tool_name == "add":
            r = tool_input["a"] + tool_input["b"]
            return StepResult(f"{r}", 0.0, False)
        if tool_name == "subtract":
            r = tool_input["a"] - tool_input["b"]
            return StepResult(f"{r}", 0.0, False)
        if tool_name == "multiply":
            r = tool_input["a"] * tool_input["b"]
            return StepResult(f"{r}", 0.0, False)
        if tool_name == "final_answer":
            value = tool_input.get("value")
            correct = value == self._answer
            self._best_reward = 1.0 if correct else 0.0
            return StepResult(
                "correct" if correct else f"incorrect (expected {self._answer})",
                self._best_reward,
                True,
                {"answer": self._answer, "submitted": value},
            )
        return StepResult(f"Unknown tool: {tool_name}", 0.0, False)
