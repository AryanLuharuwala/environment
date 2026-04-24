from __future__ import annotations

import random
from typing import Any

from .base import StepResult, Task


class GuessNumber(Task):
    name = "guess_number"
    description = (
        "A number from 1 to 100 is chosen. Use the `guess` tool to submit guesses. "
        "After each guess, you'll be told 'higher', 'lower', or 'correct'. "
        "Find the number in as few guesses as possible."
    )
    max_steps = 10
    tools = [
        {
            "name": "guess",
            "description": "Submit a guess for the secret number (1-100).",
            "input_schema": {
                "type": "object",
                "properties": {"value": {"type": "integer", "minimum": 1, "maximum": 100}},
                "required": ["value"],
            },
        }
    ]

    def reset(self) -> str:
        rng = random.Random(self.seed)
        self._target = rng.randint(1, 100)
        self._guesses = 0
        self._best_reward = 0.0
        self._solved = False
        return "I'm thinking of a number between 1 and 100. Start guessing."

    def step(self, tool_name: str, tool_input: dict[str, Any]) -> StepResult:
        if tool_name != "guess":
            return StepResult(f"Unknown tool: {tool_name}", 0.0, False)
        self._guesses += 1
        value = int(tool_input.get("value", 0))
        if value == self._target:
            self._solved = True
            # Optimal binary search needs ~7 guesses for 1-100. Reward decays linearly.
            self._best_reward = max(0.0, 1.0 - (self._guesses - 1) / 10.0)
            return StepResult(
                f"correct (in {self._guesses} guesses)",
                self._best_reward,
                True,
                {"guesses": self._guesses, "target": self._target},
            )
        hint = "higher" if value < self._target else "lower"
        return StepResult(hint, 0.0, False, {"guess": value})
