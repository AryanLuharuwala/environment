from __future__ import annotations

import random
from typing import Any

from .base import StepResult, Task


class TextNav(Task):
    name = "text_nav"
    description = (
        "Navigate a 5x5 grid from the start position to the goal. "
        "Use `move` with direction 'north', 'south', 'east', or 'west'. "
        "Walls block movement. After each move you receive your new position "
        "and which neighboring cells are walls. Reach the goal in few steps."
    )
    max_steps = 25
    tools = [
        {
            "name": "move",
            "description": "Move one cell in a cardinal direction.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["north", "south", "east", "west"]}
                },
                "required": ["direction"],
            },
        }
    ]

    SIZE = 5
    DELTAS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}

    def reset(self) -> str:
        rng = random.Random(self.seed)
        self._pos = (0, 0)
        self._goal = (self.SIZE - 1, self.SIZE - 1)
        # Sprinkle a few walls; keep them off start/goal.
        all_cells = [(x, y) for x in range(self.SIZE) for y in range(self.SIZE)]
        all_cells = [c for c in all_cells if c != self._pos and c != self._goal]
        self._walls = set(rng.sample(all_cells, k=4))
        self._steps = 0
        self._best_reward = 0.0
        # Manhattan distance is a lower bound; reward optimal-ish paths.
        self._optimal = abs(self._goal[0] - self._pos[0]) + abs(self._goal[1] - self._pos[1])
        return self._render()

    def _render(self) -> str:
        wall_dirs = []
        x, y = self._pos
        for d, (dx, dy) in self.DELTAS.items():
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.SIZE and 0 <= ny < self.SIZE) or (nx, ny) in self._walls:
                wall_dirs.append(d)
        walls_str = ", ".join(wall_dirs) if wall_dirs else "none"
        return (
            f"position={self._pos} goal={self._goal} "
            f"walls_to=[{walls_str}] step={self._steps}/{self.max_steps}"
        )

    def step(self, tool_name: str, tool_input: dict[str, Any]) -> StepResult:
        if tool_name != "move":
            return StepResult(f"Unknown tool: {tool_name}", 0.0, False)
        self._steps += 1
        direction = tool_input.get("direction", "")
        if direction not in self.DELTAS:
            return StepResult(f"Invalid direction: {direction}\n{self._render()}", 0.0, False)
        dx, dy = self.DELTAS[direction]
        nx, ny = self._pos[0] + dx, self._pos[1] + dy
        if not (0 <= nx < self.SIZE and 0 <= ny < self.SIZE):
            return StepResult(f"Out of bounds.\n{self._render()}", 0.0, False)
        if (nx, ny) in self._walls:
            return StepResult(f"Blocked by wall.\n{self._render()}", 0.0, False)
        self._pos = (nx, ny)
        if self._pos == self._goal:
            # Reward 1.0 at optimal length, decays as path gets longer.
            efficiency = self._optimal / max(self._steps, self._optimal)
            self._best_reward = efficiency
            return StepResult(
                f"reached goal in {self._steps} steps",
                self._best_reward,
                True,
                {"steps": self._steps, "optimal": self._optimal},
            )
        return StepResult(self._render(), 0.0, False)
