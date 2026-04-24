from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable

from .trace import Trace


@dataclass
class TaskResult:
    task: str
    mode: str
    model: str
    rewards: list[float]
    steps: list[int]

    @property
    def n(self) -> int:
        return len(self.rewards)

    @property
    def mean_reward(self) -> float:
        return statistics.mean(self.rewards) if self.rewards else 0.0

    @property
    def success_rate(self) -> float:
        return sum(1 for r in self.rewards if r > 0) / self.n if self.n else 0.0

    @property
    def mean_steps(self) -> float:
        return statistics.mean(self.steps) if self.steps else 0.0


def aggregate(traces: Iterable[Trace]) -> list[TaskResult]:
    groups: dict[tuple[str, str, str], TaskResult] = {}
    for t in traces:
        key = (t.task, t.mode, t.model)
        if key not in groups:
            groups[key] = TaskResult(task=t.task, mode=t.mode, model=t.model, rewards=[], steps=[])
        groups[key].rewards.append(t.final_reward)
        groups[key].steps.append(t.steps)
    return list(groups.values())


def render_table(results: list[TaskResult]) -> str:
    header = f"{'task':<16}{'mode':<8}{'n':<4}{'reward':>8}{'success':>10}{'steps':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in sorted(results, key=lambda x: (x.task, x.mode)):
        lines.append(
            f"{r.task:<16}{r.mode:<8}{r.n:<4}"
            f"{r.mean_reward:>8.3f}{r.success_rate:>10.0%}{r.mean_steps:>8.1f}"
        )
    return "\n".join(lines)
