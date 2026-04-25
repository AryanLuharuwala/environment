from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..datasets import Example


@dataclass
class Score:
    name: str
    value: float          # in [0, 1]
    detail: str = ""


@dataclass
class VerifierResult:
    scores: list[Score]
    reward: float
    feedback: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Verifier(ABC):
    name: str = "verifier"
    weight: float = 1.0

    @abstractmethod
    def check(self, prediction: str, example: Example) -> Score: ...


def composite(
    scores: list[Score],
    weights: dict[str, float] | None = None,
    hard_fail: set[str] | None = None,
) -> float:
    """Weighted mean of verifier scores. Names in `hard_fail` (e.g. "ast")
    cap the total to 0 when their score is 0 — used to penalize broken code
    no matter how good the rest of the prediction looks."""
    if not scores:
        return 0.0
    weights = weights or {}
    hard_fail = hard_fail or set()
    for s in scores:
        if s.name in hard_fail and s.value == 0.0:
            return 0.0
    weighted = [(s.value, weights.get(s.name, s.__dict__.get("_weight", 1.0))) for s in scores]
    total_w = sum(w for _, w in weighted) or 1.0
    return sum(v * w for v, w in weighted) / total_w
