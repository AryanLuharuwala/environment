from __future__ import annotations

import difflib

from ..datasets import Example
from .base import Score, Verifier


def _normalize(s: str) -> str:
    # Collapse runs of whitespace and strip blank lines so cosmetic-only
    # mismatches don't dominate the score.
    lines = [line.rstrip() for line in s.splitlines()]
    lines = [line for line in lines if line.strip()]
    return "\n".join(lines)


class DiffSimilarityVerifier(Verifier):
    """How close is the predicted code to the ground-truth post-commit code?

    Uses difflib.SequenceMatcher.ratio() over normalized text. Cheap,
    deterministic, no external deps. Not a great signal for big rewrites
    but good enough to drive preferences when paired with AST + tests."""

    name = "diff_similarity"
    weight = 0.4

    def check(self, prediction: str, example: Example) -> Score:
        if not prediction.strip():
            return Score(name=self.name, value=0.0, detail="empty prediction")
        a = _normalize(prediction)
        b = _normalize(example.target)
        ratio = difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()
        return Score(name=self.name, value=ratio, detail=f"ratio={ratio:.3f}")
