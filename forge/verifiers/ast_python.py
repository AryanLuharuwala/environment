from __future__ import annotations

import ast

from ..datasets import Example
from .base import Score, Verifier


class PythonAstVerifier(Verifier):
    """Penalize predictions that don't parse as Python.

    Score is binary: 1.0 if `ast.parse` succeeds, 0.0 otherwise. Wired as a
    hard-fail signal in the composite reward so AST errors zero the whole
    rollout — the coding reward system the user asked for."""

    name = "ast"
    weight = 0.4

    def check(self, prediction: str, example: Example) -> Score:
        if not prediction.strip():
            return Score(name=self.name, value=0.0, detail="empty prediction")
        try:
            ast.parse(prediction)
            return Score(name=self.name, value=1.0, detail="parsed")
        except SyntaxError as e:
            return Score(name=self.name, value=0.0, detail=f"SyntaxError: {e.msg} at line {e.lineno}")
