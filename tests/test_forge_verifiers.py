"""AST + diff-similarity + tests + grounding (with scripted fake provider)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from forge.datasets import Example
from forge.verifiers import (
    DiffSimilarityVerifier,
    GroundingVerifier,
    PythonAstVerifier,
    PytestVerifier,
    composite,
)


def _ex(target: str = "", **meta: Any) -> Example:
    return Example(id="t", context="c", target=target, metadata=dict(meta))


def test_ast_passes_valid_python():
    s = PythonAstVerifier().check("def f(x): return x + 1\n", _ex())
    assert s.value == 1.0


def test_ast_fails_on_syntax_error():
    s = PythonAstVerifier().check("def f(:\n", _ex())
    assert s.value == 0.0
    assert "SyntaxError" in s.detail


def test_ast_zero_for_empty_prediction():
    s = PythonAstVerifier().check("   \n", _ex())
    assert s.value == 0.0


def test_diff_similarity_identical_is_one():
    code = "def f():\n    return 1\n"
    s = DiffSimilarityVerifier().check(code, _ex(target=code))
    assert s.value == 1.0


def test_diff_similarity_normalizes_whitespace():
    a = "def f():\n    return 1\n"
    b = "def f():\n    return 1\n\n\n"  # extra blank lines
    s = DiffSimilarityVerifier().check(a, _ex(target=b))
    assert s.value == 1.0


def test_diff_similarity_drops_for_unrelated_code():
    s = DiffSimilarityVerifier().check(
        "def f(): return 1",
        _ex(target="class K:\n    def m(self, x, y, z): pass"),
    )
    assert s.value < 0.5


def test_composite_hard_fails_on_ast_zero():
    from forge.verifiers import Score
    scores = [
        Score(name="ast", value=0.0),
        Score(name="diff_similarity", value=1.0),
    ]
    assert composite(scores, weights={"ast": 0.4, "diff_similarity": 0.6}, hard_fail={"ast"}) == 0.0


def test_composite_weighted_mean_when_no_hard_fail():
    from forge.verifiers import Score
    scores = [
        Score(name="ast", value=1.0),
        Score(name="diff_similarity", value=0.5),
    ]
    r = composite(scores, weights={"ast": 0.4, "diff_similarity": 0.6}, hard_fail={"ast"})
    assert abs(r - (1.0 * 0.4 + 0.5 * 0.6)) < 1e-9


def test_pytest_verifier_passes_when_tests_pass(tmp_path: Path):
    if not shutil.which(sys.executable):
        return
    code = "def add(a, b): return a + b\n"
    test_src = "from mod import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    ex = _ex(module_filename="mod.py", test_source=test_src)
    s = PytestVerifier().check(code, ex)
    # Pytest may or may not be installed; allow skipped detail too.
    if "skipped" in s.detail or "not installed" in s.detail:
        return
    assert s.value == 1.0


def test_pytest_verifier_fails_when_tests_fail(tmp_path: Path):
    code = "def add(a, b): return a - b\n"  # bug
    test_src = "from mod import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    ex = _ex(module_filename="mod.py", test_source=test_src)
    s = PytestVerifier().check(code, ex)
    if "not installed" in s.detail:
        return
    assert s.value == 0.0


def test_pytest_verifier_skipped_without_tests():
    s = PytestVerifier().check("def f(): pass\n", _ex())
    assert s.value == 0.0
    assert "skipped" in s.detail


# --- Grounding (fake provider) ------------------------------------------

class _FakeProvider:
    name = "fake"
    model = "fake"

    def __init__(self, response_text: str):
        self._response = response_text
        self.last_user: str | None = None

    def complete(self, system: str, messages, tools=None, tool_choice="auto", max_tokens=2048):
        self.last_user = messages[0]["content"]
        from ale.providers.base import LLMResponse, Usage
        return LLMResponse(text=self._response, tool_calls=[], usage=Usage())

    def echo_assistant(self, response):
        return {"role": "assistant", "content": response.text}

    def tool_results(self, results):
        return []


def test_grounding_parses_supported_score():
    fake = _FakeProvider("SUPPORT: 0.85\nISSUES: minor over-claim about scope\n")
    v = GroundingVerifier(provider=fake)
    s = v.check("Some answer.", _ex(sources=["src 1", "src 2"], question="Q"))
    assert abs(s.value - 0.85) < 1e-9
    assert "minor over-claim" in s.detail


def test_grounding_skips_when_no_sources():
    fake = _FakeProvider("SUPPORT: 1.0\nISSUES: none\n")
    v = GroundingVerifier(provider=fake)
    s = v.check("Some answer.", _ex())  # no sources
    assert s.value == 0.0
    assert "skipped" in s.detail


def test_grounding_handles_unparseable_response():
    fake = _FakeProvider("blah blah no score here")
    v = GroundingVerifier(provider=fake)
    s = v.check("Answer", _ex(sources=["s"]))
    assert s.value == 0.0
