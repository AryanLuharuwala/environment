from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..datasets import Example
from .base import Score, Verifier


@dataclass
class PytestVerifier(Verifier):
    """Run pytest against the predicted code in a sandbox.

    SAFETY: This executes code emitted by an LLM. It is opt-in and runs in a
    subprocess inside a TemporaryDirectory with a timeout. Do not enable it
    for predictions you wouldn't run yourself.

    Expected `Example.metadata`:
        - `module_filename`: where to write the prediction (e.g. "mod.py")
        - `test_source`: pytest-compatible test file content

    If either is missing, the verifier returns 0.0 with detail="skipped"
    so it can be left in the verifier list without causing failures.
    """

    name: str = "tests"
    weight: float = 0.2
    timeout_s: float = 30.0

    def check(self, prediction: str, example: Example) -> Score:
        meta = example.metadata
        module_filename = meta.get("module_filename")
        test_source = meta.get("test_source")
        if not (module_filename and test_source):
            return Score(name=self.name, value=0.0, detail="skipped: no tests in example")

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            (tmpdir / module_filename).write_text(prediction)
            (tmpdir / "test_predicted.py").write_text(test_source)
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "test_predicted.py"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
            except subprocess.TimeoutExpired:
                return Score(name=self.name, value=0.0, detail=f"timeout after {self.timeout_s}s")
            except FileNotFoundError:
                return Score(name=self.name, value=0.0, detail="pytest not installed")

        if result.returncode == 0:
            return Score(name=self.name, value=1.0, detail="all tests passed")
        # pytest exit codes: 0 success, 1 test failures, others = collection/run errors
        tail = (result.stdout + result.stderr).strip().splitlines()
        return Score(
            name=self.name,
            value=0.0,
            detail=f"pytest exit={result.returncode}: {' | '.join(tail[-3:])[:200]}",
        )


def have_pytest() -> bool:
    return shutil.which(sys.executable) is not None and \
        subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True, text=True,
        ).returncode == 0
