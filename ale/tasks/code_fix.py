from __future__ import annotations

from typing import Any

from .base import StepResult, Task


PROBLEMS = [
    {
        "broken": "def add(a, b):\n    return a - b\n",
        "tests": [("add(2, 3)", 5), ("add(0, 0)", 0), ("add(-1, 1)", 0)],
        "hint": "Function `add` is broken. It should add, not subtract.",
    },
    {
        "broken": "def is_even(n):\n    return n % 2 == 1\n",
        "tests": [("is_even(2)", True), ("is_even(3)", False), ("is_even(0)", True)],
        "hint": "Function `is_even` is broken. It returns True for odd numbers.",
    },
    {
        "broken": "def reverse(s):\n    return s\n",
        "tests": [("reverse('abc')", "cba"), ("reverse('')", ""), ("reverse('a')", "a")],
        "hint": "Function `reverse` should reverse a string.",
    },
    {
        "broken": "def factorial(n):\n    if n == 0:\n        return 0\n    return n * factorial(n - 1)\n",
        "tests": [("factorial(0)", 1), ("factorial(1)", 1), ("factorial(5)", 120)],
        "hint": "Function `factorial` has a wrong base case.",
    },
]


class CodeFix(Task):
    name = "code_fix"
    description = (
        "A Python function is broken. Submit a corrected version using the `submit` tool. "
        "The fixed function will be run against test cases. You may submit multiple times; "
        "after each submission you'll see which tests pass."
    )
    max_steps = 4
    tools = [
        {
            "name": "submit",
            "description": "Submit a corrected Python function definition.",
            "input_schema": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Full function source"}},
                "required": ["code"],
            },
        }
    ]

    def reset(self) -> str:
        problem = PROBLEMS[self.seed % len(PROBLEMS)]
        self._broken = problem["broken"]
        self._tests = problem["tests"]
        self._hint = problem["hint"]
        self._best_reward = 0.0
        return f"{self._hint}\n\nBroken code:\n```python\n{self._broken}```\n"

    def step(self, tool_name: str, tool_input: dict[str, Any]) -> StepResult:
        if tool_name != "submit":
            return StepResult(f"Unknown tool: {tool_name}", 0.0, False)
        code = tool_input.get("code", "")
        passed, total, failures = self._run_tests(code)
        reward = passed / total if total else 0.0
        self._best_reward = max(self._best_reward, reward)
        if passed == total:
            return StepResult(f"All {total} tests passed.", reward, True, {"passed": passed})
        msg = f"{passed}/{total} tests passed.\n" + "\n".join(failures[:3])
        return StepResult(msg, 0.0, False, {"passed": passed})

    def _run_tests(self, code: str) -> tuple[int, int, list[str]]:
        # Tests are author-controlled string literals (see PROBLEMS); eval is bounded
        # to the namespace produced by the submitted function definition.
        ns: dict[str, Any] = {}
        try:
            exec(code, ns)
        except Exception as e:
            return 0, len(self._tests), [f"compile error: {e}"]
        failures = []
        passed = 0
        for expr, expected in self._tests:
            try:
                actual = eval(expr, ns)
                if actual == expected:
                    passed += 1
                else:
                    failures.append(f"{expr} == {actual!r} (expected {expected!r})")
            except Exception as e:
                failures.append(f"{expr} raised {e}")
        return passed, len(self._tests), failures
