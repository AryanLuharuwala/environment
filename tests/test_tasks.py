"""Offline tests for task environments. No API calls."""
from __future__ import annotations

from ale.tasks import make


def test_guess_number_solves():
    task = make("guess_number", seed=7)
    obs = task.reset()
    assert "1 and 100" in obs
    lo, hi = 1, 100
    done = False
    while not done:
        mid = (lo + hi) // 2
        r = task.step("guess", {"value": mid})
        done = r.done
        if "higher" in r.observation:
            lo = mid + 1
        elif "lower" in r.observation:
            hi = mid - 1
    assert task.episode_reward() > 0


def test_text_nav_reaches_goal():
    task = make("text_nav", seed=3)
    task.reset()
    # Naive east-then-south: may bump walls but should eventually hit the goal
    # within max_steps for most seeds; we just check step mechanics.
    r = task.step("move", {"direction": "east"})
    assert "position=" in r.observation


def test_code_fix_all_pass():
    task = make("code_fix", seed=0)  # add(a, b)
    task.reset()
    r = task.step("submit", {"code": "def add(a, b):\n    return a + b\n"})
    assert r.done is True
    assert task.episode_reward() == 1.0


def test_tool_math_arithmetic():
    task = make("tool_math", seed=42)
    task.reset()
    r = task.step("add", {"a": 2, "b": 3})
    assert r.observation == "5"
    r = task.step("multiply", {"a": 4, "b": 5})
    assert r.observation == "20"


def test_unknown_tool_returns_error():
    task = make("guess_number", seed=0)
    task.reset()
    r = task.step("nope", {})
    assert "Unknown tool" in r.observation
    assert r.done is False
