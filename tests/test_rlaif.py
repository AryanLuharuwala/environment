"""RLAIF export + preference tests. Works offline using synthetic traces."""
from __future__ import annotations

from pathlib import Path

from ale.harness import Trace, TraceEvent
from ale.rlaif import (
    Preference,
    pair_to_dpo,
    prompt_text,
    trajectory_text,
    write_dpo_jsonl,
)
from ale.rlaif.preference import judge_pair
from ale.rlaif.rollout import RolloutPair


def _trace(task: str, seed: int, reward: float, ops: list[tuple[str, dict]]) -> Trace:
    t = Trace(task=task, seed=seed, mode="single", model="fake")
    t.events.append(TraceEvent(kind="observation", data={"text": "init", "initial": True}))
    for kind, data in ops:
        t.events.append(TraceEvent(kind=kind, data=data))
    t.final_reward = reward
    t.steps = sum(1 for k, _ in ops if k == "tool_use")
    return t


def test_trajectory_text_renders_intent_and_calls():
    t = _trace("guess_number", 0, 1.0, [
        ("message", {"text": "I'll binary-search."}),
        ("tool_use", {"name": "guess", "input": {"value": 50}}),
        ("observation", {"text": "lower", "reward": 0.0, "done": False}),
        ("tool_use", {"name": "guess", "input": {"value": 25}}),
        ("observation", {"text": "correct", "reward": 1.0, "done": True}),
    ])
    text = trajectory_text(t)
    assert "think: I'll binary-search." in text
    assert 'call: guess({"value":50})' in text
    assert "obs:  correct" in text


def test_pair_to_dpo_skips_ties_and_tags_metadata(tmp_path: Path):
    a = _trace("guess_number", 0, 1.0, [("tool_use", {"name": "guess", "input": {"value": 50}})])
    b = _trace("guess_number", 0, 0.0, [("tool_use", {"name": "guess", "input": {"value": 1}})])
    pair = RolloutPair(task_name="guess_number", seed=0, a=a, a_label="multi", b=b, b_label="single")

    pref_win = Preference("A", "higher reward", 1.0, 0.0, "reward")
    ex = pair_to_dpo(pair, pref_win, task_description="Guess a number 1-100.")
    assert ex is not None
    assert ex["metadata"]["chosen_label"] == "multi"
    assert ex["metadata"]["reward_chosen"] == 1.0
    assert "prompt" in ex and "chosen" in ex and "rejected" in ex

    pref_tie = Preference("tie", "equal", 1.0, 1.0, "reward")
    assert pair_to_dpo(pair, pref_tie, task_description="desc") is None

    # Writer roundtrip
    path = tmp_path / "pairs.jsonl"
    write_dpo_jsonl([ex], path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    import json
    row = json.loads(lines[0])
    assert row["metadata"]["task"] == "guess_number"


def test_judge_uses_reward_when_unequal():
    a = _trace("t", 0, 0.8, [])
    b = _trace("t", 0, 0.3, [])
    pair = RolloutPair(task_name="t", seed=0, a=a, a_label="x", b=b, b_label="y")
    pref = judge_pair(pair, judge=None, require_llm_for_equal_rewards=False)
    assert pref.winner == "A"
    assert pref.source == "reward"


def test_judge_reports_tie_when_equal_and_no_llm():
    a = _trace("t", 0, 0.5, [])
    b = _trace("t", 0, 0.5, [])
    pair = RolloutPair(task_name="t", seed=0, a=a, a_label="x", b=b, b_label="y")
    pref = judge_pair(pair, judge=None, require_llm_for_equal_rewards=False)
    assert pref.winner == "tie"


def test_prompt_text_includes_plan_if_present():
    t = _trace("t", 0, 1.0, [("plan", {"text": "1. Guess 50"})])
    p = prompt_text(t, task_description="Guess a number 1-100.")
    assert "Guess a number 1-100" in p
    assert "1. Guess 50" in p
