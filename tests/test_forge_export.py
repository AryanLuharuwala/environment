from __future__ import annotations

import json
from pathlib import Path

from forge import Example, RolloutResult, pair_to_dpo, write_dpo_jsonl
from forge.verifiers import Score


def _result(label: str, prediction: str, reward: float, scores: list[Score]) -> RolloutResult:
    return RolloutResult(
        example_id="ex1", config_label=label, prediction=prediction,
        scores=scores, reward=reward, retrieved_ids=["doc-a"],
    )


def test_pair_to_dpo_picks_higher_reward_as_chosen():
    ex = Example(id="ex1", context="prompt", target="t", metadata={"repo": "r"})
    a = _result("with_rag", "good code", 0.9, [Score(name="ast", value=1.0), Score(name="diff_similarity", value=0.8)])
    b = _result("no_rag", "bad code", 0.3, [Score(name="ast", value=1.0), Score(name="diff_similarity", value=0.0)])

    row = pair_to_dpo(ex, a, b)
    assert row is not None
    assert row["prompt"] == "prompt"
    assert row["chosen"] == "good code"
    assert row["rejected"] == "bad code"
    assert row["metadata"]["chosen_label"] == "with_rag"
    assert row["metadata"]["rejected_label"] == "no_rag"
    assert row["metadata"]["reward_chosen"] == 0.9
    assert row["metadata"]["repo"] == "r"  # forwarded from example.metadata
    assert row["metadata"]["scores_chosen"][0]["name"] == "ast"


def test_pair_to_dpo_drops_ties():
    ex = Example(id="ex1", context="p", target="t")
    a = _result("a", "x", 0.5, [])
    b = _result("b", "y", 0.5, [])
    assert pair_to_dpo(ex, a, b) is None


def test_write_dpo_jsonl_roundtrip(tmp_path: Path):
    rows = [
        {"prompt": "p", "chosen": "c", "rejected": "r", "metadata": {"k": 1}},
        {"prompt": "p2", "chosen": "c2", "rejected": "r2", "metadata": {"k": 2}},
    ]
    out = tmp_path / "pairs.jsonl"
    write_dpo_jsonl(rows, out)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["prompt"] == "p2"


def test_pair_strips_large_sources_from_metadata():
    """sources field can be huge; export should not duplicate it."""
    ex = Example(id="ex1", context="p", target="t",
                 metadata={"sources": ["very long source text" * 1000], "topic": "law"})
    a = _result("rag", "ans1", 0.7, [])
    b = _result("plain", "ans2", 0.4, [])
    row = pair_to_dpo(ex, a, b)
    assert "sources" not in row["metadata"]
    assert row["metadata"]["topic"] == "law"
