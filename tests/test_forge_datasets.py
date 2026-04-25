"""GitCommitSource + JsonlSource. Builds a real temp git repo so the
subprocess git plumbing in `forge.datasets.git_commits` is exercised
end-to-end without any network."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from forge.datasets import GitCommitSource, JsonlSource


def _has_git() -> bool:
    return shutil.which("git") is not None


pytestmark = pytest.mark.skipif(not _has_git(), reason="git not available")


def _git(repo: Path, *args: str) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def _build_demo_repo(tmp_path: Path) -> Path:
    """Create a tiny throwaway git repo with two commits. If the environment
    forces commit signing (and signing fails), skip — the GitCommitSource
    code under test is environment-agnostic; the fixture isn't."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@x")
    _git(repo, "config", "user.name", "t")
    try:
        (repo / "a.py").write_text("def f(): return 1\n")
        _git(repo, "add", "a.py")
        _git(repo, "commit", "-q", "-m", "add a")
        (repo / "a.py").write_text("def f(): return 2\n")
        _git(repo, "add", "a.py")
        _git(repo, "commit", "-q", "-m", "tweak f")
    except subprocess.CalledProcessError as e:
        pytest.skip(f"git commit blocked in this environment: exit {e.returncode}")
    return repo


def test_git_commit_source_yields_examples_per_commit(tmp_path: Path):
    repo = _build_demo_repo(tmp_path)

    src = GitCommitSource(repo=repo, max_commits=10, suffix=".py")
    examples = list(src.iter_examples())
    assert len(examples) == 2
    # Newest commit comes first in `git log`.
    assert "tweak f" in examples[0].metadata["message"]
    assert examples[0].target.strip() == "def f(): return 2"
    assert examples[1].target.strip() == "def f(): return 1"
    assert examples[0].metadata["language"] == "python"
    assert examples[0].metadata["sha"]
    assert examples[0].id.endswith("a.py")


def test_jsonl_source_loads_metadata(tmp_path: Path):
    path = tmp_path / "ds.jsonl"
    rows = [
        {"id": "q1", "context": "What is X?", "target": "X is Y", "topic": "law"},
        {"id": "q2", "context": "What is Z?", "target": "Z is W"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    src = JsonlSource(path=path)
    examples = list(src.iter_examples())
    assert [e.id for e in examples] == ["q1", "q2"]
    assert examples[0].metadata == {"topic": "law"}


def test_jsonl_source_rejects_missing_fields(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "x", "context": "y"}) + "\n")  # missing target
    with pytest.raises(ValueError):
        list(JsonlSource(path=path).iter_examples())
