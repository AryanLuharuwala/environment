from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .base import Example


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _list_commits(repo: Path, max_commits: int, branch: str | None) -> list[str]:
    args = ["log", f"-n{max_commits}", "--no-merges", "--format=%H"]
    if branch:
        args.append(branch)
    return [line for line in _git(repo, *args).splitlines() if line]


def _changed_files(repo: Path, sha: str, suffix: str | None) -> list[str]:
    out = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", sha)
    files = [f for f in out.splitlines() if f]
    if suffix:
        files = [f for f in files if f.endswith(suffix)]
    return files


def _show(repo: Path, ref: str) -> str:
    """git show <ref>; returns empty string for non-existent refs (e.g.
    pre-commit version of a brand-new file)."""
    result = subprocess.run(
        ["git", "-C", str(repo), "show", ref],
        capture_output=True, text=True, check=False,
    )
    return result.stdout if result.returncode == 0 else ""


def _commit_message(repo: Path, sha: str) -> str:
    return _git(repo, "log", "-n1", "--format=%B", sha).strip()


def _diff(repo: Path, sha: str, path: str) -> str:
    return _git(repo, "diff", f"{sha}^", sha, "--", path)


@dataclass
class GitCommitSource:
    """Mine commits from a local git checkout. One Example per (commit, file)
    pair: the model sees the pre-commit file + commit message and is asked
    what the post-commit file should look like."""

    repo: Path
    max_commits: int = 50
    branch: str | None = None
    suffix: str = ".py"
    max_target_lines: int = 200  # skip massive rewrites; bad signal for diff-prediction

    def iter_examples(self, **_filters) -> Iterator[Example]:
        repo = Path(self.repo).resolve()
        if not (repo / ".git").exists():
            raise FileNotFoundError(f"not a git repo: {repo}")
        for sha in _list_commits(repo, self.max_commits, self.branch):
            try:
                msg = _commit_message(repo, sha)
            except RuntimeError:
                continue
            for path in _changed_files(repo, sha, self.suffix):
                pre = _show(repo, f"{sha}^:{path}")
                post = _show(repo, f"{sha}:{path}")
                if not post:
                    continue  # file deleted in this commit
                if post.count("\n") > self.max_target_lines:
                    continue
                try:
                    diff = _diff(repo, sha, path)
                except RuntimeError:
                    diff = ""
                example_id = f"{repo.name}@{sha[:10]}:{path}"
                context = (
                    f"Repository: {repo.name}\n"
                    f"File: {path}\n"
                    f"Commit message: {msg}\n\n"
                    f"--- Pre-commit version ---\n{pre or '(new file)'}\n"
                )
                yield Example(
                    id=example_id,
                    context=context,
                    target=post,
                    metadata={
                        "repo": repo.name,
                        "sha": sha,
                        "path": path,
                        "message": msg,
                        "diff": diff,
                        "language": "python" if path.endswith(".py") else "",
                    },
                )
