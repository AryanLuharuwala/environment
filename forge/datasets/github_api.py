"""STUB: GitHub API source.

The Source abstraction is shape-compatible with a remote-fetch path. Filling
this in needs (a) a token (`GITHUB_TOKEN`), (b) a rate-limit-aware fetcher,
and (c) decisions about which PRs / commits qualify (size, language,
review-approved, etc.). Out of scope for the first pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from .base import Example


@dataclass
class GitHubApiSource:
    repo: str  # "owner/name"
    token: str | None = None

    def iter_examples(self, **_filters: Any) -> Iterator[Example]:
        raise NotImplementedError(
            "GitHubApiSource is a placeholder. Use GitCommitSource against a local "
            "clone for now: `git clone https://github.com/owner/name && "
            "forge collect-coder --repo name ...`"
        )
