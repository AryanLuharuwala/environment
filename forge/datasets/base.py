from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable


@dataclass
class Example:
    """One unit of supervised reasoning. The model sees `context`; `target`
    is the ground truth (a diff, post-commit code, or expected answer)."""

    id: str
    context: str
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Source(Protocol):
    """Anything that can stream Examples. Implemented by GitCommitSource,
    JsonlSource, and (eventually) GitHubApiSource."""

    def iter_examples(self, **filters: Any) -> Iterator[Example]: ...
