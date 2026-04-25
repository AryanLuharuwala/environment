"""forge — reasoning-from-history training data factory.

A sibling package to `ale`. Mines real artifacts (git commits, legal docs,
compliance corpora, ...), runs a reason-then-verify rollout, and exports
DPO training pairs in the same JSONL schema that `ale rlaif train`
consumes.

Independent of `ale` apart from the provider abstraction
(`ale.providers`).
"""
from .datasets import Example, GitCommitSource, JsonlSource, Source
from .export import pair_to_dpo, write_dpo_jsonl
from .retrieval import Document, Retriever, TfidfRetriever, from_directory, from_jsonl
from .rollout import (
    CoderConfig,
    CoderRollout,
    ReasoningConfig,
    ReasoningRollout,
    RolloutResult,
)
from .verifiers import (
    DiffSimilarityVerifier,
    GroundingVerifier,
    PythonAstVerifier,
    PytestVerifier,
    Score,
    Verifier,
    composite,
)

__version__ = "0.1.0"

__all__ = [
    "CoderConfig",
    "CoderRollout",
    "DiffSimilarityVerifier",
    "Document",
    "Example",
    "GitCommitSource",
    "GroundingVerifier",
    "JsonlSource",
    "PythonAstVerifier",
    "PytestVerifier",
    "ReasoningConfig",
    "ReasoningRollout",
    "Retriever",
    "RolloutResult",
    "Score",
    "Source",
    "TfidfRetriever",
    "Verifier",
    "composite",
    "from_directory",
    "from_jsonl",
    "pair_to_dpo",
    "write_dpo_jsonl",
]
