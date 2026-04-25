from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ale.providers import Provider, make_provider

from ..datasets import Example
from ..retrieval import Retriever
from ..verifiers import (
    DiffSimilarityVerifier,
    PythonAstVerifier,
    PytestVerifier,
    Verifier,
    composite,
)
from .base import RolloutResult, extract_python_code


CODER_SYSTEM = """You are a senior Python engineer reproducing the next change to a file.

You will see:
- The current contents of a file.
- The commit message describing the intended change.
- (Optional) similar code from a corpus of high-quality projects.

Return the COMPLETE post-change file. Wrap your output in a single ```python fenced block. No explanation outside the block.
"""


@dataclass
class CoderConfig:
    label: str = "coder"
    use_rag: bool = True
    k: int = 3
    provider: Provider | None = None
    extra_system: str = ""

    def __post_init__(self) -> None:
        if self.provider is None:
            self.provider = make_provider("anthropic")


@dataclass
class CoderRollout:
    """Pipeline for code prediction:

    example -> [retrieval] -> prompt -> model -> verifiers -> reward.

    AST failures are a hard penalty (composite reward goes to 0) — this is
    the AST-aware reward system requested in the spec.
    """

    config: CoderConfig
    retriever: Retriever | None = None
    verifiers: list[Verifier] = field(default_factory=list)
    weights: dict[str, float] = field(
        default_factory=lambda: {"ast": 0.4, "diff_similarity": 0.4, "tests": 0.2}
    )
    hard_fail: set[str] = field(default_factory=lambda: {"ast"})

    @classmethod
    def default(cls, config: CoderConfig, retriever: Retriever | None = None,
                with_tests: bool = False) -> "CoderRollout":
        verifiers: list[Verifier] = [PythonAstVerifier(), DiffSimilarityVerifier()]
        if with_tests:
            verifiers.append(PytestVerifier())
        return cls(config=config, retriever=retriever, verifiers=verifiers)

    def run(self, example: Example) -> RolloutResult:
        retrieved = []
        if self.config.use_rag and self.retriever is not None:
            retrieved = self.retriever.search(example.context, k=self.config.k)

        retrieval_block = ""
        if retrieved:
            retrieval_block = "\n\nSimilar code from the corpus:\n"
            for d in retrieved:
                retrieval_block += f"\n--- {d.id} ---\n{d.text}\n"

        user = (
            f"{example.context}\n"
            f"{retrieval_block}\n"
            f"Now produce the post-change version of the file."
        )
        system = CODER_SYSTEM + ("\n" + self.config.extra_system if self.config.extra_system else "")

        response = self.config.provider.complete(
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=None,
            max_tokens=4096,
        )
        prediction = extract_python_code(response.text).strip()

        scores = [v.check(prediction, example) for v in self.verifiers]
        reward = composite(scores, weights=self.weights, hard_fail=self.hard_fail)

        return RolloutResult(
            example_id=example.id,
            config_label=self.config.label,
            prediction=prediction,
            scores=scores,
            reward=reward,
            retrieved_ids=[d.id for d in retrieved],
            metadata={
                "model": self.config.provider.model,
                "use_rag": self.config.use_rag,
                "raw_response_text": response.text,
            },
        )
