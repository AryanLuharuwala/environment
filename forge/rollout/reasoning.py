from __future__ import annotations

import re
from dataclasses import dataclass, field

from ale.providers import Provider, make_provider

from ..datasets import Example
from ..retrieval import Retriever
from ..verifiers import GroundingVerifier, Score, Verifier, composite
from .base import RolloutResult


REASONING_SYSTEM = """You are a careful expert (law / compliance / domain analyst). You will be given:
- A question.
- Authoritative source materials retrieved from a corpus.

Answer the question using ONLY information that the sources support. Cite each non-trivial claim with a bracketed source number like [1], [2]. If the sources don't answer the question, say so explicitly.

Format:
ANSWER:
<your answer with [n] citations>

CITATIONS_USED: <comma-separated source numbers>
"""


_CITATION_RE = re.compile(r"\[(\d+)\]")


class CitationFormatVerifier(Verifier):
    """Cheap structural check: is the answer non-trivial AND does it cite at
    least one source? Catches answers that hallucinate confidently without
    grounding markers, before we even pay for the LLM judge."""

    name = "citation_format"
    weight = 0.3

    def check(self, prediction: str, example: Example) -> Score:
        if len(prediction.strip()) < 20:
            return Score(name=self.name, value=0.0, detail="answer too short")
        sources = example.metadata.get("sources") or []
        cites = {int(m) for m in _CITATION_RE.findall(prediction)}
        if not cites:
            return Score(name=self.name, value=0.0, detail="no [n] citations")
        valid = {n for n in cites if 1 <= n <= len(sources)}
        ratio = len(valid) / len(cites) if cites else 0.0
        return Score(
            name=self.name,
            value=ratio,
            detail=f"{len(valid)}/{len(cites)} citations valid",
        )


@dataclass
class ReasoningConfig:
    label: str = "grounded"
    use_rag: bool = True
    k: int = 4
    provider: Provider | None = None  # the answering agent

    def __post_init__(self) -> None:
        if self.provider is None:
            self.provider = make_provider("anthropic")


@dataclass
class ReasoningRollout:
    """For domains where 'correctness' is grounding-against-sources rather
    than executable: law, compliance, policy, regulated reasoning.

    Two agents:
    - The answering agent (`config.provider`) — can be any backend.
    - The grounding judge inside `GroundingVerifier` — defaults to Claude
      with cache_control on the system prompt.

    Reward = weighted mean of citation-format check + grounding score.
    """

    config: ReasoningConfig
    retriever: Retriever | None = None
    grounding: GroundingVerifier = field(default_factory=GroundingVerifier)
    citation: CitationFormatVerifier = field(default_factory=CitationFormatVerifier)
    weights: dict[str, float] = field(
        default_factory=lambda: {"grounding": 0.7, "citation_format": 0.3}
    )

    def run(self, example: Example) -> RolloutResult:
        question = example.metadata.get("question") or example.context
        sources: list[str] = list(example.metadata.get("sources") or [])
        retrieved = []

        if self.config.use_rag and self.retriever is not None:
            retrieved = self.retriever.search(question, k=self.config.k)
            sources = [d.text for d in retrieved] + sources

        joined = "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(sources))
        user = f"Question:\n{question}\n\nSources:\n{joined}\n\nWrite your answer."
        response = self.config.provider.complete(
            system=REASONING_SYSTEM,
            messages=[{"role": "user", "content": user}],
            tools=None,
            max_tokens=1024,
        )
        prediction = response.text.strip()

        # Inject the live `sources` into the example so the grounding judge
        # sees what the answering agent saw — without mutating the caller's
        # example.
        scoring_example = Example(
            id=example.id,
            context=example.context,
            target=example.target,
            metadata={**example.metadata, "sources": sources, "question": question},
        )
        scores = [
            self.citation.check(prediction, scoring_example),
            self.grounding.check(prediction, scoring_example),
        ]
        reward = composite(scores, weights=self.weights)

        return RolloutResult(
            example_id=example.id,
            config_label=self.config.label,
            prediction=prediction,
            scores=scores,
            reward=reward,
            retrieved_ids=[d.id for d in retrieved],
            metadata={"model": self.config.provider.model, "use_rag": self.config.use_rag},
        )
