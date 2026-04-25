from __future__ import annotations

import re
from dataclasses import dataclass, field

from ale.providers import Provider, make_provider

from ..datasets import Example
from .base import Score, Verifier


GROUNDING_SYSTEM = """You are a grounding judge. Given an answer to a question and a set of source documents, score how well the answer is SUPPORTED by the sources.

Penalize:
- Claims that contradict the sources.
- Claims that go beyond what the sources say (hallucinations).
- Citations that don't appear in the sources.

Reward:
- Answers that stay within what the sources support.
- Clear references back to specific sources.

Respond in EXACTLY this format:
SUPPORT: <0.0-1.0>
ISSUES: <one short sentence; "none" if fully supported>
"""


@dataclass
class GroundingVerifier(Verifier):
    """LLM-as-judge grounding score. Used for law/compliance and any other
    'reason then verify against authoritative sources' domain.

    The judge runs against an `ale.providers.Provider` — Anthropic by
    default, so the system prompt is sent with `cache_control` and stays
    cheap across rollouts. Pass any other provider for offline / hosted
    judges.

    Expected `Example.metadata`:
        - `sources`: list[str] — authoritative source texts (typically the
          K results returned by the retriever)
        - `question` (optional) — the original question; falls back to
          `example.context` when missing.
    """

    name: str = "grounding"
    weight: float = 0.7
    provider: Provider | None = None

    def __post_init__(self) -> None:
        if self.provider is None:
            self.provider = make_provider("anthropic")

    def check(self, prediction: str, example: Example) -> Score:
        sources = example.metadata.get("sources") or []
        question = example.metadata.get("question") or example.context
        if not sources:
            return Score(name=self.name, value=0.0, detail="skipped: no sources in metadata")

        joined = "\n\n".join(f"[{i + 1}]\n{s}" for i, s in enumerate(sources))
        user = (
            f"Question:\n{question}\n\n"
            f"Sources:\n{joined}\n\n"
            f"Answer to grade:\n{prediction}\n\n"
            f"Score and explain."
        )
        response = self.provider.complete(
            system=GROUNDING_SYSTEM,
            messages=[{"role": "user", "content": user}],
            tools=None,
            max_tokens=256,
        )
        value, detail = _parse(response.text)
        return Score(name=self.name, value=value, detail=detail)


_SUPPORT_RE = re.compile(r"SUPPORT:\s*([0-9.]+)", re.IGNORECASE)
_ISSUES_RE = re.compile(r"ISSUES:\s*(.+)", re.IGNORECASE)


def _parse(text: str) -> tuple[float, str]:
    m = _SUPPORT_RE.search(text)
    value = 0.0
    if m:
        try:
            value = max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            value = 0.0
    issues = _ISSUES_RE.search(text)
    detail = issues.group(1).strip() if issues else text.strip().splitlines()[0][:120]
    return value, detail
