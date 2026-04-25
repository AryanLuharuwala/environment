from __future__ import annotations

import math
from dataclasses import dataclass, field

from .base import Document, Retriever


@dataclass
class EmbeddingRetriever(Retriever):
    """Optional embedding-based retriever. Ships with sentence-transformers
    when installed via `pip install -e '.[forge-embed]'`. Imports are lazy
    so the rest of forge works without the heavy deps."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    documents: list[Document] = field(default_factory=list)
    _vectors: list[list[float]] = field(default_factory=list, repr=False)
    _norms: list[float] = field(default_factory=list, repr=False)
    _model: object | None = field(default=None, repr=False)

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "EmbeddingRetriever needs sentence-transformers. "
                "Install with: pip install 'ale[forge-embed]' (or sentence-transformers directly)."
            ) from e
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def index(self, documents: list[Document]) -> "EmbeddingRetriever":
        self.documents = list(documents)
        if not self.documents:
            self._vectors = []
            self._norms = []
            return self
        model = self._load_model()
        embs = model.encode([d.text for d in self.documents], show_progress_bar=False)
        self._vectors = [list(map(float, v)) for v in embs]
        self._norms = [_norm(v) for v in self._vectors]
        return self

    def search(self, query: str, k: int = 3) -> list[Document]:
        if not self.documents:
            return []
        model = self._load_model()
        q = list(map(float, model.encode([query], show_progress_bar=False)[0]))
        qn = _norm(q) or 1.0
        scored: list[tuple[float, int]] = []
        for i, v in enumerate(self._vectors):
            dn = self._norms[i] or 1.0
            scored.append((_dot(q, v) / (qn * dn), i))
        scored.sort(reverse=True)
        return [self.documents[i] for _, i in scored[:k]]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))
