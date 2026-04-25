from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .base import Document, Retriever


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|[+\-*/=<>!]+|[.,;:(){}\[\]]")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _tokenize(text: str) -> list[str]:
    """Code-aware tokenizer: identifiers split into their snake_case and
    camelCase components, then lowercased. So `hash_password` and
    `hashPassword` both produce ['hash', 'password'] — query 'hash password'
    matches either form. Punctuation passes through as its own tokens."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        if not raw or not raw[0].isalpha() and raw[0] != "_":
            out.append(raw.lower())
            continue
        for piece in raw.split("_"):
            for sub in _CAMEL_RE.split(piece):
                if sub:
                    out.append(sub.lower())
    return out


@dataclass
class TfidfRetriever(Retriever):
    """Tiny stdlib TF-IDF + cosine similarity. Good enough for retrieving
    'similar code' from a corpus of a few thousand snippets. For larger
    corpora or semantic similarity, install `forge[embed]` and use
    `embed.EmbeddingRetriever`."""

    documents: list[Document] = field(default_factory=list)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _vectors: list[dict[str, float]] = field(default_factory=list, repr=False)
    _norms: list[float] = field(default_factory=list, repr=False)

    def index(self, documents: list[Document]) -> "TfidfRetriever":
        self.documents = list(documents)
        n_docs = len(self.documents) or 1
        df: Counter[str] = Counter()
        token_lists: list[list[str]] = []
        for d in self.documents:
            toks = _tokenize(d.text)
            token_lists.append(toks)
            for term in set(toks):
                df[term] += 1
        # Smoothed IDF: log((1+N) / (1+df)) + 1
        self._idf = {t: math.log((1 + n_docs) / (1 + c)) + 1.0 for t, c in df.items()}
        self._vectors = [self._vectorize(toks) for toks in token_lists]
        self._norms = [_norm(v) for v in self._vectors]
        return self

    def search(self, query: str, k: int = 3) -> list[Document]:
        if not self.documents:
            return []
        qv = self._vectorize(_tokenize(query))
        qn = _norm(qv) or 1.0
        scores: list[tuple[float, int]] = []
        for i, dv in enumerate(self._vectors):
            dn = self._norms[i] or 1.0
            scores.append((_dot(qv, dv) / (qn * dn), i))
        scores.sort(reverse=True)
        return [self.documents[i] for _, i in scores[:k] if _ > 0]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        tf: Counter[str] = Counter(tokens)
        max_tf = max(tf.values()) or 1
        return {t: (0.5 + 0.5 * c / max_tf) * self._idf.get(t, 0.0) for t, c in tf.items()}

    # --- persistence -----------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [{"id": d.id, "text": d.text, "metadata": d.metadata} for d in self.documents],
            "idf": self._idf,
            "vectors": self._vectors,
            "norms": self._norms,
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "TfidfRetriever":
        path = Path(path)
        payload = json.loads(path.read_text())
        r = cls()
        r.documents = [Document(**d) for d in payload["documents"]]
        r._idf = {k: float(v) for k, v in payload["idf"].items()}
        r._vectors = [{k: float(v) for k, v in vec.items()} for vec in payload["vectors"]]
        r._norms = [float(n) for n in payload["norms"]]
        return r


def _dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _norm(v: dict[str, float]) -> float:
    return math.sqrt(sum(x * x for x in v.values()))
