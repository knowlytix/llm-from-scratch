"""Bag-of-words retrieval stub for the RAG demonstration in Chapter 24."""

from __future__ import annotations

import math
from collections import Counter


class RetrievalStub:
    """Tiny BoW retrieval: tokenize on whitespace, cosine similarity."""

    def __init__(self, corpus: list[str]) -> None:
        self.docs = corpus
        self._vecs = [Counter(d.lower().split()) for d in corpus]

    def _cos(self, a: Counter, b: Counter) -> float:
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        da = math.sqrt(sum(v * v for v in a.values()))
        db = math.sqrt(sum(v * v for v in b.values()))
        if da == 0 or db == 0:
            return 0.0
        return num / (da * db)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[int, float]]:
        q = Counter(query.lower().split())
        scored = [(i, self._cos(q, v)) for i, v in enumerate(self._vecs)]
        scored.sort(key=lambda t: -t[1])
        return scored[:top_k]
