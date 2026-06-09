"""Deduplication of documents.

Two flavors:

* ``exact_dedup`` removes byte-equal duplicates by hash.
* ``near_dedup`` removes near-duplicates by MinHash + Jaccard similarity over
  character n-grams. The implementation is a reference one, not a fast
  production implementation.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable


def _hash_doc(doc: str) -> str:
    return hashlib.sha256(doc.encode("utf-8")).hexdigest()


def exact_dedup(docs: Iterable[str]) -> list[str]:
    """Return documents in input order, dropping byte-equal duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for d in docs:
        h = _hash_doc(d)
        if h in seen:
            continue
        seen.add(h)
        out.append(d)
    return out


def _ngram_set(text: str, n: int) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union


class _MinHash:
    """A small MinHash implementation suitable for teaching."""

    def __init__(self, num_perm: int = 64, seed: int = 0):
        self.num_perm = num_perm
        rng = random.Random(seed)
        # 64-bit hash parameters; doc-level MinHash uses a + b style universal hashing.
        self.a = [rng.randrange(1, 2**61 - 1) for _ in range(num_perm)]
        self.b = [rng.randrange(0, 2**61 - 1) for _ in range(num_perm)]
        self._mersenne = 2**61 - 1

    def signature(self, items: set[str]) -> list[int]:
        if not items:
            return [self._mersenne] * self.num_perm
        hashed = [int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:15], 16) for s in items]
        sig = []
        for a, b in zip(self.a, self.b):
            min_h = self._mersenne
            for h in hashed:
                v = (a * h + b) % self._mersenne
                if v < min_h:
                    min_h = v
            sig.append(min_h)
        return sig

    @staticmethod
    def estimated_jaccard(sig1: list[int], sig2: list[int]) -> float:
        n = len(sig1)
        eq = sum(1 for x, y in zip(sig1, sig2) if x == y)
        return eq / n if n else 0.0


def near_dedup(
    docs: Iterable[str],
    ngram: int = 13,
    threshold: float = 0.8,
    num_perm: int = 64,
    seed: int = 0,
) -> list[str]:
    """Remove documents whose Jaccard similarity (over character n-grams) with
    an already-kept document exceeds ``threshold``.

    We use MinHash signatures to estimate the similarity; this is the
    teaching version, not the production one. For corpora larger than a few
    thousand documents use a real LSH index.
    """
    mh = _MinHash(num_perm=num_perm, seed=seed)
    kept: list[str] = []
    kept_sigs: list[list[int]] = []
    for d in docs:
        sig = mh.signature(_ngram_set(d, ngram))
        is_dup = False
        for ks in kept_sigs:
            if mh.estimated_jaccard(sig, ks) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(d)
            kept_sigs.append(sig)
    return kept


def true_jaccard(a: str, b: str, ngram: int = 13) -> float:
    """True Jaccard similarity over character n-grams.

    Slow; useful for verifying the MinHash estimate in tests and notebooks.
    """
    return _jaccard(_ngram_set(a, ngram), _ngram_set(b, ngram))
