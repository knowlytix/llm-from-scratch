"""Diagnostic summaries of a corpus."""

from __future__ import annotations

from collections import Counter

from llm_from_scratch.text.corpus import TextCorpus
from llm_from_scratch.text.dedup import _hash_doc


def corpus_summary(corpus: TextCorpus) -> dict[str, float | int | str]:
    """Return a dict of common corpus diagnostics.

    Includes document count, total characters, average length, distinct
    character vocabulary and an estimated duplicate rate.
    """
    docs = corpus.documents
    lengths = [len(d) for d in docs]
    chars = Counter()
    for d in docs:
        chars.update(d)
    hashes = [_hash_doc(d) for d in docs]
    duplicate_rate = 1 - (len(set(hashes)) / max(1, len(hashes)))

    return {
        "num_documents": len(docs),
        "total_chars": sum(lengths),
        "avg_doc_chars": float(sum(lengths) / max(1, len(lengths))),
        "min_doc_chars": int(min(lengths)) if lengths else 0,
        "max_doc_chars": int(max(lengths)) if lengths else 0,
        "distinct_chars": len(chars),
        "duplicate_rate": float(duplicate_rate),
    }
