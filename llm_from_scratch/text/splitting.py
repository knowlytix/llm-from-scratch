"""Document-level train/validation/test splits and contamination detection."""

from __future__ import annotations

import random

from llm_from_scratch.text.corpus import TextCorpus


def split_corpus(
    corpus: TextCorpus,
    train: float = 0.9,
    valid: float = 0.05,
    test: float = 0.05,
    seed: int = 42,
) -> tuple[TextCorpus, TextCorpus, TextCorpus]:
    """Split a corpus at the document level into train, validation and test.

    Documents are shuffled with the given seed and partitioned according to
    the fractions, which must sum to (approximately) one.
    """
    fractions = (train, valid, test)
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {fractions}")

    docs = list(corpus.documents)
    rng = random.Random(seed)
    rng.shuffle(docs)

    n = len(docs)
    n_train = int(round(n * train))
    n_valid = int(round(n * valid))
    train_docs = docs[:n_train]
    valid_docs = docs[n_train : n_train + n_valid]
    test_docs = docs[n_train + n_valid :]

    base_meta = dict(corpus.metadata)
    return (
        TextCorpus(train_docs, metadata={**base_meta, "split": "train", "seed": seed}),
        TextCorpus(valid_docs, metadata={**base_meta, "split": "valid", "seed": seed}),
        TextCorpus(test_docs, metadata={**base_meta, "split": "test", "seed": seed}),
    )


def detect_contamination(
    train: TextCorpus,
    eval_corpus: TextCorpus,
    ngram: int = 50,
) -> list[tuple[int, int, str]]:
    """Find spans of length ``ngram`` chars from ``eval_corpus`` that appear
    verbatim in ``train``.

    Returns a list of ``(eval_doc_idx, start_in_eval_doc, span)`` triples.
    The detection is a simple substring scan; suitable for teaching but not
    for very large train corpora.
    """
    train_text = "\n".join(train.documents)
    hits: list[tuple[int, int, str]] = []
    for i, eval_doc in enumerate(eval_corpus.documents):
        if len(eval_doc) < ngram:
            continue
        for j in range(len(eval_doc) - ngram + 1):
            span = eval_doc[j : j + ngram]
            if span in train_text:
                hits.append((i, j, span))
    return hits
