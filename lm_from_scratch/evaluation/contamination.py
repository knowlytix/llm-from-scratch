"""Contamination check: re-uses Chapter 2's detector."""

from __future__ import annotations

from lm_from_scratch.text.corpus import TextCorpus
from lm_from_scratch.text.splitting import detect_contamination as _detect


def contamination_check(prompts: list[str], training_text: str, ngram: int = 30) -> list[tuple[int, int, str]]:
    """Find verbatim spans of length ``ngram`` from any prompt in the training text."""
    train_corpus = TextCorpus([training_text])
    eval_corpus = TextCorpus(prompts)
    return _detect(train_corpus, eval_corpus, ngram=ngram)
