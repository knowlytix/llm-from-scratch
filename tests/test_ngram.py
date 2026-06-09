"""Tests for the n-gram language model (Chapter 5)."""

import math
import random

import pytest

from llm_from_scratch.models.ngram import NGramLanguageModel


def test_unigram_probs_sum_to_one_with_smoothing() -> None:
    m = NGramLanguageModel(n=1, smoothing="add_k", k=0.1)
    m.fit([0, 1, 2, 0, 1])
    total = sum(m.prob((), i) for i in range(m.vocab_size))
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_bigram_simple() -> None:
    # On [0,1,0,1,0,1] context 0 always precedes 1; context 1 always precedes 0.
    m = NGramLanguageModel(n=2, smoothing="none")
    m.fit([0, 1, 0, 1, 0, 1])
    assert m.prob((0,), 1) == 1.0
    assert m.prob((1,), 0) == 1.0


def test_perplexity_matches_exp_neg_logprob_over_n() -> None:
    m = NGramLanguageModel(n=2, smoothing="add_k", k=0.1)
    m.fit([0, 1, 2, 0, 1, 2])
    seq = [0, 1, 2]
    lp = m.log_prob_sequence(seq)
    n = len(seq)
    assert m.perplexity(seq) == pytest.approx(math.exp(-lp / n), rel=1e-9)


def test_perplexity_training_lower_than_random() -> None:
    m = NGramLanguageModel(n=2, smoothing="add_k", k=0.1)
    train = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    m.fit(train)
    ppl_train = m.perplexity(train)
    ppl_random = m.perplexity([2, 0, 0, 1, 2, 1, 0, 0, 2])
    assert ppl_train < ppl_random


def test_sample_is_deterministic_with_seeded_rng() -> None:
    m = NGramLanguageModel(n=2, smoothing="add_k")
    m.fit([0, 1, 2, 0, 1, 2, 0, 1, 2])
    a = m.sample([0], max_new_tokens=10, rng=random.Random(123))
    b = m.sample([0], max_new_tokens=10, rng=random.Random(123))
    assert a == b


def test_zero_prob_without_smoothing_raises_in_perplexity() -> None:
    m = NGramLanguageModel(n=2, smoothing="none")
    m.fit([0, 1, 0, 1])
    with pytest.raises(ValueError):
        # Token 2 never appears as a target after any context; without
        # smoothing the perplexity is undefined.
        m.perplexity([0, 1, 2])


def test_n_validation() -> None:
    with pytest.raises(ValueError):
        NGramLanguageModel(n=0)


def test_smoothing_validation() -> None:
    with pytest.raises(ValueError):
        NGramLanguageModel(n=2, smoothing="bogus")
