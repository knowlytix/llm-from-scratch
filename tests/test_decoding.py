"""Tests for Chapter 19 decoding."""

import torch

from lm_from_scratch.generation.decoding import beam_search, greedy_decode, sample_decode
from lm_from_scratch.generation.sampling import top_k_filter, top_p_filter
from lm_from_scratch.models.configs import GPTConfig
from lm_from_scratch.models.gpt import TinyGPT


def _model():
    torch.manual_seed(0)
    return TinyGPT(GPTConfig(vocab_size=20, block_size=16, embedding_dim=16,
                             num_layers=2, num_heads=4, dropout=0.0))


def test_greedy_deterministic() -> None:
    m = _model()
    a = greedy_decode(m, [1, 2, 3], max_new_tokens=5)
    b = greedy_decode(m, [1, 2, 3], max_new_tokens=5)
    assert a == b


def test_temperature_zero_equivalent_to_greedy() -> None:
    m = _model()
    g = greedy_decode(m, [1, 2, 3], max_new_tokens=5)
    s = sample_decode(m, [1, 2, 3], max_new_tokens=5, temperature=1e-9)
    assert g == s


def test_top_k_filter_keeps_only_k() -> None:
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    out = top_k_filter(logits, k=2)
    finite = (out > float("-inf")).sum().item()
    assert finite == 2


def test_top_p_filter_keeps_minimal_set() -> None:
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0, 10.0])  # one dominant token
    out = top_p_filter(logits, p=0.9)
    finite = (out > float("-inf")).sum().item()
    assert finite == 1


def test_beam_search_returns_num_beams_candidates() -> None:
    m = _model()
    beams = beam_search(m, [1, 2, 3], max_new_tokens=4, num_beams=3)
    assert len(beams) == 3
