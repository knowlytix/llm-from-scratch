"""Tests for Chapter 14 perplexity and diagnostics."""

import math

import torch

from lm_from_scratch.datasets.sequence_dataset import CausalLMDataset
from lm_from_scratch.evaluation.diagnostics import per_position_loss, token_accuracy
from lm_from_scratch.evaluation.perplexity import evaluate_perplexity


class _UniformLM(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, target_ids=None):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size, device=input_ids.device)
        if target_ids is None:
            return logits, None
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(B * T, self.vocab_size), target_ids.reshape(B * T)
        )
        return logits, loss

    def loss(self, input_ids, target_ids):
        return self.forward(input_ids, target_ids)[1]


def test_perplexity_of_uniform_is_V() -> None:
    V = 17
    m = _UniformLM(V)
    ids = [i % V for i in range(50)]
    ds = CausalLMDataset(ids, block_size=8, stride=8)
    out = evaluate_perplexity(m, ds, batch_size=4, device="cpu")
    assert math.isclose(out["perplexity"], V, abs_tol=1e-3)


def test_bits_per_byte_conversion() -> None:
    V = 17
    m = _UniformLM(V)
    ids = [i % V for i in range(50)]
    ds = CausalLMDataset(ids, block_size=8, stride=8)
    out = evaluate_perplexity(m, ds, batch_size=4, device="cpu", bytes_per_token=2.0)
    expected_bpb = math.log(V) / (math.log(2) * 2.0)
    assert math.isclose(out["bits_per_byte"], expected_bpb, abs_tol=1e-6)


def test_token_accuracy_uniform_is_one_over_V() -> None:
    # Uniform logits -> argmax is always 0, so accuracy ≈ fraction of
    # tokens equal to 0.
    torch.manual_seed(0)
    V = 5
    m = _UniformLM(V)
    ids = [0] * 40  # all zeros, argmax also 0 -> accuracy 1.0
    ds = CausalLMDataset(ids, block_size=8, stride=8)
    assert token_accuracy(m, ds, batch_size=4, device="cpu") == 1.0


def test_per_position_loss_shape() -> None:
    V = 7
    m = _UniformLM(V)
    ds = CausalLMDataset([i % V for i in range(60)], block_size=10, stride=10)
    pp = per_position_loss(m, ds, batch_size=4, device="cpu")
    assert pp.shape == (10,)
    # Uniform model: every position has loss ln(V)
    assert torch.allclose(pp, torch.full_like(pp, math.log(V)), atol=1e-5)
