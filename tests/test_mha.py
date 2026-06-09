"""Tests for Chapter 10 multi-head causal self-attention."""

import torch

from llm_from_scratch.models.attention import (
    MultiHeadCausalSelfAttention,
    TinyMHACausalLM,
)


def test_mha_output_shape() -> None:
    m = MultiHeadCausalSelfAttention(embedding_dim=32, num_heads=4, block_size=16)
    x = torch.randn(2, 10, 32)
    out = m(x)
    assert out.shape == (2, 10, 32)


def test_mha_returns_per_head_weights() -> None:
    m = MultiHeadCausalSelfAttention(embedding_dim=32, num_heads=4, block_size=16)
    x = torch.randn(2, 10, 32)
    _, w = m(x, return_weights=True)
    assert w.shape == (2, 4, 10, 10)


def test_mha_rejects_indivisible_dim() -> None:
    import pytest

    with pytest.raises(ValueError):
        MultiHeadCausalSelfAttention(embedding_dim=30, num_heads=4, block_size=16)


def test_mha_rejects_too_long_sequence() -> None:
    import pytest

    m = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, block_size=4)
    x = torch.randn(1, 5, 8)
    with pytest.raises(ValueError):
        m(x)


def test_per_head_attention_weights_are_causal() -> None:
    m = MultiHeadCausalSelfAttention(embedding_dim=16, num_heads=4, block_size=8)
    x = torch.randn(1, 8, 16)
    _, w = m(x, return_weights=True)
    upper = torch.triu(w, diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)


def test_per_head_rows_sum_to_one() -> None:
    m = MultiHeadCausalSelfAttention(embedding_dim=16, num_heads=4, block_size=8)
    x = torch.randn(2, 8, 16)
    _, w = m(x, return_weights=True)
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_mha_no_future_leakage() -> None:
    # Perturb a single position and confirm earlier positions are unchanged.
    torch.manual_seed(0)
    m = MultiHeadCausalSelfAttention(embedding_dim=16, num_heads=4, block_size=8)
    x1 = torch.randn(1, 8, 16)
    x2 = x1.clone()
    x2[0, 4] += 10.0
    diff = (m(x2) - m(x1)).abs().sum(dim=-1).squeeze()
    assert diff[0].item() < 1e-4
    assert diff[3].item() < 1e-4
    assert diff[4].item() > 0.1


def test_mha_gradients_flow() -> None:
    m = MultiHeadCausalSelfAttention(embedding_dim=16, num_heads=4, block_size=8)
    x = torch.randn(2, 8, 16, requires_grad=True)
    out = m(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


def test_tiny_mha_lm_forward_and_loss() -> None:
    m = TinyMHACausalLM(vocab_size=20, embedding_dim=16, num_heads=4, block_size=8)
    x = torch.randint(0, 20, (2, 8))
    y = torch.randint(0, 20, (2, 8))
    logits = m(x)
    assert logits.shape == (2, 8, 20)
    loss = m.loss(x, y)
    assert loss.dim() == 0
    loss.backward()
