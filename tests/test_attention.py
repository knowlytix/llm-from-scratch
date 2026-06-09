"""Tests for the Chapter 8 attention primitives."""

import math

import torch

from llm_from_scratch.models.attention import (
    SingleHeadAttention,
    scaled_dot_product_attention,
)
from llm_from_scratch.utils.env import set_seed


def test_output_shape() -> None:
    q = torch.randn(2, 5, 4)
    k = torch.randn(2, 7, 4)
    v = torch.randn(2, 7, 6)
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 5, 6)


def test_weights_non_negative_and_normalized() -> None:
    set_seed(0)
    q = torch.randn(1, 3, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    _, w = scaled_dot_product_attention(q, k, v, return_weights=True)
    assert (w >= 0).all()
    assert torch.allclose(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-5)


def test_recovers_closest_value() -> None:
    # One key is strongly aligned with q; the others are orthogonal.
    # Attention should put almost all weight on that key.
    q = torch.tensor([[[10.0, 0.0]]])  # (1, 1, 2)
    k = torch.tensor([[[10.0, 0.0], [0.0, 10.0], [0.0, -10.0]]])  # (1, 3, 2)
    v = torch.tensor([[[10.0], [20.0], [30.0]]])  # (1, 3, 1)
    out = scaled_dot_product_attention(q, k, v)
    assert abs(out.item() - 10.0) < 0.1


def test_scaling_prevents_softmax_saturation() -> None:
    # With high d_k and no scaling, dot products are large in absolute
    # value and softmax saturates; the scaling controls the variance.
    set_seed(0)
    d = 128
    q = torch.randn(1, 1, d)
    k = torch.randn(1, 4, d)
    v = torch.randn(1, 4, 8)

    # With scaling, weights have nontrivial entropy.
    _, w = scaled_dot_product_attention(q, k, v, return_weights=True)
    entropy_scaled = -(w * (w + 1e-12).log()).sum().item()

    # Without scaling, weights are more peaked.
    scores = q @ k.transpose(-2, -1)
    w_unscaled = torch.softmax(scores, dim=-1)
    entropy_unscaled = -(w_unscaled * (w_unscaled + 1e-12).log()).sum().item()

    assert entropy_scaled > entropy_unscaled


def test_mask_blocks_positions() -> None:
    q = torch.randn(1, 2, 4)
    k = torch.randn(1, 3, 4)
    v = torch.randn(1, 3, 4)
    mask = torch.tensor([[[0.0, float("-inf"), 0.0], [0.0, 0.0, float("-inf")]]])
    _, w = scaled_dot_product_attention(q, k, v, mask=mask, return_weights=True)
    assert w[0, 0, 1].item() == 0.0
    assert w[0, 1, 2].item() == 0.0


def test_single_head_attention_forward_and_grads() -> None:
    m = SingleHeadAttention(embedding_dim=16, head_dim=8)
    x = torch.randn(2, 5, 16, requires_grad=True)
    out = m(x)
    assert out.shape == (2, 5, 8)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and x.grad.shape == x.shape
