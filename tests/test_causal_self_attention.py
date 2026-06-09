"""Tests for Chapter 9 causal self-attention."""

import torch

from llm_from_scratch.models.attention import (
    SingleHeadCausalSelfAttention,
    TinyCausalAttnLM,
    causal_mask,
)


def test_causal_mask_blocks_future() -> None:
    m = causal_mask(4)
    # Position 0 cannot attend to positions 1, 2 or 3.
    assert m[0, 0].item() == 0.0
    assert m[0, 1].item() == float("-inf")
    assert m[0, 3].item() == float("-inf")
    # Position 2 attends to 0, 1 and 2 but not 3.
    assert m[2, 1].item() == 0.0
    assert m[2, 3].item() == float("-inf")


def test_no_information_leakage_through_attention() -> None:
    # If we change a position's value, only positions at or after it
    # should change in the output.
    torch.manual_seed(0)
    m = SingleHeadCausalSelfAttention(embedding_dim=8, head_dim=4, block_size=6)
    x1 = torch.randn(1, 6, 8)
    x2 = x1.clone()
    x2[0, 3] += 5.0  # perturb position 3 only

    out1 = m(x1)
    out2 = m(x2)
    diffs = (out2 - out1).abs().sum(dim=-1).squeeze()
    # Positions 0..2 should be unchanged; positions 3..5 will differ.
    assert diffs[0].item() < 1e-4
    assert diffs[1].item() < 1e-4
    assert diffs[2].item() < 1e-4
    assert diffs[3].item() > 0.1


def test_output_shape() -> None:
    m = SingleHeadCausalSelfAttention(embedding_dim=16, head_dim=8, block_size=20)
    x = torch.randn(2, 10, 16)
    out = m(x)
    assert out.shape == (2, 10, 16)


def test_tiny_causal_attn_lm_forward_and_loss() -> None:
    m = TinyCausalAttnLM(vocab_size=20, embedding_dim=16, head_dim=8, block_size=8)
    x = torch.randint(0, 20, (2, 8))
    y = torch.randint(0, 20, (2, 8))
    logits = m(x)
    assert logits.shape == (2, 8, 20)
    loss = m.loss(x, y)
    assert loss.dim() == 0
    loss.backward()
