"""Tests for Chapter 11 positional encodings."""

import math

import pytest
import torch

from lm_from_scratch.models.attention import scaled_dot_product_attention
from lm_from_scratch.models.positional import (
    ALiBi,
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    make_positional,
)


# --- Permutation equivariance without position ------------------------


def test_attention_alone_is_permutation_equivariant() -> None:
    # No positional info: permuting the input permutes the output the same way.
    torch.manual_seed(0)
    x = torch.randn(1, 5, 8)
    q_w = torch.randn(8, 8)
    perm = torch.tensor([3, 0, 4, 1, 2])

    def run(xs):
        q = xs @ q_w
        k = xs @ q_w
        v = xs
        return scaled_dot_product_attention(q, k, v)

    out_a = run(x)
    out_b = run(x[:, perm])
    assert torch.allclose(out_a[:, perm], out_b, atol=1e-5)


# --- Learned positional ------------------------------------------------


def test_learned_positional_adds_position_signal() -> None:
    pe = LearnedPositionalEmbedding(block_size=10, embedding_dim=4)
    x = torch.zeros(1, 5, 4)
    out = pe(x)
    # Output is exactly the position embedding at zero input.
    assert torch.allclose(out, pe.table.weight[:5].unsqueeze(0))


def test_learned_positional_rejects_too_long_sequence() -> None:
    pe = LearnedPositionalEmbedding(block_size=4, embedding_dim=4)
    with pytest.raises(ValueError):
        pe(torch.zeros(1, 5, 4))


# --- Sinusoidal --------------------------------------------------------


def test_sinusoidal_table_first_row_is_zero_phase() -> None:
    pe = SinusoidalPositionalEmbedding(embedding_dim=16, max_len=128)
    # Position 0: sin(0)=0 in even slots, cos(0)=1 in odd slots.
    row0 = pe.pe[0]
    even = row0[0::2]
    odd = row0[1::2]
    assert torch.allclose(even, torch.zeros_like(even), atol=1e-6)
    assert torch.allclose(odd, torch.ones_like(odd), atol=1e-6)


def test_sinusoidal_rejects_odd_dim() -> None:
    with pytest.raises(ValueError):
        SinusoidalPositionalEmbedding(embedding_dim=15)


# --- Rotary -----------------------------------------------------------


def test_rope_preserves_norms() -> None:
    rope = RotaryPositionalEmbedding(head_dim=8, max_len=32)
    q = torch.randn(2, 4, 16, 8)  # (B, H, T, head_dim)
    k = torch.randn(2, 4, 16, 8)
    q_rot, k_rot = rope.apply_rotary(q, k)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)


def test_rope_rotation_is_relative() -> None:
    # The dot product of rotated q at position i with rotated k at
    # position j should equal the dot product of rotated q at i+d with
    # rotated k at j+d for any d, demonstrating relative-position behavior.
    torch.manual_seed(0)
    rope = RotaryPositionalEmbedding(head_dim=8, max_len=32)
    q_base = torch.randn(1, 1, 8)  # one vector
    k_base = torch.randn(1, 1, 8)
    # Place them at positions (3, 5) and at (7, 9).
    q_a = q_base.expand(1, 32, 8).clone()
    k_a = k_base.expand(1, 32, 8).clone()
    q_rot, k_rot = rope.apply_rotary(q_a, k_a)
    dot_3_5 = (q_rot[0, 3] * k_rot[0, 5]).sum()
    dot_7_9 = (q_rot[0, 7] * k_rot[0, 9]).sum()
    assert torch.allclose(dot_3_5, dot_7_9, atol=1e-4)


def test_rope_rejects_odd_dim() -> None:
    with pytest.raises(ValueError):
        RotaryPositionalEmbedding(head_dim=7)


# --- ALiBi ------------------------------------------------------------


def test_alibi_slopes_power_of_2() -> None:
    a = ALiBi(num_heads=8)
    # Published formula: slopes are 2^{-8/n}, 2^{-16/n}, ..., 2^{-8}.
    expected_first = 2.0 ** (-8 / 8)
    expected_last = 2.0 ** (-8)
    assert math.isclose(float(a.slopes[0]), expected_first, rel_tol=1e-5)
    assert math.isclose(float(a.slopes[-1]), expected_last, rel_tol=1e-5)


def test_alibi_bias_shape() -> None:
    a = ALiBi(num_heads=4, max_len=16)
    b = a(10)
    assert b.shape == (4, 10, 10)


def test_alibi_bias_is_zero_on_diagonal() -> None:
    a = ALiBi(num_heads=4, max_len=8)
    b = a(8)
    diag = b.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag))


# --- Factory ----------------------------------------------------------


def test_factory_dispatches() -> None:
    pe1 = make_positional("learned", block_size=4, embedding_dim=4)
    assert isinstance(pe1, LearnedPositionalEmbedding)
    pe2 = make_positional("sinusoidal", embedding_dim=4)
    assert isinstance(pe2, SinusoidalPositionalEmbedding)
    pe3 = make_positional("rotary", head_dim=4)
    assert isinstance(pe3, RotaryPositionalEmbedding)
    pe4 = make_positional("alibi", num_heads=4)
    assert isinstance(pe4, ALiBi)


def test_factory_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        make_positional("nonsense")  # type: ignore[arg-type]
