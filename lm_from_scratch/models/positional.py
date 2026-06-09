"""Positional encodings: learned, sinusoidal, rotary (RoPE) and ALiBi.

Self-attention is permutation-equivariant. Without position information,
shuffling the input shuffles the output the same way. The four modules in
this file are the four standard ways to break that symmetry.

The modules are intentionally simple. Chapter 12 (transformer block) and
Chapter 13 (TinyGPT) integrate them into the model. Here, each module is
standalone so the reader can compare them on identical inputs.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Learned positional embeddings
# ---------------------------------------------------------------------------


class LearnedPositionalEmbedding(nn.Module):
    """A trainable table indexed by position."""

    def __init__(self, block_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.table = nn.Embedding(block_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")
        positions = torch.arange(T, device=x.device)
        return x + self.table(positions)


# ---------------------------------------------------------------------------
# Sinusoidal positional embeddings (Vaswani et al. 2017)
# ---------------------------------------------------------------------------


class SinusoidalPositionalEmbedding(nn.Module):
    r"""Fixed sin/cos table.

    Position \(t\), dimension pair \(2i\) and \(2i+1\):

    .. math::
        PE_{t, 2i} = \sin(t / 10000^{2i/d}), \qquad
        PE_{t, 2i+1} = \cos(t / 10000^{2i/d}).

    The table is computed at construction time up to ``max_len`` positions
    and never trained.
    """

    def __init__(self, embedding_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even, got {embedding_dim}")
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")
        return x + self.pe[:T]

    def positional_table(self, length: int | None = None) -> torch.Tensor:
        n = length if length is not None else self.max_len
        return self.pe[:n].detach().clone()


# ---------------------------------------------------------------------------
# Rotary positional embeddings (Su et al. 2021, RoFormer)
# ---------------------------------------------------------------------------


class RotaryPositionalEmbedding(nn.Module):
    r"""Rotary positional embedding (RoPE).

    Each consecutive pair of dimensions \((2i, 2i+1)\) is treated as a
    point in \(\mathbb{R}^2\) and rotated by an angle \(t \theta_i\) at
    position \(t\), where \(\theta_i = 10000^{-2i/d_k}\).

    Apply the rotation only to queries and keys, never to values. The
    inner product of rotated query and key depends only on the relative
    position offset, which is the entire reason RoPE works.
    """

    def __init__(self, head_dim: int, max_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.max_len = max_len
        self.base = base
        # Frequencies per dimension pair.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_len, dtype=torch.float32)
        # Outer product produces angles at each position.
        angles = torch.einsum("i,j->ij", t, inv_freq)  # (max_len, head_dim/2)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def _rotate(self, x: torch.Tensor, T: int) -> torch.Tensor:
        # x: (..., T, head_dim). Split last dim into pairs.
        x_even = x[..., 0::2]  # (..., T, head_dim/2)
        x_odd = x[..., 1::2]
        cos = self.cos[:T]  # (T, head_dim/2)
        sin = self.sin[:T]
        # Broadcast cos/sin against the leading dims of x.
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        # Interleave back into the original layout.
        out = torch.empty_like(x)
        out[..., 0::2] = rot_even
        out[..., 1::2] = rot_odd
        return out

    def apply_rotary(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate Q and K in place along the last dimension."""
        T = q.size(-2)
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")
        return self._rotate(q, T), self._rotate(k, T)


# ---------------------------------------------------------------------------
# ALiBi (Press et al. 2021)
# ---------------------------------------------------------------------------


def _alibi_slopes(num_heads: int) -> torch.Tensor:
    r"""Per-head slopes from the published recipe.

    For :math:`n` heads with :math:`n` a power of 2, the slopes are
    :math:`2^{-8/n}, 2^{-16/n}, \dots, 2^{-8}`. The published recipe
    interpolates for non-power-of-2 head counts; for simplicity we
    fall back to a geometric series in those cases.
    """
    def _power_of_2_slopes(n: int) -> list[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3.0)))
        return [start * start ** i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(_power_of_2_slopes(num_heads), dtype=torch.float32)
    # Geometric fallback.
    return torch.tensor(
        [2.0 ** (-8.0 * (i + 1) / num_heads) for i in range(num_heads)], dtype=torch.float32
    )


class ALiBi(nn.Module):
    r"""Attention with Linear Biases.

    Produces a per-head additive bias of shape ``(num_heads, T, T)`` that
    is added to attention scores before the softmax. The bias is
    :math:`-m_h \cdot |i - j|` for head :math:`h` with slope :math:`m_h`.
    """

    def __init__(self, num_heads: int, max_len: int = 4096) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        slopes = _alibi_slopes(num_heads)
        # Distance matrix |i - j|.
        i = torch.arange(max_len).unsqueeze(1)
        j = torch.arange(max_len).unsqueeze(0)
        dist = -(i - j).float().abs()  # negative absolute distance
        # Shape (num_heads, max_len, max_len).
        bias = slopes.view(-1, 1, 1) * dist.unsqueeze(0)
        self.register_buffer("bias", bias, persistent=False)
        self.register_buffer("slopes", slopes, persistent=False)

    def forward(self, T: int) -> torch.Tensor:
        """Return the ``(num_heads, T, T)`` bias to add to attention scores."""
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")
        return self.bias[:, :T, :T]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


PositionalName = Literal["learned", "sinusoidal", "rotary", "alibi"]


def make_positional(name: PositionalName, **kwargs) -> nn.Module:
    """Construct a positional encoding by name with keyword args."""
    if name == "learned":
        return LearnedPositionalEmbedding(**kwargs)
    if name == "sinusoidal":
        return SinusoidalPositionalEmbedding(**kwargs)
    if name == "rotary":
        return RotaryPositionalEmbedding(**kwargs)
    if name == "alibi":
        return ALiBi(**kwargs)
    raise ValueError(f"unknown positional encoding: {name}")
