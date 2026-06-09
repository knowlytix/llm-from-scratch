"""Attention as soft retrieval, built up incrementally across chapters.

Chapter 8 introduces ``scaled_dot_product_attention`` and ``SingleHeadAttention``.
Chapter 9 adds the causal mask and ``SingleHeadCausalSelfAttention``.
Chapter 10 adds ``MultiHeadCausalSelfAttention``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    r"""Scaled dot-product attention.

    Computes

    .. math::

        \alpha_{ij} = \softmax\!\left(\frac{q_i \cdot k_j}{\sqrt{d_k}} + M_{ij}\right),
        \qquad \text{out}_i = \sum_j \alpha_{ij} v_j.

    Parameters
    ----------
    q:
        Query tensor of shape ``(..., Tq, dk)``.
    k:
        Key tensor of shape ``(..., Tk, dk)``.
    v:
        Value tensor of shape ``(..., Tk, dv)``.
    mask:
        Optional additive mask of shape broadcastable to ``(..., Tq, Tk)``.
        Use ``float('-inf')`` for positions to block.
    return_weights:
        If True, also return attention weights ``(..., Tq, Tk)``.
    """
    dk = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(dk)
    if mask is not None:
        scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    out = weights @ v
    if return_weights:
        return out, weights
    return out


class SingleHeadAttention(nn.Module):
    """A single attention head, no causal mask.

    Takes ``x`` and produces queries, keys and values via three linear
    projections, then runs scaled dot-product attention. Used in Chapter
    8's toy retrieval task; later chapters add the causal mask.
    """

    def __init__(self, embedding_dim: int, head_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_proj = nn.Linear(embedding_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if return_weights:
            out, weights = scaled_dot_product_attention(q, k, v, return_weights=True)
            return self.dropout(out), weights
        out = scaled_dot_product_attention(q, k, v)
        return self.dropout(out)


def causal_mask(
    block_size: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an upper-triangular ``-inf`` mask of shape ``(block_size, block_size)``.

    Position ``i`` is allowed to attend to positions ``0..i``. Adding this
    mask to attention scores before softmax sets the disallowed positions
    to zero weight.
    """
    mask = torch.zeros(block_size, block_size, dtype=dtype, device=device)
    mask = mask.masked_fill(
        torch.triu(torch.ones(block_size, block_size, dtype=torch.bool, device=device), diagonal=1),
        float("-inf"),
    )
    return mask


class SingleHeadCausalSelfAttention(nn.Module):
    """A single attention head with a causal mask, fit for a causal LM."""

    def __init__(
        self,
        embedding_dim: int,
        head_dim: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(embedding_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, head_dim, bias=False)
        self.out_proj = nn.Linear(head_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size
        # Buffer so the mask follows the module across devices.
        self.register_buffer("mask", causal_mask(block_size), persistent=False)

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        mask = self.mask[:T, :T]
        if return_weights:
            out, weights = scaled_dot_product_attention(q, k, v, mask=mask, return_weights=True)
            return self.out_proj(self.dropout(out)), weights
        out = scaled_dot_product_attention(q, k, v, mask=mask)
        return self.out_proj(self.dropout(out))


class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with a fused QKV projection.

    Splits ``embedding_dim`` across ``num_heads`` heads each of size
    ``head_dim = embedding_dim // num_heads``. Each head runs scaled
    dot-product attention with the same causal mask; the outputs are
    concatenated and projected back to ``embedding_dim``.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.block_size = block_size
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", causal_mask(block_size), persistent=False)

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")
        # Fused QKV; split into three tensors.
        qkv = self.qkv_proj(x)  # (B, T, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)
        # Reshape each into per-head form: (B, num_heads, T, head_dim).
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        mask = self.mask[:T, :T]
        if return_weights:
            out, weights = scaled_dot_product_attention(q, k, v, mask=mask, return_weights=True)
            # out: (B, num_heads, T, head_dim) -> (B, T, D)
            out = out.transpose(1, 2).contiguous().view(B, T, D)
            return self.out_proj(self.dropout(out)), weights
        out = scaled_dot_product_attention(q, k, v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(self.dropout(out))


def mha_forward_with_cache(
    module: "MultiHeadCausalSelfAttention",
    x: torch.Tensor,
    past_k: torch.Tensor | None = None,
    past_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward through MHA returning ``(out, new_full_k, new_full_v)``.

    ``past_k`` and ``past_v`` are the cached K/V for prior tokens (or
    None for prefill). The returned full-cache tensors include the new
    tokens appended.
    """
    B, T, D = x.shape
    qkv = module.qkv_proj(x)
    q, k, v = qkv.chunk(3, dim=-1)
    H = module.num_heads
    d_h = module.head_dim
    q = q.view(B, T, H, d_h).transpose(1, 2)
    k = k.view(B, T, H, d_h).transpose(1, 2)
    v = v.view(B, T, H, d_h).transpose(1, 2)
    if past_k is not None:
        full_k = torch.cat([past_k, k], dim=-2)
        full_v = torch.cat([past_v, v], dim=-2)
    else:
        full_k = k
        full_v = v
    # During prefill (T > 1 with no past), apply causal mask over T.
    if past_k is None and T > 1:
        mask = module.mask[:T, :T]
    else:
        mask = None
    out = scaled_dot_product_attention(q, full_k, full_v, mask=mask)
    out = out.transpose(1, 2).contiguous().view(B, T, D)
    out = module.out_proj(out)
    return out, full_k, full_v


class TinyMHACausalLM(nn.Module):
    """Minimal causal LM used in Chapter 10: token + position embeddings + MHA + LM head."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_heads: int = 4,
        block_size: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.attn = MultiHeadCausalSelfAttention(
            embedding_dim, num_heads, block_size, dropout=dropout
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.block_size = block_size

    def forward(
        self, input_ids: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        T = input_ids.size(1)
        positions = torch.arange(T, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        if return_attn:
            attn_out, weights = self.attn(x, return_weights=True)
            x = self.norm(x + attn_out)
            return self.lm_head(x), weights
        x = self.norm(x + self.attn(x))
        return self.lm_head(x)

    def loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        from llm_from_scratch.training.losses import cross_entropy_loss

        return cross_entropy_loss(self.forward(input_ids), target_ids)


class TinyCausalAttnLM(nn.Module):
    """A minimal causal LM used in Chapter 9: embeddings + one attention head + LM head."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        head_dim: int = 64,
        block_size: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.attn = SingleHeadCausalSelfAttention(embedding_dim, head_dim, block_size, dropout=dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.block_size = block_size

    def forward(
        self, input_ids: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        T = input_ids.size(1)
        positions = torch.arange(T, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        if return_attn:
            attn_out, weights = self.attn(x, return_weights=True)
            x = self.norm(x + attn_out)
            return self.lm_head(x), weights
        x = self.norm(x + self.attn(x))
        return self.lm_head(x)

    def loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        from llm_from_scratch.training.losses import cross_entropy_loss

        return cross_entropy_loss(self.forward(input_ids), target_ids)
