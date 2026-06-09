"""The reusable Transformer block.

A block is the unit that stacks to form a Transformer. The book uses a
pre-norm architecture by default: layer normalization is applied before
each sublayer, then the sublayer's output is added to the input
(residual). Pre-norm trains deeper stacks reliably; post-norm is shown
for comparison in the chapter's notebook.

The MLP supports GELU and SwiGLU activations. SwiGLU costs slightly
more parameters at the same hidden width because of the gate; the
canonical comparison normalizes for parameter count.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from llm_from_scratch.models.attention import MultiHeadCausalSelfAttention
from llm_from_scratch.models.norms import RMSNorm


Activation = Literal["gelu", "relu", "swiglu"]
NormStyle = Literal["pre", "post"]


class MLP(nn.Module):
    """Two-layer feedforward with a configurable activation.

    For ``gelu`` and ``relu`` activations the MLP is
    ``Linear -> activation -> Linear`` with hidden width ``mlp_ratio * dim``.

    For ``swiglu`` the gate is implemented as
    ``Linear(gate) * Silu(Linear(up))``, followed by ``Linear(down)``.
    """

    def __init__(self, dim: int, mlp_ratio: int = 4, activation: Activation = "gelu") -> None:
        super().__init__()
        self.activation = activation
        hidden = mlp_ratio * dim
        if activation == "swiglu":
            self.gate_proj = nn.Linear(dim, hidden, bias=False)
            self.up_proj = nn.Linear(dim, hidden, bias=False)
            self.down_proj = nn.Linear(hidden, dim, bias=False)
        elif activation in ("gelu", "relu"):
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, dim)
        else:
            raise ValueError(f"unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            return self.down_proj(self.gate_proj(x) * torch.nn.functional.silu(self.up_proj(x)))
        if self.activation == "gelu":
            return self.fc2(torch.nn.functional.gelu(self.fc1(x)))
        return self.fc2(torch.nn.functional.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    """One Transformer block: attention sublayer + MLP sublayer with residuals.

    Pre-norm:
        ``x = x + attn(LN(x))``
        ``x = x + mlp(LN(x))``

    Post-norm:
        ``x = LN(x + attn(x))``
        ``x = LN(x + mlp(x))``
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        block_size: int = 1024,
        norm_style: NormStyle = "pre",
        activation: Activation = "gelu",
        norm_class: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm_style = norm_style
        self.ln1 = norm_class(embedding_dim)
        self.attn = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
        )
        self.ln2 = norm_class(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_ratio=mlp_ratio, activation=activation)
        self.dropout = nn.Dropout(dropout)

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pre-norm only: cached forward, returns new K/V to store in the cache."""
        from llm_from_scratch.models.attention import mha_forward_with_cache

        attn_in = self.ln1(x)
        attn_out, new_k, new_v = mha_forward_with_cache(self.attn, attn_in, past_k, past_v)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x, new_k, new_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_style == "pre":
            x = x + self.dropout(self.attn(self.ln1(x)))
            x = x + self.dropout(self.mlp(self.ln2(x)))
        else:  # post
            x = self.ln1(x + self.dropout(self.attn(x)))
            x = self.ln2(x + self.dropout(self.mlp(x)))
        return x
