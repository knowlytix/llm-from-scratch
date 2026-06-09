"""Configuration dataclass for TinyGPT.

The single source of truth for a model's architecture. Two TinyGPTs with
the same parameter count and different ``GPTConfig`` fields are different
artifacts. The reproducibility contract is: code commit + tokenizer file
+ ``GPTConfig`` + checkpoint + seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class GPTConfig:
    """Architecture description for ``TinyGPT``.

    Attributes
    ----------
    vocab_size:
        Tokenizer vocabulary size.
    block_size:
        Maximum context length the model is trained at.
    embedding_dim:
        Width of token and position embeddings.
    num_layers:
        Number of Transformer blocks.
    num_heads:
        Number of attention heads per block.
    mlp_ratio:
        Hidden-to-input ratio of the MLP sublayer.
    dropout:
        Dropout probability (after each sublayer; in attention and MLP).
    positional:
        Which positional encoding to use. ``"learned"`` and
        ``"sinusoidal"`` are added to the input embedding; ``"rotary"``
        and ``"alibi"`` are not (they live inside attention). The
        TinyGPT in this book uses learned by default.
    activation:
        MLP activation: ``"gelu"``, ``"relu"`` or ``"swiglu"``.
    norm_style:
        ``"pre"`` (default, modern) or ``"post"`` (original Transformer).
    tie_embeddings:
        If True, share the token embedding matrix with the LM head.
    init_std:
        Standard deviation of the normal initialization for linear
        weights. The output projections in each block are additionally
        scaled by ``1 / sqrt(2 * num_layers)`` per the GPT-2 recipe.
    """

    vocab_size: int = 1024
    block_size: int = 128
    embedding_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1
    positional: Literal["learned", "sinusoidal"] = "learned"
    activation: Literal["gelu", "relu", "swiglu"] = "gelu"
    norm_style: Literal["pre", "post"] = "pre"
    tie_embeddings: bool = True
    init_std: float = 0.02
