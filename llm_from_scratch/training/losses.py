"""Loss functions used throughout the book.

A thin wrapper around ``torch.nn.functional.cross_entropy`` so that every
chapter calls the same function with the same default. The wrapper exists
so the book has a single place to extend (label smoothing, masking,
weighting) when later chapters need it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Standard cross-entropy over a flattened ``(B, T, V)`` logit tensor.

    Parameters
    ----------
    logits:
        Tensor of shape ``(B, T, V)`` or ``(N, V)``.
    targets:
        Tensor of shape ``(B, T)`` or ``(N,)`` with class indices.
    ignore_index:
        Target value to ignore. Defaults to -100, matching PyTorch.
    reduction:
        ``"mean"``, ``"sum"`` or ``"none"``.
    """
    if logits.dim() == 3:
        B, T, V = logits.shape
        logits = logits.reshape(B * T, V)
        targets = targets.reshape(B * T)
    return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction=reduction)
