"""Collators stack batches of training examples into single tensors.

For the causal LM dataset every example has the same length, so the
collator simply stacks them. Variable-length collators arrive in later
chapters with instruction tuning and preference data.
"""

from __future__ import annotations

import torch


def causal_lm_collator(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack a list of ``(x, y)`` pairs into ``(X, Y)`` batched tensors."""
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys
