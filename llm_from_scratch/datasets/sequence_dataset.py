"""Causal language modeling dataset.

Slides a window of ``block_size`` tokens over a flat token stream. For each
position we return an ``(x, y)`` pair where ``y`` is ``x`` shifted left by
one. This is the standard input/target construction for training a causal
LM with cross-entropy loss.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class CausalLMDataset(Dataset):
    """Sliding-window causal language modeling dataset.

    Parameters
    ----------
    token_ids:
        Flat sequence of token ids as a Python list, NumPy array or 1-D
        torch tensor.
    block_size:
        Number of tokens per training example.
    stride:
        Step between consecutive windows. Defaults to ``block_size`` (no
        overlap). Set smaller for more training examples at the cost of
        overlap; do not use a small stride in evaluation (Chapter 2).
    """

    def __init__(
        self,
        token_ids: list[int] | np.ndarray | torch.Tensor,
        block_size: int,
        stride: int | None = None,
    ) -> None:
        if stride is None:
            stride = block_size
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.cpu().numpy()
        else:
            ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError(f"token_ids must be 1-D, got shape {ids.shape}")
        # +1 so we always have a target for the last input position.
        if ids.shape[0] < block_size + 1:
            raise ValueError(
                f"need at least block_size+1 = {block_size + 1} tokens, got {ids.shape[0]}"
            )
        self.ids = torch.from_numpy(ids).long()
        self.block_size = block_size
        self.stride = stride
        self._length = max(0, (ids.shape[0] - block_size - 1) // stride + 1)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        start = idx * self.stride
        x = self.ids[start : start + self.block_size]
        y = self.ids[start + 1 : start + self.block_size + 1]
        return x, y
