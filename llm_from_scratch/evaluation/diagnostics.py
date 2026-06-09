"""Diagnostic metrics that complement perplexity."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def token_accuracy(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 64,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
) -> float:
    """Fraction of positions where the argmax prediction equals the target."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    correct = 0
    total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model(x) if hasattr(model, "forward") and "target_ids" in model.forward.__code__.co_varnames else (model(x), None)
        # Defensive: some models return only logits, some return (logits, loss).
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = logits.argmax(dim=-1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)


@torch.no_grad()
def per_position_loss(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 64,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
) -> torch.Tensor:
    """Mean cross-entropy at each position in the sequence.

    Returns a 1-D tensor of length ``block_size``.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    sums: torch.Tensor | None = None
    counts: torch.Tensor | None = None
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model(x) if "target_ids" in model.forward.__code__.co_varnames else (model(x), None)
        if isinstance(logits, tuple):
            logits = logits[0]
        # per-position cross-entropy
        per_tok = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), y, reduction="none"
        )  # (B, T)
        sum_b = per_tok.sum(dim=0)
        cnt_b = torch.full_like(sum_b, fill_value=float(per_tok.size(0)))
        if sums is None:
            sums = sum_b
            counts = cnt_b
        else:
            sums = sums + sum_b
            counts = counts + cnt_b
    assert sums is not None and counts is not None
    return (sums / counts).cpu()
