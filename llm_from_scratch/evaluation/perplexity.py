"""Perplexity evaluation for causal language models."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 64,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
    bytes_per_token: float | None = None,
) -> dict[str, float]:
    """Compute validation loss, perplexity and (optionally) bits per byte.

    Parameters
    ----------
    model:
        A causal LM exposing ``loss(input_ids, target_ids) -> Tensor``.
    dataset:
        A ``Dataset`` returning ``(x, y)`` pairs of equal shape.
    batch_size:
        Mini-batch size for evaluation.
    device:
        Where to run. Defaults to the model's current device.
    max_batches:
        Cap on number of batches; useful for quick spot checks.
    bytes_per_token:
        If provided, also returns bits-per-byte for cross-tokenizer
        comparison. Compute it as total corpus bytes / total token count
        from the tokenizer that produced the dataset.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    total_loss = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss = model.loss(x, y)
        # loss is a mean over the batch tokens; recover total nats.
        nt = y.numel()
        total_loss += float(loss.item()) * nt
        total_tokens += nt
    mean_nats = total_loss / max(1, total_tokens)
    out = {
        "loss_nats": mean_nats,
        "loss_bits": mean_nats / math.log(2),
        "perplexity": math.exp(mean_nats),
        "num_tokens": total_tokens,
    }
    if bytes_per_token is not None and bytes_per_token > 0:
        out["bits_per_byte"] = mean_nats / (math.log(2) * bytes_per_token)
    return out
