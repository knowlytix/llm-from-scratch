"""Intrinsic evaluation: combines perplexity, accuracy and calibration into one report."""

from __future__ import annotations

from typing import Any

import torch

from lm_from_scratch.evaluation.diagnostics import token_accuracy
from lm_from_scratch.evaluation.perplexity import evaluate_perplexity


def evaluate_intrinsic(
    model: torch.nn.Module,
    dataset,
    *,
    batch_size: int = 64,
    device: str | torch.device | None = None,
    bytes_per_token: float | None = None,
) -> dict[str, Any]:
    """Composed report: PPL, BPB, token accuracy."""
    rep = evaluate_perplexity(model, dataset, batch_size=batch_size, device=device,
                              bytes_per_token=bytes_per_token)
    rep["token_accuracy"] = token_accuracy(model, dataset, batch_size=batch_size, device=device)
    return rep
