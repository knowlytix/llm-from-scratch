"""Calibration: expected calibration error and reliability diagram."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def expected_calibration_error(probs: np.ndarray, correct: np.ndarray, num_bins: int = 15) -> float:
    """Standard ECE: bin by predicted probability, compare bin accuracy to bin confidence."""
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    n = len(probs)
    ece = 0.0
    for b in range(num_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(probs[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_diagram(probs: np.ndarray, correct: np.ndarray, num_bins: int = 15) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    accs = np.zeros(num_bins)
    confs = np.zeros(num_bins)
    for b in range(num_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        accs[b] = correct[mask].mean()
        confs[b] = probs[mask].mean()
    return confs, accs


@torch.no_grad()
def gather_top1_probs(model, dataset: Dataset, batch_size: int = 64, device=None, max_tokens: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    """Return (top1 prob, correct?) arrays for the model's argmax predictions on the dataset."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs_all: list[float] = []
    correct_all: list[int] = []
    seen = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits, _ = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        p = torch.softmax(logits, dim=-1)
        top_p, top_id = p.max(dim=-1)
        c = (top_id == y).long()
        probs_all.extend(top_p.flatten().cpu().tolist())
        correct_all.extend(c.flatten().cpu().tolist())
        seen += y.numel()
        if seen >= max_tokens:
            break
    return np.array(probs_all[:max_tokens]), np.array(correct_all[:max_tokens])
