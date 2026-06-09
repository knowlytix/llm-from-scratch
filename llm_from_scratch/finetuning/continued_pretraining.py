"""Continued pretraining: ordinary causal LM training on a new corpus."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


def continued_pretrain(model: torch.nn.Module, dataset: Dataset, *,
                       batch_size: int = 16, lr: float = 1e-4,
                       max_steps: int = 500, eval_every: int = 50,
                       device=None) -> dict:
    """Train an existing model on a domain corpus with the usual LM loss."""
    if device is None: device = next(model.parameters()).device
    model.to(device).train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    history = {"steps": [], "loss": []}
    step = 0
    while step < max_steps:
        for x, y in loader:
            if step >= max_steps: break
            x = x.to(device); y = y.to(device)
            loss = model.loss(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0 or step == 1:
                history["steps"].append(step); history["loss"].append(float(loss.item()))
    return history
