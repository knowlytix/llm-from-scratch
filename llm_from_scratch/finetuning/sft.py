"""Supervised fine-tuning: cross-entropy on response tokens only."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def sft_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Loss-masked cross-entropy.

    ``logits`` (B, T, V), ``targets`` (B, T), ``mask`` (B, T) with 1 on
    response tokens.
    """
    V = logits.size(-1)
    per_token = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0)


def sft_train(model, train_dataset, valid_dataset=None, *, batch_size=8, lr=1e-4,
              max_steps=500, eval_every=50, device=None) -> dict:
    if device is None:
        device = next(model.parameters()).device
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    history = {"steps": [], "train_loss": [], "valid_loss": []}
    step = 0
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            x, y, m = (t.to(device) for t in batch)
            logits, _ = model(x)
            loss = sft_loss(logits, y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0 or step == 1:
                history["steps"].append(step)
                history["train_loss"].append(float(loss.item()))
                if valid_dataset is not None:
                    model.eval()
                    total = 0.0; n = 0
                    with torch.no_grad():
                        for vb in DataLoader(valid_dataset, batch_size=batch_size):
                            vx, vy, vm = (t.to(device) for t in vb)
                            vlogits, _ = model(vx)
                            total += float(sft_loss(vlogits, vy, vm).item())
                            n += 1
                    history["valid_loss"].append(total / max(1, n))
                    model.train()
                else:
                    history["valid_loss"].append(float("nan"))
    return history
