"""A minimal training loop.

Intentionally simple: AdamW, no schedule, no callbacks, no checkpointing.
Used in Chapter 6 to train the first neural language models with as
little ceremony as possible. The full ``Trainer`` arrives in Chapter 15.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainingHistory:
    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    valid_loss: list[float] = field(default_factory=list)


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            total_loss += float(model.loss(x, y).item())
            total_batches += 1
    model.train()
    return total_loss / max(1, total_batches)


def simple_train(
    model: torch.nn.Module,
    train_dataset: Dataset,
    valid_dataset: Dataset | None,
    *,
    batch_size: int = 64,
    lr: float = 3e-3,
    max_steps: int = 2_000,
    eval_every: int = 200,
    device: str | torch.device = "cpu",
    weight_decay: float = 0.01,
) -> TrainingHistory:
    """Train ``model`` for ``max_steps`` AdamW steps.

    The model must expose a ``loss(input_ids, target_ids)`` method. Returns
    a :class:`TrainingHistory` with periodic loss measurements.
    """
    device = torch.device(device)
    model.to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = (
        DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        if valid_dataset is not None
        else None
    )
    history = TrainingHistory()
    step = 0
    while step < max_steps:
        for x, y in train_loader:
            if step >= max_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss = model.loss(x, y)
            loss.backward()
            optim.step()
            step += 1
            if step % eval_every == 0 or step == 1:
                history.steps.append(step)
                history.train_loss.append(float(loss.item()))
                history.valid_loss.append(_evaluate(model, valid_loader, device) if valid_loader else float("nan"))
    return history
