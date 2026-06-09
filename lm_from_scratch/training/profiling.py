"""A small training-step benchmarker."""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader, Dataset


def benchmark_step(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 32,
    steps: int = 50,
    warmup: int = 10,
    device: str | torch.device | None = None,
    autocast_dtype: str | None = None,
) -> dict[str, float]:
    """Measure tokens/sec, step time and peak memory."""
    if device is None:
        device = next(model.parameters()).device
    model.to(device).train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    def _step(x, y):
        optim.zero_grad(set_to_none=True)
        if autocast_dtype is not None and device.type == "cuda":
            from lm_from_scratch.training.efficiency import autocast_context
            with autocast_context(autocast_dtype):
                loss = model.loss(x, y)
        else:
            loss = model.loss(x, y)
        loss.backward()
        optim.step()

    it = iter(loader)
    for _ in range(warmup):
        x, y = next(it)
        _step(x.to(device), y.to(device))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    total_tokens = 0
    for _ in range(steps):
        x, y = next(it)
        _step(x.to(device), y.to(device))
        total_tokens += x.numel()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = (
        torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
    )
    return {
        "steps": steps,
        "elapsed_sec": elapsed,
        "step_sec": elapsed / max(1, steps),
        "tokens_per_sec": total_tokens / elapsed,
        "peak_memory_bytes": int(peak),
    }
