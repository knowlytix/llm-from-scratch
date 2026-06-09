"""Training diagnostics: gradient monitor, NaN detector, dead-neuron tracker,
overfit-one-batch smoke test."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn

from lm_from_scratch.training.callbacks import Callback


class GradientMonitor(Callback):
    """Record the global gradient norm at each evaluation."""

    def __init__(self) -> None:
        self.steps: list[int] = []
        self.norms: list[float] = []

    def on_eval_end(self, trainer, step, metrics) -> None:  # noqa: ANN001
        g = metrics.get("grad_norm")
        if g is not None:
            self.steps.append(step)
            self.norms.append(float(g))


class NaNDetector(Callback):
    """Raise on the first NaN or Inf in the latest train loss."""

    def on_step_end(self, trainer, step, train_loss) -> None:  # noqa: ANN001
        if not torch.isfinite(torch.tensor(train_loss)).item():
            raise RuntimeError(f"NaN or Inf loss at step {step}: {train_loss}")


class ActivationMonitor:
    """Track per-layer activation statistics via forward hooks.

    Use as a context manager:

        with ActivationMonitor(model, modules=[m for m in model.blocks]) as mon:
            ... run a forward pass ...
        stats = mon.summary()
    """

    def __init__(self, model: nn.Module, modules: list[nn.Module]) -> None:
        self.model = model
        self.modules = modules
        self._records: dict[int, list[dict[str, float]]] = defaultdict(list)
        self._hooks: list[Any] = []

    def __enter__(self) -> "ActivationMonitor":
        for i, m in enumerate(self.modules):
            self._hooks.append(m.register_forward_hook(self._make_hook(i)))
        return self

    def __exit__(self, *args: Any) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_hook(self, idx: int):
        def _hook(module, inputs, output):
            t = output if isinstance(output, torch.Tensor) else output[0]
            self._records[idx].append(
                {
                    "mean": float(t.mean().item()),
                    "std": float(t.std().item()),
                    "dead_fraction": float((t == 0).float().mean().item()),
                }
            )
        return _hook

    def summary(self) -> dict[int, dict[str, float]]:
        out: dict[int, dict[str, float]] = {}
        for idx, recs in self._records.items():
            n = max(1, len(recs))
            out[idx] = {
                "mean": sum(r["mean"] for r in recs) / n,
                "std": sum(r["std"] for r in recs) / n,
                "dead_fraction": sum(r["dead_fraction"] for r in recs) / n,
            }
        return out


def overfit_one_batch(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    steps: int = 200,
    lr: float = 3e-3,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Train the model on a single batch for ``steps`` steps. Returns ``{final_loss, initial_loss}``.

    A healthy model should drive the loss on one batch close to zero. If
    it cannot, the model, the data pipeline or the loss is broken.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    initial = float(model.loss(x, y).item())
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = model.loss(x, y)
        loss.backward()
        opt.step()
    final = float(model.loss(x, y).item())
    return {"initial_loss": initial, "final_loss": final, "steps": steps}
