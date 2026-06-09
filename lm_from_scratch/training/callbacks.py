"""Callback base class and a few useful concrete callbacks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch


class Callback:
    """Hooks called by the Trainer. Override any subset."""

    def on_train_begin(self, trainer: "Trainer") -> None: ...  # type: ignore[name-defined]
    def on_train_end(self, trainer: "Trainer") -> None: ...    # type: ignore[name-defined]
    def on_step_end(self, trainer: "Trainer", step: int, train_loss: float) -> None: ...
    def on_eval_end(self, trainer: "Trainer", step: int, metrics: dict[str, float]) -> None: ...


class LoggingCallback(Callback):
    """Print a short line at each evaluation."""

    def __init__(self, every: int = 1) -> None:
        self.every = every
        self._t0: float = 0.0

    def on_train_begin(self, trainer) -> None:  # noqa: ANN001
        self._t0 = time.time()

    def on_eval_end(self, trainer, step, metrics) -> None:  # noqa: ANN001
        elapsed = time.time() - self._t0
        bits = [f"step {step:>6d}", f"t={elapsed:6.1f}s"]
        for k, v in metrics.items():
            bits.append(f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}")
        print("  ".join(bits))


class CheckpointCallback(Callback):
    """Save a checkpoint every ``every`` steps. Optionally keep only the best by validation loss."""

    def __init__(self, path: str | Path, every: int = 1000, keep_best: bool = True) -> None:
        self.path = Path(path)
        self.every = every
        self.keep_best = keep_best
        self._best: float = float("inf")

    def on_eval_end(self, trainer, step, metrics) -> None:  # noqa: ANN001
        if step % self.every != 0:
            return
        if self.keep_best:
            v = metrics.get("valid_loss")
            if v is not None and v < self._best:
                self._best = v
                from lm_from_scratch.training.checkpointing import save_checkpoint
                save_checkpoint(self.path, trainer.model, trainer.optimizer, step=step,
                                extras={"valid_loss": v})
        else:
            from lm_from_scratch.training.checkpointing import save_checkpoint
            save_checkpoint(self.path, trainer.model, trainer.optimizer, step=step)


class EarlyStoppingCallback(Callback):
    """Stop training if validation loss hasn't improved for ``patience`` evals."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best: float = float("inf")
        self._misses: int = 0

    def on_eval_end(self, trainer, step, metrics) -> None:  # noqa: ANN001
        v = metrics.get("valid_loss")
        if v is None:
            return
        if v < self._best - self.min_delta:
            self._best = v
            self._misses = 0
        else:
            self._misses += 1
            if self._misses >= self.patience:
                trainer.stop()
