"""The Trainer used from Chapter 15 onward.

Mid-level convenience over a manual training loop: configurable AdamW
with decoupled weight decay, gradient clipping, warmup + cosine schedule,
gradient accumulation, and a small callback API. No mixed precision or
distributed training; those come in Chapter 18.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from llm_from_scratch.training.callbacks import Callback
from llm_from_scratch.training.optim import param_groups_for_weight_decay
from llm_from_scratch.training.schedules import warmup_cosine


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    warmup_steps: int = 100
    max_steps: int = 5000
    min_lr_ratio: float = 0.1
    eval_every: int = 250
    seed: int = 0


@dataclass
class TrainingHistory:
    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    valid_loss: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)


class Trainer:
    """A small Trainer for causal language models."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        valid_dataset: Dataset | None,
        config: TrainingConfig,
        callbacks: Iterable[Callback] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.config = config
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.callbacks: list[Callback] = list(callbacks) if callbacks else []
        self.history = TrainingHistory()
        self._stopped = False
        # Optimizer with decoupled decay groups.
        groups = param_groups_for_weight_decay(model, config.weight_decay)
        self.optimizer = torch.optim.AdamW(groups, lr=config.learning_rate)

    def stop(self) -> None:
        self._stopped = True

    @torch.no_grad()
    def _evaluate(self) -> float:
        if self.valid_dataset is None:
            return float("nan")
        loader = DataLoader(self.valid_dataset, batch_size=self.config.batch_size, shuffle=False)
        self.model.eval()
        total = 0.0
        count = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            total += float(self.model.loss(x, y).item())
            count += 1
        self.model.train()
        return total / max(1, count)

    def fit(self) -> TrainingHistory:
        torch.manual_seed(self.config.seed)
        self.model.to(self.device)
        self.model.train()
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        for cb in self.callbacks:
            cb.on_train_begin(self)
        step = 0
        accum = 0
        loss_running = 0.0
        for cb in self.callbacks:
            pass
        while step < self.config.max_steps and not self._stopped:
            for x, y in loader:
                if step >= self.config.max_steps or self._stopped:
                    break
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                # Learning-rate schedule.
                lr_scale = warmup_cosine(
                    step,
                    self.config.warmup_steps,
                    self.config.max_steps,
                    self.config.min_lr_ratio,
                )
                lr = self.config.learning_rate * lr_scale
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr
                loss = self.model.loss(x, y) / self.config.grad_accum_steps
                loss.backward()
                loss_running += float(loss.item()) * self.config.grad_accum_steps
                accum += 1
                if accum < self.config.grad_accum_steps:
                    continue
                # Gradient clip + step.
                gn = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                step += 1
                accum = 0
                train_loss = loss_running / self.config.grad_accum_steps
                loss_running = 0.0
                for cb in self.callbacks:
                    cb.on_step_end(self, step, train_loss)
                if step % self.config.eval_every == 0 or step == 1:
                    valid_loss = self._evaluate()
                    self.history.steps.append(step)
                    self.history.train_loss.append(train_loss)
                    self.history.valid_loss.append(valid_loss)
                    self.history.learning_rate.append(lr)
                    self.history.grad_norm.append(float(gn))
                    metrics = {
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                        "lr": lr,
                        "grad_norm": float(gn),
                    }
                    for cb in self.callbacks:
                        cb.on_eval_end(self, step, metrics)
        for cb in self.callbacks:
            cb.on_train_end(self)
        return self.history
