"""Tests for Chapter 16 training diagnostics."""

import pytest
import torch
import torch.nn as nn

from llm_from_scratch.datasets.sequence_dataset import CausalLMDataset
from llm_from_scratch.training.diagnostics import (
    ActivationMonitor,
    GradientMonitor,
    NaNDetector,
    overfit_one_batch,
)


class _Toy(nn.Module):
    def __init__(self, V=8):
        super().__init__()
        self.emb = nn.Embedding(V, 16)
        self.head = nn.Linear(16, V)
    def forward(self, x):
        return self.head(self.emb(x))
    def loss(self, x, y):
        logits = self.forward(x)
        return nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def test_overfit_one_batch_drives_loss_down() -> None:
    torch.manual_seed(0)
    V = 8
    ds = CausalLMDataset([i % V for i in range(64)], block_size=8, stride=8)
    x, y = ds[0]
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    out = overfit_one_batch(_Toy(V=V), (x, y), steps=200, lr=3e-3)
    assert out["final_loss"] < 0.5  # below ln(V)
    assert out["final_loss"] < out["initial_loss"]


def test_nan_detector_raises_on_nan_loss() -> None:
    det = NaNDetector()
    with pytest.raises(RuntimeError):
        det.on_step_end(trainer=None, step=10, train_loss=float("nan"))


def test_gradient_monitor_records() -> None:
    mon = GradientMonitor()
    mon.on_eval_end(trainer=None, step=100, metrics={"grad_norm": 1.5})
    mon.on_eval_end(trainer=None, step=200, metrics={"grad_norm": 1.2})
    assert mon.steps == [100, 200]
    assert mon.norms == [1.5, 1.2]


def test_activation_monitor_records_stats() -> None:
    m = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    with ActivationMonitor(m, modules=[m[0], m[1]]) as mon:
        out = m(torch.randn(2, 8))
    stats = mon.summary()
    assert 0 in stats and 1 in stats
    # ReLU output has dead_fraction > 0 on average.
    assert stats[1]["dead_fraction"] > 0
