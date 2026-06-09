"""Tests for Chapter 15 optimization, schedules and trainer."""

import torch

from llm_from_scratch.training.optim import HandRolledAdamW, param_groups_for_weight_decay
from llm_from_scratch.training.schedules import (
    cosine_decay,
    linear_warmup,
    warmup_cosine,
)
from llm_from_scratch.training.trainer import Trainer, TrainingConfig


def test_hand_rolled_adamw_matches_torch_adamw() -> None:
    torch.manual_seed(0)
    p1 = torch.nn.Linear(8, 8)
    p2 = torch.nn.Linear(8, 8)
    p2.load_state_dict(p1.state_dict())
    opt1 = torch.optim.AdamW(p1.parameters(), lr=1e-3, weight_decay=0.01)
    opt2 = HandRolledAdamW(p2.parameters(), lr=1e-3, weight_decay=0.01)
    x = torch.randn(4, 8)
    target = torch.randn(4, 8)
    for _ in range(20):
        opt1.zero_grad()
        opt2.zero_grad()
        l1 = ((p1(x) - target) ** 2).mean()
        l2 = ((p2(x) - target) ** 2).mean()
        l1.backward()
        l2.backward()
        opt1.step()
        opt2.step()
    diff = (p1.weight - p2.weight).abs().max().item()
    assert diff < 1e-5


def test_cosine_decay_monotone() -> None:
    vals = [cosine_decay(s, 100, min_lr_ratio=0.1) for s in range(0, 101, 5)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_warmup_then_decay_shape() -> None:
    vals = [warmup_cosine(s, 10, 100) for s in range(101)]
    # Increasing during warmup, decreasing afterward.
    assert vals[0] < vals[5] < vals[10]
    assert vals[50] > vals[100]


def test_linear_warmup_zero_warmup_returns_one() -> None:
    assert linear_warmup(0, 0) == 1.0


def test_param_groups_for_weight_decay_splits_correctly() -> None:
    m = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8))
    groups = param_groups_for_weight_decay(m, 0.1)
    assert len(groups) == 2
    decay, no_decay = groups
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    decay_ids = {id(p) for p in decay["params"]}
    no_decay_ids = {id(p) for p in no_decay["params"]}
    for name, p in m.named_parameters():
        if p.dim() < 2 or name.endswith(".bias"):
            assert id(p) in no_decay_ids
        else:
            assert id(p) in decay_ids


class _Toy(torch.nn.Module):
    def __init__(self, V=8):
        super().__init__()
        self.emb = torch.nn.Embedding(V, 16)
        self.head = torch.nn.Linear(16, V)
    def forward(self, x):
        return self.head(self.emb(x))
    def loss(self, x, y):
        logits = self.forward(x)
        return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def test_trainer_fits_and_records_history() -> None:
    torch.manual_seed(0)
    from llm_from_scratch.datasets.sequence_dataset import CausalLMDataset
    V = 8
    ids = [i % V for i in range(200)]
    ds = CausalLMDataset(ids, block_size=8, stride=8)
    model = _Toy(V=V)
    cfg = TrainingConfig(batch_size=4, learning_rate=3e-3, weight_decay=0.01,
                         grad_clip=1.0, warmup_steps=5, max_steps=50, eval_every=10)
    trainer = Trainer(model, ds, ds, cfg, device="cpu")
    hist = trainer.fit()
    assert len(hist.steps) >= 4
    assert hist.train_loss[-1] < hist.train_loss[0]
