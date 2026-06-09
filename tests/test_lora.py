"""Tests for Chapter 24 LoRA and helpers."""

import torch
import torch.nn as nn

from lm_from_scratch.finetuning.lora import (
    LoRALinear,
    apply_lora,
    lora_state_dict,
    trainable_param_count,
)
from lm_from_scratch.finetuning.rag_stub import RetrievalStub
from lm_from_scratch.finetuning.synthetic import SyntheticGenerator, filter_synthetic


def test_lora_at_init_matches_base() -> None:
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    wrapped = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(2, 8)
    # B is zero-initialized, so output should equal base(x).
    assert torch.allclose(wrapped(x), base(x), atol=1e-6)


def test_lora_only_a_and_b_are_trainable() -> None:
    base = nn.Linear(16, 32)
    wrapped = LoRALinear(base, rank=4, alpha=8.0)
    trainable = [n for n, p in wrapped.named_parameters() if p.requires_grad]
    assert "base.weight" not in trainable
    assert "A" in trainable and "B" in trainable


def test_apply_lora_modifies_target_modules() -> None:
    mod = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    # Children are named "0" and "1". We replace any child named with "0".
    apply_lora(mod, target_module_names=["0"], rank=2, alpha=4.0)
    assert isinstance(mod[0], LoRALinear)
    assert not isinstance(mod[1], LoRALinear)


def test_synthetic_filter_dedup_and_length() -> None:
    pairs = [("p1", "short"), ("p2", "a much longer response here"), ("p3", "short")]
    out = filter_synthetic(pairs, min_response_length=10, dedup=True)
    assert len(out) == 1


def test_retrieval_stub_recovers_exact_match() -> None:
    docs = ["banking complaint about overdraft fees", "weather forecast for Tuesday", "recipe for sourdough"]
    r = RetrievalStub(docs)
    top = r.retrieve("overdraft fee complaint", top_k=1)
    assert top[0][0] == 0
