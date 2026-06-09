"""Tests for Chapter 27 failure-mode probes."""

import torch

from llm_from_scratch.evaluation.failure_modes.probes import (
    hallucination_check,
    prompt_sensitivity,
    refusal_probe,
    template_bias_test,
)


class _StubModel(torch.nn.Module):
    """A tiny LM that always predicts the same next token."""
    def __init__(self, V=20):
        super().__init__()
        self.V = V
        self.head = torch.nn.Linear(8, V, bias=False)
        torch.manual_seed(0)
    def forward(self, x, target_ids=None):
        B, T = x.shape
        return torch.zeros(B, T, self.V), None
    def loss(self, x, y):
        return torch.zeros(())


class _StubTok:
    pad_id = 0
    def encode(self, s): return [1, 2, 3]
    def decode(self, ids): return "stub" + str(len(ids))


def test_hallucination_check_returns_results() -> None:
    model = _StubModel(); tok = _StubTok()
    out = hallucination_check(model, tok, [("Q?", "X")])
    assert "results" in out
    assert "correct_fraction" in out


def test_prompt_sensitivity_runs() -> None:
    model = _StubModel(); tok = _StubTok()
    perts = [lambda p: p, lambda p: p.upper()]
    out = prompt_sensitivity(model, tok, ["hello"], perts)
    assert "results" in out


def test_template_bias_returns_deltas() -> None:
    model = _StubModel(); tok = _StubTok()
    out = template_bias_test(model, tok, "the {} is", [("king", "queen")])
    assert len(out["deltas"]) == 1


def test_refusal_probe_zero_with_no_keywords() -> None:
    model = _StubModel(); tok = _StubTok()
    out = refusal_probe(model, tok, ["hi", "bye"], ["I cannot"])
    assert out["refusal_rate"] == 0.0
