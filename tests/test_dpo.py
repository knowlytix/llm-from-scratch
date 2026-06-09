"""Tests for Chapter 25 DPO and reasoning."""

import math

import torch

from lm_from_scratch.finetuning.dpo import dpo_loss
from lm_from_scratch.finetuning.reasoning import extract_answer, format_cot


def test_dpo_loss_zero_margin_is_log_2() -> None:
    z = torch.zeros(1)
    loss = dpo_loss(z, z, z, z, beta=0.1)
    assert math.isclose(loss.item(), math.log(2), abs_tol=1e-5)


def test_dpo_loss_gradient_pushes_chosen_up() -> None:
    pc = torch.tensor([0.0], requires_grad=True)
    pr = torch.tensor([0.0], requires_grad=True)
    rc = torch.tensor([0.0])
    rr = torch.tensor([0.0])
    loss = dpo_loss(pc, pr, rc, rr, beta=0.5)
    loss.backward()
    # Gradient sign on chosen should be negative (we want to increase chosen logp).
    assert pc.grad.item() < 0
    assert pr.grad.item() > 0


def test_format_cot_includes_reasoning_and_answer() -> None:
    s = format_cot("Q?", "step1; step2", "yes")
    assert "Think:" in s and "Answer: yes" in s


def test_extract_answer() -> None:
    assert extract_answer("blah\nAnswer: 42") == "42"
    assert extract_answer("no marker here") == "no marker here"
