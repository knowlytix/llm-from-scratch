"""Tests for Chapter 18 efficiency helpers (CPU-friendly subset)."""

import torch

from llm_from_scratch.training.efficiency import autocast_context, use_fused_sdpa


def test_use_fused_sdpa_returns_bool() -> None:
    assert isinstance(use_fused_sdpa(), bool)


def test_autocast_context_dtype_map() -> None:
    if torch.cuda.is_available():
        with autocast_context("bf16"):
            x = torch.randn(2, 2, device="cuda")
            y = x @ x
        assert y.shape == (2, 2)


def test_enable_gradient_checkpointing_does_not_break_forward() -> None:
    from llm_from_scratch.models.configs import GPTConfig
    from llm_from_scratch.models.gpt import TinyGPT
    from llm_from_scratch.training.efficiency import enable_gradient_checkpointing

    cfg = GPTConfig(vocab_size=32, block_size=8, embedding_dim=16, num_layers=2, num_heads=4, dropout=0.0)
    m = TinyGPT(cfg)
    enable_gradient_checkpointing(m)
    x = torch.randint(0, 32, (1, 8))
    y = torch.randint(0, 32, (1, 8))
    logits, loss = m(x, y)
    assert logits.shape == (1, 8, 32)
    loss.backward()
