"""Tests for Chapter 13 TinyGPT and checkpointing."""

from pathlib import Path

import pytest
import torch

from llm_from_scratch.models.configs import GPTConfig
from llm_from_scratch.models.gpt import TinyGPT
from llm_from_scratch.training.checkpointing import load_checkpoint, save_checkpoint


def _small_config(**overrides) -> GPTConfig:
    base = dict(
        vocab_size=64,
        block_size=16,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.0,
    )
    base.update(overrides)
    return GPTConfig(**base)


def test_tinygpt_forward_shape() -> None:
    m = TinyGPT(_small_config())
    x = torch.randint(0, 64, (2, 10))
    logits, loss = m(x)
    assert logits.shape == (2, 10, 64)
    assert loss is None


def test_tinygpt_forward_with_loss() -> None:
    m = TinyGPT(_small_config())
    x = torch.randint(0, 64, (2, 10))
    y = torch.randint(0, 64, (2, 10))
    logits, loss = m(x, y)
    assert logits.shape == (2, 10, 64)
    assert loss.dim() == 0


def test_tinygpt_tied_embeddings_share_weight() -> None:
    m = TinyGPT(_small_config(tie_embeddings=True))
    assert m.lm_head.weight.data_ptr() == m.token_embedding.weight.data_ptr()


def test_tinygpt_untied_embeddings_have_distinct_weight() -> None:
    m = TinyGPT(_small_config(tie_embeddings=False))
    assert m.lm_head.weight.data_ptr() != m.token_embedding.weight.data_ptr()


def test_tinygpt_param_count_drops_when_tied() -> None:
    tied = TinyGPT(_small_config(tie_embeddings=True)).num_parameters()
    untied = TinyGPT(_small_config(tie_embeddings=False)).num_parameters()
    assert tied < untied


def test_tinygpt_rejects_too_long_input() -> None:
    m = TinyGPT(_small_config(block_size=8))
    x = torch.randint(0, 64, (1, 10))
    with pytest.raises(ValueError):
        m(x)


def test_tinygpt_generate_extends_prompt() -> None:
    torch.manual_seed(0)
    m = TinyGPT(_small_config())
    out = m.generate([1, 2, 3], max_new_tokens=5, temperature=1.0)
    assert len(out) == 3 + 5
    assert out[:3] == [1, 2, 3]


def test_tinygpt_save_load_round_trip(tmp_path: Path) -> None:
    m1 = TinyGPT(_small_config())
    opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    save_checkpoint(tmp_path / "ckpt.pt", m1, opt1, step=42, extras={"note": "smoke"})
    m2 = TinyGPT(_small_config())
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    payload = load_checkpoint(tmp_path / "ckpt.pt", m2, opt2)
    assert payload["step"] == 42
    assert payload["extras"]["note"] == "smoke"
    # Same forward output before/after load.
    x = torch.randint(0, 64, (1, 8))
    out1, _ = m1(x)
    out2, _ = m2(x)
    assert torch.allclose(out1, out2)
