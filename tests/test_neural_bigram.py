"""Tests for the Chapter 6 datasets, neural bigram and feedforward LM."""

import torch
import pytest

from llm_from_scratch.datasets.collators import causal_lm_collator
from llm_from_scratch.datasets.sequence_dataset import CausalLMDataset
from llm_from_scratch.models.feedforward_lm import FeedForwardLM
from llm_from_scratch.models.neural_bigram import NeuralBigramLM
from llm_from_scratch.training.losses import cross_entropy_loss
from llm_from_scratch.training.simple_loop import simple_train
from llm_from_scratch.utils.env import set_seed


# --- CausalLMDataset ---------------------------------------------------


def test_causal_lm_dataset_x_y_shift_invariant() -> None:
    ds = CausalLMDataset([0, 1, 2, 3, 4, 5, 6, 7], block_size=3, stride=3)
    x0, y0 = ds[0]
    x1, y1 = ds[1]
    assert x0.tolist() == [0, 1, 2]
    assert y0.tolist() == [1, 2, 3]
    assert x1.tolist() == [3, 4, 5]
    assert y1.tolist() == [4, 5, 6]


def test_causal_lm_dataset_overlapping_windows() -> None:
    ds = CausalLMDataset([0, 1, 2, 3, 4, 5], block_size=3, stride=1)
    # Windows starting at 0, 1, 2: (need block+1 tokens after start)
    assert len(ds) == 3


def test_causal_lm_dataset_collator_stacks_correctly() -> None:
    ds = CausalLMDataset(list(range(20)), block_size=4, stride=4)
    batch = [ds[i] for i in range(3)]
    x, y = causal_lm_collator(batch)
    assert x.shape == (3, 4)
    assert y.shape == (3, 4)


def test_causal_lm_dataset_rejects_too_short_stream() -> None:
    with pytest.raises(ValueError):
        CausalLMDataset([1, 2], block_size=4)


# --- Cross-entropy -----------------------------------------------------


def test_cross_entropy_flatten_BTV() -> None:
    set_seed(0)
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor([[0, 1, 2], [3, 4, 0]])
    loss = cross_entropy_loss(logits, targets)
    assert loss.item() > 0


# --- Neural bigram -----------------------------------------------------


def test_neural_bigram_forward_shape() -> None:
    m = NeuralBigramLM(vocab_size=10, embedding_dim=8)
    x = torch.randint(0, 10, (2, 5))
    out = m(x)
    assert out.shape == (2, 5, 10)


def test_neural_bigram_overfits_tiny_batch() -> None:
    set_seed(0)
    torch.manual_seed(0)
    vocab = 7
    # Deterministic small sequence to overfit.
    ids = ([0, 1, 2, 3, 4, 5, 6, 0, 1, 2] * 20)
    ds = CausalLMDataset(ids, block_size=4, stride=2)
    m = NeuralBigramLM(vocab_size=vocab, embedding_dim=8)
    history = simple_train(m, ds, valid_dataset=None, max_steps=200, batch_size=16, lr=3e-3, eval_every=50)
    final = history.train_loss[-1]
    assert final < 1.0  # below uniform-prediction loss ln(7) ≈ 1.95


# --- Feedforward LM ----------------------------------------------------


def test_feedforward_lm_forward_shape() -> None:
    m = FeedForwardLM(vocab_size=10, embedding_dim=8, context_size=4, hidden_dim=16)
    x = torch.randint(0, 10, (2, 5))
    out = m(x)
    assert out.shape == (2, 5, 10)


def test_feedforward_lm_loss_returns_scalar() -> None:
    m = FeedForwardLM(vocab_size=10, embedding_dim=8, context_size=4, hidden_dim=16)
    x = torch.randint(0, 10, (3, 6))
    y = torch.randint(0, 10, (3, 6))
    loss = m.loss(x, y)
    assert loss.dim() == 0
