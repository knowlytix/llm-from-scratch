"""Tests for Chapter 26 interpretability."""

import numpy as np
import torch

from llm_from_scratch.interpretability.embeddings import embedding_2d
from llm_from_scratch.interpretability.probes import LinearProbe
from llm_from_scratch.models.configs import GPTConfig
from llm_from_scratch.models.gpt import TinyGPT


def test_embedding_2d_returns_2d() -> None:
    m = TinyGPT(GPTConfig(vocab_size=32, block_size=8, embedding_dim=16, num_layers=2, num_heads=4, dropout=0.0))
    coords = embedding_2d(m, list(range(10)))
    assert coords.shape == (10, 2)


def test_linear_probe_accuracy_above_chance() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    y = (X[:, 0] > 0).astype(int)
    p = LinearProbe()
    p.fit(X, y)
    assert p.score(X, y) > 0.9
