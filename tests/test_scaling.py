"""Tests for Chapter 17 scaling utilities."""

import math

from lm_from_scratch.models.configs import GPTConfig
from lm_from_scratch.models.gpt import TinyGPT
from lm_from_scratch.utils.scaling import (
    chinchilla_optimal,
    count_parameters,
    inference_flops,
    memory_footprint,
    time_estimate,
    training_flops,
)


def test_count_parameters_matches_actual_within_5_percent() -> None:
    cfg = GPTConfig(vocab_size=256, block_size=64, embedding_dim=64, num_layers=2,
                    num_heads=4, mlp_ratio=4, dropout=0.0)
    estimate = count_parameters(cfg)
    actual = TinyGPT(cfg).num_parameters()
    assert abs(estimate - actual) / actual < 0.05


def test_flops_formulas() -> None:
    N, D = 1_000, 10_000
    assert training_flops(N, D) == 6 * N * D
    assert inference_flops(N, D) == 2 * N * D


def test_chinchilla_uses_20x_rule() -> None:
    out = chinchilla_optimal(target_params=1_000_000)
    assert math.isclose(out["tokens"], 20_000_000, rel_tol=1e-9)


def test_chinchilla_from_compute() -> None:
    C = 6e15
    out = chinchilla_optimal(compute_budget=C)
    assert math.isclose(out["compute"], C, rel_tol=1e-9)


def test_memory_footprint_has_expected_keys() -> None:
    cfg = GPTConfig(vocab_size=1024, block_size=128, embedding_dim=128, num_layers=4, num_heads=4)
    mem = memory_footprint(cfg, batch_size=32, dtype="bf16")
    expected = {"params_bytes", "grads_bytes", "optimizer_bytes", "activations_bytes", "total_bytes"}
    assert expected.issubset(mem.keys())
    assert mem["total_bytes"] > 0


def test_time_estimate() -> None:
    assert time_estimate(1e12, 1e9) == 1000.0
