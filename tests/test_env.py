"""Tests for llm_from_scratch.utils.env."""

import numpy as np
import torch

from llm_from_scratch.utils.env import check_environment, set_seed


def test_check_environment_has_expected_keys() -> None:
    env = check_environment()
    expected_keys = {
        "python_version",
        "platform",
        "torch_version",
        "cuda_available",
        "cuda_version",
        "device_name",
        "num_cuda_devices",
    }
    assert expected_keys.issubset(env.keys())


def test_check_environment_types() -> None:
    env = check_environment()
    assert isinstance(env["python_version"], str)
    assert isinstance(env["torch_version"], str)
    assert isinstance(env["cuda_available"], bool)
    assert isinstance(env["num_cuda_devices"], int)


def test_set_seed_makes_torch_reproducible() -> None:
    set_seed(0)
    a = torch.randn(8)
    set_seed(0)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_set_seed_makes_numpy_reproducible() -> None:
    set_seed(123)
    a = np.random.randn(8)
    set_seed(123)
    b = np.random.randn(8)
    assert np.array_equal(a, b)


def test_set_seed_makes_random_reproducible() -> None:
    import random as pyrandom

    set_seed(7)
    a = [pyrandom.random() for _ in range(8)]
    set_seed(7)
    b = [pyrandom.random() for _ in range(8)]
    assert a == b
