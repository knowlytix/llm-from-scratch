"""Tests for Chapter 21 intrinsic evaluation."""

import math

import numpy as np

from llm_from_scratch.evaluation.calibration import (
    expected_calibration_error,
    reliability_diagram,
)


def test_ece_zero_on_perfect_calibration() -> None:
    # If predictions match probabilities exactly, ECE = 0.
    probs = np.array([0.5] * 100)
    correct = np.array([1] * 50 + [0] * 50)
    ece = expected_calibration_error(probs, correct, num_bins=10)
    assert ece < 0.05


def test_ece_high_on_overconfident_predictions() -> None:
    # All predictions at high confidence but only 50% correct: high ECE.
    probs = np.array([0.95] * 100)
    correct = np.array([1] * 50 + [0] * 50)
    ece = expected_calibration_error(probs, correct, num_bins=10)
    assert ece > 0.4


def test_reliability_diagram_returns_arrays() -> None:
    probs = np.random.rand(200)
    correct = (np.random.rand(200) < probs).astype(int)
    confs, accs = reliability_diagram(probs, correct, num_bins=10)
    assert confs.shape == (10,)
    assert accs.shape == (10,)
