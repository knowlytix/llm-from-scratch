"""Bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(scores: list[float] | np.ndarray, n_resamples: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, low, high) bootstrap CI."""
    arr = np.asarray(scores, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(0)
    means = []
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(sample.mean())
    means = np.array(means)
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), low, high


def paired_bootstrap(a: list[float] | np.ndarray, b: list[float] | np.ndarray,
                     n_resamples: int = 2000) -> dict[str, float]:
    """Paired-sample bootstrap of (a - b)."""
    A = np.asarray(a, dtype=float); B = np.asarray(b, dtype=float)
    diffs = A - B
    mean, low, high = bootstrap_ci(diffs, n_resamples=n_resamples)
    return {"mean_diff": mean, "ci_low": low, "ci_high": high}
