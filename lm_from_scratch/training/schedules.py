"""Learning-rate schedules."""

from __future__ import annotations

import math


def linear_warmup(step: int, warmup_steps: int) -> float:
    """Linear warm-up: 0 at step 0, 1 at step ``warmup_steps``, clipped above."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def cosine_decay(step: int, total_steps: int, min_lr_ratio: float = 0.1) -> float:
    """Cosine decay from 1.0 to ``min_lr_ratio`` over ``total_steps``."""
    if total_steps <= 0:
        return 1.0
    t = min(step, total_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * t / total_steps))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine


def warmup_cosine(
    step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1
) -> float:
    """Linear warm-up followed by cosine decay to ``min_lr_ratio``."""
    if step < warmup_steps:
        return linear_warmup(step, warmup_steps)
    return cosine_decay(step - warmup_steps, total_steps - warmup_steps, min_lr_ratio)
