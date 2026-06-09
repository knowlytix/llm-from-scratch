"""Sampling primitives: filters and the single-token sampler."""

from __future__ import annotations

import torch


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out (set to -inf) all but the top-``k`` logits."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    topk_vals, _ = torch.topk(logits, k=k, dim=-1)
    threshold = topk_vals[..., -1, None]
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose probabilities sum to >= p."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    # Mark tokens past the nucleus, then shift right by one so the first token
    # whose cumulative probability crosses ``p`` is kept — without this shift
    # the kept set has total mass strictly less than ``p``.
    mask = cum > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(-1, sorted_idx, sorted_logits)
    return out


def apply_repetition_penalty(
    logits: torch.Tensor, past_tokens: list[int], penalty: float = 1.1
) -> torch.Tensor:
    """Push logits of previously-seen tokens toward ``-inf`` so they become less likely.

    With ``penalty > 1`` we divide positive logits by ``penalty`` and multiply
    negative ones by ``penalty``. Both moves push the logit further from zero
    in the negative direction, lowering the post-softmax probability. Naively
    always dividing would *raise* the probability of negative-logit tokens
    (since dividing brings them closer to zero), which is the opposite of a
    repetition penalty.
    """
    if penalty == 1.0 or not past_tokens:
        return logits
    out = logits.clone()
    ids = torch.tensor(list(set(past_tokens)), device=logits.device, dtype=torch.long)
    scores = out[..., ids]
    out[..., ids] = torch.where(scores >= 0, scores / penalty, scores * penalty)
    return out


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    past_tokens: list[int] | None = None,
) -> int:
    """Sample the next token id from logits with optional filters."""
    z = logits.detach().clone()
    if repetition_penalty is not None and past_tokens is not None:
        z = apply_repetition_penalty(z, past_tokens, repetition_penalty)
    z = z / max(temperature, 1e-8)
    if top_k is not None:
        z = top_k_filter(z, top_k)
    if top_p is not None:
        z = top_p_filter(z, top_p)
    probs = torch.softmax(z, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
