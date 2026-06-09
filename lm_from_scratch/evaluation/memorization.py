"""Memorization probes: verbatim continuation and canary check."""

from __future__ import annotations

import torch

from lm_from_scratch.generation.decoding import greedy_decode


@torch.no_grad()
def verbatim_match_rate(
    model: torch.nn.Module,
    tokenizer,
    pairs: list[tuple[str, str]],
    *,
    prefix_tokens: int = 50,
    continuation_tokens: int = 50,
) -> float:
    """Fraction of (prefix, target) where greedy decoding from prefix matches target exactly."""
    matches = 0
    for prefix, target in pairs:
        p_ids = tokenizer.encode(prefix)[:prefix_tokens]
        out = greedy_decode(model, p_ids, continuation_tokens)
        continuation_ids = out[len(p_ids) :]
        decoded = tokenizer.decode(continuation_ids)
        if decoded.startswith(target):
            matches += 1
    return matches / max(1, len(pairs))


@torch.no_grad()
def canary_log_prob(model, tokenizer, canary: str) -> float:
    """Return the model's average log-probability per token on the canary string."""
    ids = tokenizer.encode(canary)
    if len(ids) < 2:
        return 0.0
    device = next(model.parameters()).device
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]
    log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
    targets = torch.tensor(ids[1:], device=device)
    return float(log_probs.gather(1, targets.unsqueeze(1)).mean().item())
