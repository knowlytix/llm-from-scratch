"""Decoding strategies: greedy and sampling. Beam search included as a teaching reference."""

from __future__ import annotations

import torch

from lm_from_scratch.generation.sampling import sample_next_token


@torch.no_grad()
def greedy_decode(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    stop_token_ids: list[int] | None = None,
) -> list[int]:
    device = next(model.parameters()).device
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    out = list(prompt_ids)
    for _ in range(max_new_tokens):
        logits, _ = model(ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        next_id = int(logits[0, -1].argmax().item())
        out.append(next_id)
        if stop_token_ids and next_id in stop_token_ids:
            break
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return out


@torch.no_grad()
def sample_decode(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    stop_token_ids: list[int] | None = None,
) -> list[int]:
    device = next(model.parameters()).device
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    out = list(prompt_ids)
    for _ in range(max_new_tokens):
        logits, _ = model(ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        next_id = sample_next_token(
            logits[0, -1],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            past_tokens=out,
        )
        out.append(next_id)
        if stop_token_ids and next_id in stop_token_ids:
            break
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return out


@torch.no_grad()
def beam_search(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    num_beams: int = 4,
    length_penalty: float = 1.0,
) -> list[list[int]]:
    """Greedy beam search. Returns ``num_beams`` candidate sequences sorted by score."""
    device = next(model.parameters()).device
    # Each beam: (score, ids)
    beams = [(0.0, list(prompt_ids))]
    for _ in range(max_new_tokens):
        candidates = []
        for score, ids in beams:
            input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)
            topk_vals, topk_idx = torch.topk(log_probs, k=num_beams)
            for v, i in zip(topk_vals.tolist(), topk_idx.tolist()):
                candidates.append((score + v, ids + [i]))
        # Keep top-``num_beams`` by length-normalized score.
        candidates.sort(key=lambda x: x[0] / max(len(x[1]), 1) ** length_penalty, reverse=True)
        beams = candidates[:num_beams]
    beams.sort(key=lambda x: x[0] / max(len(x[1]), 1) ** length_penalty, reverse=True)
    return [ids for _, ids in beams]
