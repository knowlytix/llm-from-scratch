"""Logit lens: apply the LM head to each block's output."""

from __future__ import annotations

import torch


@torch.no_grad()
def logit_lens(model, tokenizer, text: str, top_k: int = 5) -> list[list[tuple[str, float]]]:
    """For each block, return the top-k predicted tokens for the last position."""
    device = next(model.parameters()).device
    ids = tokenizer.encode(text)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    T = x.size(1)
    positions = torch.arange(T, device=device)
    h = model.token_embedding(x)
    if hasattr(model.position_embedding, "table"):
        h = h + model.position_embedding.table(positions)
    else:
        h = h + model.position_embedding.pe[positions]
    per_block = []
    for block in model.blocks:
        h = block(h)
        normed = model.norm_final(h)
        logits = model.lm_head(normed)[0, -1]
        topk_vals, topk_idx = torch.topk(torch.softmax(logits, dim=-1), k=top_k)
        per_block.append([
            (tokenizer.decode([int(i)]), float(p))
            for i, p in zip(topk_idx.tolist(), topk_vals.tolist())
        ])
    return per_block
