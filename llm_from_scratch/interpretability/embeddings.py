"""Embedding-based interpretability."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA


@torch.no_grad()
def nearest_neighbors(model, tokenizer, token: str, k: int = 10) -> list[tuple[str, float]]:
    """Return the ``k`` nearest tokens to ``token`` by cosine in the embedding space."""
    ids = tokenizer.encode(token)
    if not ids:
        return []
    target = ids[0]
    E = model.token_embedding.weight.detach().cpu().numpy()
    v = E[target]
    norms = np.linalg.norm(E, axis=1)
    sims = (E @ v) / (norms * np.linalg.norm(v) + 1e-12)
    top = np.argsort(-sims)[: k + 1]
    out = []
    for idx in top:
        if int(idx) == target:
            continue
        out.append((tokenizer.decode([int(idx)]), float(sims[idx])))
        if len(out) >= k:
            break
    return out


@torch.no_grad()
def embedding_2d(model, ids: list[int]) -> np.ndarray:
    E = model.token_embedding.weight.detach().cpu().numpy()
    vecs = E[ids]
    return PCA(n_components=2).fit_transform(vecs)
