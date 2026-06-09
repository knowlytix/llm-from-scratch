"""Linear probes on model activations."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


class LinearProbe:
    """Logistic-regression probe wrapping scikit-learn."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.model.score(X, y))


@torch.no_grad()
def collect_activations(model, dataset, layer_idx: int, max_examples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Collect mean residual-stream activations at the output of ``layer_idx``
    and a label = sequence sum modulo some number, as a synthetic probe target.
    """
    device = next(model.parameters()).device
    acts: list[np.ndarray] = []
    labels: list[int] = []
    for i in range(min(max_examples, len(dataset))):
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)
        T = x.size(1)
        positions = torch.arange(T, device=device)
        h = model.token_embedding(x)
        if hasattr(model.position_embedding, "table"):
            h = h + model.position_embedding.table(positions)
        else:
            h = h + model.position_embedding.pe[positions]
        for j, block in enumerate(model.blocks):
            h = block(h)
            if j == layer_idx:
                break
        mean = h.mean(dim=1).cpu().numpy().squeeze(0)
        acts.append(mean)
        labels.append(int(x[0, 0].item() % 4))
    return np.array(acts), np.array(labels)
