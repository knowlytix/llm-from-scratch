"""Neural bigram language model.

The simplest learnable LM: an embedding table indexed by the previous
token, projected to vocabulary logits. It is the continuous relaxation of
a count-based bigram (Chapter 5) and the first model in the book that
trains by gradient descent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from llm_from_scratch.training.losses import cross_entropy_loss


class NeuralBigramLM(nn.Module):
    """Embedding -> linear -> logits.

    Parameters
    ----------
    vocab_size:
        Size of the tokenizer's vocabulary.
    embedding_dim:
        Width of the embedding vectors and of the hidden representation.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T) -> embeddings: (B, T, d) -> logits: (B, T, V)
        x = self.embedding(input_ids)
        return self.lm_head(x)

    def loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids)
        return cross_entropy_loss(logits, target_ids)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int] | torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        device: str | torch.device | None = None,
    ) -> list[int]:
        device = device or next(self.parameters()).device
        if not isinstance(prompt_ids, torch.Tensor):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        ids = prompt_ids.to(device).unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids.to(device)
        out: list[int] = ids.squeeze(0).tolist()
        for _ in range(max_new_tokens):
            logits = self.forward(ids[:, -1:])[:, -1, :]
            logits = logits / max(temperature, 1e-8)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            out.append(int(next_id.item()))
            ids = torch.cat([ids, next_id], dim=1)
        return out
