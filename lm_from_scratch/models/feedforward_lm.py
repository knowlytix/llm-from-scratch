"""Feedforward neural language model (Bengio et al. 2003).

Concatenates the embeddings of the last ``context_size`` tokens and feeds
them through an MLP that produces vocabulary logits. The first model in
the book that uses more than one preceding token; still no recurrence, no
attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lm_from_scratch.training.losses import cross_entropy_loss


class FeedForwardLM(nn.Module):
    """Concatenate-and-MLP language model.

    Parameters
    ----------
    vocab_size:
        Tokenizer vocabulary size.
    embedding_dim:
        Width of each token embedding.
    context_size:
        Number of preceding tokens whose embeddings are concatenated.
    hidden_dim:
        Width of the hidden MLP layer.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        context_size: int = 8,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Predict next-token logits at every position with a left-pad strategy.

        ``input_ids`` has shape ``(B, T)``. We return logits of shape
        ``(B, T, V)``; at position ``t`` the input is the embeddings of
        positions ``[t - context_size + 1, ..., t]``, with positions
        before zero left-padded with the embedding of token 0.
        """
        B, T = input_ids.shape
        C = self.context_size
        d = self.embedding_dim
        # Left-pad with the embedding of id 0 (pad token).
        padded = torch.cat(
            [torch.zeros((B, C - 1), dtype=torch.long, device=input_ids.device), input_ids],
            dim=1,
        )
        # Build sliding contexts: (B, T, C)
        ctxs = padded.unfold(1, C, 1)  # (B, T, C)
        embs = self.embedding(ctxs)  # (B, T, C, d)
        flat = embs.reshape(B, T, C * d)
        h = self.activation(self.fc1(flat))
        logits = self.fc2(h)
        return logits

    def loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        return cross_entropy_loss(self.forward(input_ids), target_ids)
