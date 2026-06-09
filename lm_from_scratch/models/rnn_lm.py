"""Recurrent language models built on PyTorch's `nn.RNN`/`nn.GRU`/`nn.LSTM`.

The educational point of this module is the recurrence equation and how
the hidden state passes information forward. The PyTorch primitives
provide a fast and correct implementation of the recurrence; a hand-rolled
GRU cell in `rnn_from_scratch.py` makes the gate equations concrete.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from lm_from_scratch.training.losses import cross_entropy_loss


RNNType = Literal["rnn", "gru", "lstm"]


class RNNLanguageModel(nn.Module):
    """Embedding -> RNN -> linear -> logits.

    Parameters
    ----------
    vocab_size:
        Tokenizer vocabulary size.
    embedding_dim:
        Width of token embeddings.
    hidden_dim:
        Width of the RNN hidden state.
    num_layers:
        Number of stacked RNN layers.
    rnn_type:
        Which recurrence to use: ``"rnn"``, ``"gru"`` or ``"lstm"``.
    dropout:
        Dropout between RNN layers (no effect with one layer).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        rnn_type: RNNType = "gru",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[rnn_type]
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding(input_ids)
        out, new_hidden = self.rnn(x, hidden)
        logits = self.lm_head(out)
        return logits, new_hidden

    def loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(input_ids)
        return cross_entropy_loss(logits, target_ids)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int] | torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        device: str | torch.device | None = None,
    ) -> list[int]:
        device = device or next(self.parameters()).device
        if not isinstance(prompt_ids, torch.Tensor):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        ids = prompt_ids.to(device).unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids.to(device)
        out: list[int] = ids.squeeze(0).tolist()
        # Run the prompt once to seed the hidden state, then feed each freshly
        # sampled token in turn. Feeding the prompt and the last prompt token
        # in separate calls would consume that token twice.
        hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None
        next_input = ids
        for _ in range(max_new_tokens):
            logits, hidden = self.forward(next_input, hidden)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(-1, topk_idx, topk_vals)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            out.append(int(next_id.item()))
            next_input = next_id
        return out
