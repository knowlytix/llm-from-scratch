"""TinyGPT: a small causal Transformer language model.

The model is the smallest assembly of every part built in Chapters 8-12:
token embeddings, positional embeddings, a stack of ``TransformerBlock``s,
a final norm and a language-modeling head. With ``tie_embeddings=True``
the head shares weights with the token embedding to save parameters.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from llm_from_scratch.models.configs import GPTConfig
from llm_from_scratch.models.positional import (
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from llm_from_scratch.models.transformer_block import TransformerBlock
from llm_from_scratch.training.losses import cross_entropy_loss


class TinyGPT(nn.Module):
    """A small causal Transformer language model.

    Parameters
    ----------
    config:
        :class:`GPTConfig` describing the architecture.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        if config.positional == "learned":
            self.position_embedding: nn.Module = LearnedPositionalEmbedding(
                config.block_size, config.embedding_dim
            )
        elif config.positional == "sinusoidal":
            self.position_embedding = SinusoidalPositionalEmbedding(
                config.embedding_dim, max_len=config.block_size
            )
        else:
            raise ValueError(
                f"TinyGPT supports learned and sinusoidal positions in this chapter; "
                f"got {config.positional}"
            )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=config.embedding_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    block_size=config.block_size,
                    norm_style=config.norm_style,
                    activation=config.activation,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self._init_weights()

    # --- Initialization ----------------------------------------------------

    def _init_weights(self) -> None:
        std = self.config.init_std
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)
        # GPT-2 trick: scale the residual-output projections by 1/sqrt(2 * num_layers).
        scale = 1.0 / math.sqrt(2 * self.config.num_layers)
        for block in self.blocks:
            with torch.no_grad():
                block.attn.out_proj.weight.mul_(scale)
                if hasattr(block.mlp, "fc2"):
                    block.mlp.fc2.weight.mul_(scale)
                elif hasattr(block.mlp, "down_proj"):
                    block.mlp.down_proj.weight.mul_(scale)

    # --- Forward and loss --------------------------------------------------

    def forward(
        self, input_ids: torch.Tensor, target_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        T = input_ids.size(1)
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.config.block_size}")
        x = self.token_embedding(input_ids)
        x = self.position_embedding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)
        logits = self.lm_head(x)
        loss = None if target_ids is None else cross_entropy_loss(logits, target_ids)
        return logits, loss

    @torch.no_grad()
    def forward_cached(
        self,
        input_ids: torch.Tensor,
        kv_cache=None,
        position_offset: int = 0,
    ):
        """Forward with a KV cache.

        ``input_ids`` is the new tokens (shape ``(B, T_new)``); ``kv_cache``
        is a ``KVCache`` from ``llm_from_scratch.inference.kv_cache`` or
        ``None`` to start a fresh cache. ``position_offset`` is how many
        tokens are already in the cache (used to look up position
        embeddings for the new tokens).
        """
        from llm_from_scratch.inference.kv_cache import KVCache

        T_new = input_ids.size(1)
        # Build positions for the new tokens only.
        positions = torch.arange(
            position_offset, position_offset + T_new, device=input_ids.device
        )
        if hasattr(self.position_embedding, "table"):
            pos_emb = self.position_embedding.table(positions)
        else:
            # Sinusoidal positional embedding lookup.
            pos_emb = self.position_embedding.pe[positions]
        x = self.token_embedding(input_ids) + pos_emb
        if kv_cache is None:
            kv_cache = KVCache.empty(len(self.blocks))
        for i, block in enumerate(self.blocks):
            past_k = kv_cache.keys[i] if kv_cache.keys[i] is not None else None
            past_v = kv_cache.values[i] if kv_cache.values[i] is not None else None
            x, new_full_k, new_full_v = block.forward_with_cache(x, past_k, past_v)
            kv_cache.keys[i] = new_full_k
            kv_cache.values[i] = new_full_v
        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits, kv_cache

    def loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids, target_ids)[1]  # type: ignore[return-value]

    # --- Generation --------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int] | torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        device: str | torch.device | None = None,
    ) -> list[int]:
        device = device or next(self.parameters()).device
        if not isinstance(prompt_ids, torch.Tensor):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        ids = prompt_ids.to(device).unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids.to(device)
        out: list[int] = ids.squeeze(0).tolist()
        for _ in range(max_new_tokens):
            x = ids[:, -self.config.block_size :]
            logits, _ = self.forward(x)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                vals, idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                clipped = torch.full_like(logits, float("-inf"))
                clipped.scatter_(-1, idx, vals)
                logits = clipped
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                # Remove tokens past the nucleus, but keep the first token whose
                # cumulative probability crosses ``top_p`` so the kept mass is
                # always at least ``top_p``. Shift right by one before masking.
                mask = cum > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(-1, sorted_idx, sorted_logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            out.append(int(next_id.item()))
            ids = torch.cat([ids, next_id], dim=1)
        return out

    # --- Misc --------------------------------------------------------------

    def num_parameters(self, exclude_embedding: bool = False) -> int:
        if not exclude_embedding:
            return sum(p.numel() for p in self.parameters())
        emb = self.token_embedding.weight.numel()
        if not self.config.tie_embeddings:
            emb += self.lm_head.weight.numel()
        return sum(p.numel() for p in self.parameters()) - emb
