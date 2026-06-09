"""A simple inference engine that uses the KV cache for fast generation."""

from __future__ import annotations

import torch

from llm_from_scratch.generation.sampling import sample_next_token
from llm_from_scratch.inference.kv_cache import KVCache


class InferenceEngine:
    """Wraps a TinyGPT model and a tokenizer with KV-cached generation."""

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    @torch.no_grad()
    def prefill(self, prompt_ids: list[int]) -> tuple[KVCache, torch.Tensor]:
        device = next(self.model.parameters()).device
        ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        logits, cache = self.model.forward_cached(ids, kv_cache=None, position_offset=0)
        return cache, logits

    @torch.no_grad()
    def decode_step(
        self, cache: KVCache, last_token: int
    ) -> tuple[torch.Tensor, KVCache]:
        device = next(self.model.parameters()).device
        ids = torch.tensor([[last_token]], dtype=torch.long, device=device)
        position_offset = cache.length()
        logits, cache = self.model.forward_cached(
            ids, kv_cache=cache, position_offset=position_offset
        )
        return logits, cache

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        *,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = None,
    ) -> str:
        prompt_ids = self.tokenizer.encode(prompt)
        cache, logits = self.prefill(prompt_ids)
        out = list(prompt_ids)
        for _ in range(max_new_tokens):
            next_id = sample_next_token(
                logits[0, -1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            out.append(next_id)
            logits, cache = self.decode_step(cache, next_id)
        return self.tokenizer.decode(out)
