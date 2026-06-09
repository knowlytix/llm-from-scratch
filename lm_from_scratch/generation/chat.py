"""Convenience: text in, text out."""

from __future__ import annotations

from lm_from_scratch.generation.decoding import greedy_decode, sample_decode


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 100, **sampling_kwargs) -> str:
    ids = tokenizer.encode(prompt)
    if sampling_kwargs.get("temperature", 1.0) == 0.0:
        out = greedy_decode(model, ids, max_new_tokens)
    else:
        out = sample_decode(model, ids, max_new_tokens, **sampling_kwargs)
    return tokenizer.decode(out)
