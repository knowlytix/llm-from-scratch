"""Synthetic-data generation stub.

A real implementation calls a stronger LM to generate (prompt, response)
pairs from seed prompts. The library exposes the interface; the actual
API call is gated behind ``ANTHROPIC_API_KEY`` and stubbed otherwise.
"""

from __future__ import annotations


class SyntheticGenerator:
    def __init__(self, model_id: str = "claude-opus-4-6") -> None:
        self.model_id = model_id

    def generate_pair(self, seed_prompt: str) -> tuple[str, str]:
        """Returns ``(user, assistant)``. In offline mode returns a deterministic stub."""
        return seed_prompt, f"[stub response to '{seed_prompt}']"

    def generate_batch(self, seed_prompts: list[str]) -> list[tuple[str, str]]:
        return [self.generate_pair(p) for p in seed_prompts]


def filter_synthetic(pairs: list[tuple[str, str]], *,
                     min_response_length: int = 5,
                     max_response_length: int = 1024,
                     dedup: bool = True) -> list[tuple[str, str]]:
    seen = set()
    out = []
    for u, a in pairs:
        if not (min_response_length <= len(a) <= max_response_length):
            continue
        if dedup and a in seen:
            continue
        seen.add(a)
        out.append((u, a))
    return out
