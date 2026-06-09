"""Tests for Chapter 20 KV cache and inference engine."""

import torch

from lm_from_scratch.inference.engine import InferenceEngine
from lm_from_scratch.inference.kv_cache import KVCache
from lm_from_scratch.models.configs import GPTConfig
from lm_from_scratch.models.gpt import TinyGPT


def _model():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=32, block_size=16, embedding_dim=16,
                    num_layers=2, num_heads=4, dropout=0.0)
    return TinyGPT(cfg)


def test_kv_cache_empty_state() -> None:
    cache = KVCache.empty(num_layers=3)
    assert cache.length() == 0
    assert cache.memory_bytes() == 0


def test_cached_generation_matches_greedy() -> None:
    m = _model().eval()
    prompt = [1, 2, 3]
    # Naive greedy
    ids_naive = list(prompt)
    for _ in range(5):
        x = torch.tensor(ids_naive, dtype=torch.long).unsqueeze(0)
        logits, _ = m(x)
        ids_naive.append(int(logits[0, -1].argmax().item()))

    # Cached greedy
    cache = KVCache.empty(num_layers=len(m.blocks))
    x = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)
    logits, cache = m.forward_cached(x, kv_cache=cache, position_offset=0)
    ids_cached = list(prompt)
    for _ in range(5):
        nt = int(logits[0, -1].argmax().item())
        ids_cached.append(nt)
        x = torch.tensor([[nt]], dtype=torch.long)
        logits, cache = m.forward_cached(x, kv_cache=cache, position_offset=cache.length())
    assert ids_naive == ids_cached


def test_engine_generates_text() -> None:
    class _Tok:
        def encode(self, s): return [1, 2, 3]
        def decode(self, ids): return " ".join(str(i) for i in ids)

    eng = InferenceEngine(_model(), _Tok())
    out = eng.generate("hi", max_new_tokens=4, temperature=0.0001)
    assert isinstance(out, str)
    assert len(out) > 0


def test_kv_cache_length_grows() -> None:
    m = _model().eval()
    cache = KVCache.empty(num_layers=len(m.blocks))
    x = torch.tensor([[1, 2, 3]], dtype=torch.long)
    _, cache = m.forward_cached(x, kv_cache=cache, position_offset=0)
    assert cache.length() == 3
    x2 = torch.tensor([[4]], dtype=torch.long)
    _, cache = m.forward_cached(x2, kv_cache=cache, position_offset=3)
    assert cache.length() == 4
