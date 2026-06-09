"""Tokenizer diagnostics.

Three numbers describe a tokenizer well enough for first-pass inspection:
compression ratio, average tokens per word and out-of-vocabulary rate. The
helpers in this module compute each on a held-out sample of text.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from llm_from_scratch.tokenizers.base import BaseTokenizer


def compression_ratio(tokenizer: BaseTokenizer, texts: Iterable[str]) -> float:
    """Return mean characters per token across all input texts."""
    total_chars = 0
    total_tokens = 0
    for t in texts:
        total_chars += len(t)
        total_tokens += len(tokenizer.encode(t))
    if total_tokens == 0:
        return 0.0
    return total_chars / total_tokens


def tokens_per_word(tokenizer: BaseTokenizer, texts: Iterable[str]) -> float:
    """Return mean tokens per whitespace-delimited word."""
    total_tokens = 0
    total_words = 0
    for t in texts:
        total_tokens += len(tokenizer.encode(t))
        total_words += len(t.split())
    if total_words == 0:
        return 0.0
    return total_tokens / total_words


def unknown_token_rate(tokenizer: BaseTokenizer, texts: Iterable[str]) -> float:
    """Fraction of encoded tokens equal to ``<unk>``."""
    unk = tokenizer.unk_id
    total = 0
    unk_count = 0
    for t in texts:
        ids = tokenizer.encode(t)
        total += len(ids)
        unk_count += sum(1 for i in ids if i == unk)
    if total == 0:
        return 0.0
    return unk_count / total


def tokenizer_report(tokenizer: BaseTokenizer, texts: Iterable[str]) -> dict[str, Any]:
    """A compact report combining the three primary diagnostics."""
    texts_list = list(texts)  # ensure we can iterate multiple times
    return {
        "type": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "compression_ratio": compression_ratio(tokenizer, texts_list),
        "tokens_per_word": tokens_per_word(tokenizer, texts_list),
        "unknown_token_rate": unknown_token_rate(tokenizer, texts_list),
    }
