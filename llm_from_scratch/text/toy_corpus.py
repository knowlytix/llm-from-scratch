"""Toy corpora used to make the first few chapters concrete.

The book uses Tiny Shakespeare as its running foundational example. This module
loads the corpus from a local cache or downloads it on first use.
"""

from __future__ import annotations

import urllib.request
from collections import Counter
from pathlib import Path

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
DEFAULT_CACHE = Path("data/raw/tiny_shakespeare.txt")


def load_tiny_shakespeare(path: str | Path | None = None) -> str:
    """Return the Tiny Shakespeare corpus as a single string.

    The corpus is cached on disk at ``path`` (defaulting to
    ``data/raw/tiny_shakespeare.txt`` relative to the current working
    directory). If the cache does not exist the file is downloaded.
    """
    cache_path = Path(path) if path is not None else DEFAULT_CACHE
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, cache_path)
    return cache_path.read_text(encoding="utf-8")


def next_token_counts(text: str) -> dict[tuple[str, str], int]:
    """Count ``(context_char, next_char)`` pairs across the text.

    This is the empirical bigram count table over characters. It is the
    simplest non-trivial language model: each pair contributes one count to
    the table, and the conditional distribution of the next character given
    the previous character is the row of the table corresponding to the
    previous character, normalized.

    The function returns a plain ``dict`` keyed by tuples to keep the data
    structure obvious. Real n-gram code in Chapter~5 uses a more efficient
    representation.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for a, b in zip(text, text[1:]):
        counts[(a, b)] += 1
    return dict(counts)
