"""Whitespace-based word tokenizer.

A simple tokenizer that splits on whitespace and a few punctuation marks.
Every distinct word in the training corpus becomes a token; words unseen
at inference time map to ``<unk>``. The tokenizer is intentionally crude
to make the out-of-vocabulary problem visible in the notebook.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from lm_from_scratch.tokenizers.base import SPECIAL_TOKENS, BaseTokenizer

_SPLIT_RE = re.compile(r"\s+|([.,!?;:()\"'])")


def _tokenize(text: str) -> list[str]:
    """Split on whitespace and isolate common punctuation as separate tokens."""
    parts = _SPLIT_RE.split(text)
    return [p for p in parts if p and not p.isspace()]


class WhitespaceTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = dict(SPECIAL_TOKENS)
        self._id_to_token: dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def train(self, texts: Iterable[str]) -> None:
        seen: set[str] = set()
        for t in texts:
            seen.update(_tokenize(t))
        next_id = max(self._id_to_token) + 1
        for tok in sorted(seen):
            if tok in self._token_to_id:
                continue
            self._token_to_id[tok] = next_id
            self._id_to_token[next_id] = tok
            next_id += 1

    def encode(self, text: str) -> list[int]:
        unk = self.unk_id
        return [self._token_to_id.get(tok, unk) for tok in _tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        words: list[str] = []
        for i in ids:
            tok = self._id_to_token.get(i, "")
            if tok in SPECIAL_TOKENS:
                continue
            words.append(tok)
        return " ".join(words)

    def save(self, path: str | Path) -> None:
        self._save_json(
            path,
            {
                "type": "WhitespaceTokenizer",
                "vocab": list(self._token_to_id.keys()),
                "ids": list(self._token_to_id.values()),
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "WhitespaceTokenizer":
        data = cls._load_json(path)
        if data.get("type") != "WhitespaceTokenizer":
            raise ValueError(f"Not a WhitespaceTokenizer file: {path}")
        obj = cls()
        obj._token_to_id = dict(zip(data["vocab"], data["ids"]))
        obj._id_to_token = {v: k for k, v in obj._token_to_id.items()}
        return obj
