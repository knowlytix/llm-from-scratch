"""Byte-level tokenizer.

The universal-coverage tokenizer. Each UTF-8 byte is one token, plus the
four special tokens. The vocabulary is 260 ids; the encoder and decoder
never throw on unseen input.

This tokenizer requires no training. ``train`` is a no-op so the class
plugs into the same pipeline code as the others.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from llm_from_scratch.tokenizers.base import SPECIAL_TOKENS, BaseTokenizer

_OFFSET = max(SPECIAL_TOKENS.values()) + 1  # ids 0..3 are specials, bytes start at 4


class ByteTokenizer(BaseTokenizer):
    @property
    def vocab_size(self) -> int:
        return _OFFSET + 256

    def train(self, texts: Iterable[str]) -> None:
        # No training: the byte vocabulary is fixed.
        return None

    def encode(self, text: str) -> list[int]:
        return [_OFFSET + b for b in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        # Drop specials; reassemble bytes back into a string.
        byts = bytearray()
        for i in ids:
            if i < _OFFSET:
                continue
            byts.append(i - _OFFSET)
        # ``errors="replace"`` keeps decode total even if the id stream is corrupt.
        return byts.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        self._save_json(path, {"type": "ByteTokenizer", "offset": _OFFSET})

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        data = cls._load_json(path)
        if data.get("type") != "ByteTokenizer":
            raise ValueError(f"Not a ByteTokenizer file: {path}")
        return cls()
