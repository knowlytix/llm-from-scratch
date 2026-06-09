"""Abstract base for tokenizers used throughout the book.

A tokenizer is a deterministic map from text to a list of integer ids and
back. Beyond that contract the implementations are free to differ. The four
concrete tokenizers in the book (character, whitespace, byte, BPE) all
subclass this base and pass the same round-trip tests.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


SPECIAL_TOKENS: dict[str, int] = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
}


class BaseTokenizer(abc.ABC):
    """Abstract tokenizer interface.

    Subclasses must implement :meth:`train`, :meth:`encode`, :meth:`decode`,
    :meth:`save` and :meth:`load`. The base class fills in batched variants
    and exposes the special-token ids.
    """

    #: Token-to-id mapping for the four special tokens used in the book.
    special_tokens: dict[str, int] = dict(SPECIAL_TOKENS)

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        """Total vocabulary size, including specials."""

    @abc.abstractmethod
    def train(self, texts: Iterable[str]) -> None:
        """Fit the tokenizer to a corpus."""

    @abc.abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text to a list of integer ids."""

    @abc.abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decode a list of integer ids to text."""

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode_batch(self, batches: Iterable[list[int]]) -> list[str]:
        return [self.decode(b) for b in batches]

    # --- Special-token accessors --------------------------------------

    @property
    def pad_id(self) -> int:
        return self.special_tokens["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.special_tokens["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.special_tokens["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.special_tokens["<eos>"]

    # --- Persistence ---------------------------------------------------

    @abc.abstractmethod
    def save(self, path: str | Path) -> None:
        """Serialize the tokenizer to disk."""

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str | Path) -> "BaseTokenizer":
        """Load a tokenizer previously saved with :meth:`save`."""

    # --- Helpers for subclasses ---------------------------------------

    def _save_json(self, path: str | Path, data: dict[str, Any]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _load_json(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
