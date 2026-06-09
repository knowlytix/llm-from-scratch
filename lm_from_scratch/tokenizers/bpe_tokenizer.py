"""Byte-pair encoding tokenizer, built from scratch.

The implementation follows the classical BPE recipe:

1. Pre-tokenize text into words using a GPT-2-style regex pattern.
2. Initialize the vocabulary as the base alphabet (bytes 0--255) plus a
   small set of special tokens.
3. While the vocabulary is smaller than ``vocab_size``, find the most
   frequent adjacent token pair across the pre-tokenized corpus, add a
   new vocabulary entry for the merged pair and apply the merge.
4. Save the ordered list of merges; encoding text means greedily applying
   the merges in their training order.

This implementation is byte-level: it operates on bytes rather than
characters, so it handles arbitrary UTF-8 input.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from lm_from_scratch.tokenizers.base import SPECIAL_TOKENS, BaseTokenizer
from lm_from_scratch.tokenizers.bpe_internals import get_pair_counts, merge_pair

#: GPT-2 style pre-tokenization regex. Splits on word boundaries while keeping
#: leading whitespace attached to the following word.
_GPT2_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+"
)


def _pre_tokenize(text: str) -> list[bytes]:
    return [m.encode("utf-8") for m in _GPT2_PATTERN.findall(text)]


_BYTE_OFFSET = max(SPECIAL_TOKENS.values()) + 1  # bytes start at id 4


class BPETokenizer(BaseTokenizer):
    """Byte-level BPE tokenizer.

    Parameters
    ----------
    vocab_size:
        Target final vocabulary size, including specials and the 256 base
        bytes. Must be at least ``len(SPECIAL_TOKENS) + 256``.
    """

    def __init__(self, vocab_size: int = 1024) -> None:
        min_size = _BYTE_OFFSET + 256
        if vocab_size < min_size:
            raise ValueError(
                f"vocab_size must be at least {min_size} (specials + 256 bytes), got {vocab_size}"
            )
        self._target_vocab_size = vocab_size
        # The token table maps token id -> bytes representation.
        self._token_to_bytes: dict[int, bytes] = {}
        # Specials are bytes-less and decoded specially.
        for sp, sp_id in SPECIAL_TOKENS.items():
            self._token_to_bytes[sp_id] = sp.encode("utf-8")
        for b in range(256):
            self._token_to_bytes[_BYTE_OFFSET + b] = bytes([b])
        # Ordered list of merges (pair -> new id) for use at encode time.
        self._merges: list[tuple[tuple[int, int], int]] = []

    # --- Required API --------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_bytes)

    @property
    def merges(self) -> list[tuple[tuple[int, int], int]]:
        return list(self._merges)

    def train(self, texts: Iterable[str]) -> None:
        """Run the BPE training loop until the vocabulary reaches ``vocab_size``."""
        # Pre-tokenize all texts; encode each word as a list of byte-token ids.
        pre_tokens: list[list[int]] = []
        for t in texts:
            for word_bytes in _pre_tokenize(t):
                pre_tokens.append([_BYTE_OFFSET + b for b in word_bytes])

        target = self._target_vocab_size
        next_id = max(self._token_to_bytes) + 1
        while len(self._token_to_bytes) < target:
            counts = get_pair_counts(pre_tokens)
            if not counts:
                break
            (a, b), _ = max(counts.items(), key=lambda kv: kv[1])
            self._token_to_bytes[next_id] = self._token_to_bytes[a] + self._token_to_bytes[b]
            self._merges.append(((a, b), next_id))
            pre_tokens = merge_pair(pre_tokens, (a, b), next_id)
            next_id += 1

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids using the learned merges."""
        ids_list: list[int] = []
        for word_bytes in _pre_tokenize(text):
            piece = [_BYTE_OFFSET + b for b in word_bytes]
            # Apply merges in their training order; this is the standard
            # greedy BPE encoding. A more efficient implementation uses a
            # priority queue or a regex-based merger.
            for pair, new_id in self._merges:
                piece = self._apply_merge(piece, pair, new_id)
            ids_list.extend(piece)
        return ids_list

    @staticmethod
    def _apply_merge(piece: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        if len(piece) < 2:
            return piece
        a, b = pair
        if a not in piece:
            return piece
        out: list[int] = []
        i = 0
        while i < len(piece):
            if i < len(piece) - 1 and piece[i] == a and piece[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(piece[i])
                i += 1
        return out

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text."""
        byts = bytearray()
        for i in ids:
            if i in SPECIAL_TOKENS.values():
                # Specials do not contribute readable bytes.
                continue
            byts.extend(self._token_to_bytes.get(i, b""))
        return byts.decode("utf-8", errors="replace")

    # --- Persistence ---------------------------------------------------

    def save(self, path: str | Path) -> None:
        self._save_json(
            path,
            {
                "type": "BPETokenizer",
                "target_vocab_size": self._target_vocab_size,
                # Tokens stored as a parallel list of (id, hex-bytes) pairs.
                "tokens": [(i, b.hex()) for i, b in self._token_to_bytes.items()],
                "merges": [[a, b, new_id] for (a, b), new_id in self._merges],
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = cls._load_json(path)
        if data.get("type") != "BPETokenizer":
            raise ValueError(f"Not a BPETokenizer file: {path}")
        obj = cls(vocab_size=int(data["target_vocab_size"]))
        obj._token_to_bytes = {int(i): bytes.fromhex(h) for i, h in data["tokens"]}
        obj._merges = [((int(a), int(b)), int(new_id)) for a, b, new_id in data["merges"]]
        return obj
