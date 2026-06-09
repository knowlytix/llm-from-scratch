"""Character-level tokenizer.

The simplest possible tokenizer: every distinct character in the training
text becomes a token. Special tokens occupy ids 0..3. The vocabulary is
small (a few dozen for English) and there is no out-of-vocabulary problem
within the trained set; unseen characters at inference time map to
``<unk>``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from llm_from_scratch.tokenizers.base import SPECIAL_TOKENS, BaseTokenizer


class CharTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        self._char_to_id: dict[str, int] = dict(SPECIAL_TOKENS)
        self._id_to_char: dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}

    # --- Required API --------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._char_to_id)

    def train(self, texts: Iterable[str]) -> None:
        chars: set[str] = set()
        for t in texts:
            chars.update(t)
        next_id = max(self._id_to_char) + 1
        for ch in sorted(chars):
            if ch in self._char_to_id:
                continue
            self._char_to_id[ch] = next_id
            self._id_to_char[next_id] = ch
            next_id += 1

    def encode(self, text: str) -> list[int]:
        unk = self.unk_id
        return [self._char_to_id.get(ch, unk) for ch in text]

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for i in ids:
            ch = self._id_to_char.get(i, "")
            # Skip specials when decoding back to readable text.
            if ch in SPECIAL_TOKENS:
                continue
            out.append(ch)
        return "".join(out)

    # --- Persistence ---------------------------------------------------

    def save(self, path: str | Path) -> None:
        self._save_json(
            path,
            {
                "type": "CharTokenizer",
                "vocab": list(self._char_to_id.keys()),
                "ids": list(self._char_to_id.values()),
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = cls._load_json(path)
        if data.get("type") != "CharTokenizer":
            raise ValueError(f"Not a CharTokenizer file: {path}")
        obj = cls()
        obj._char_to_id = dict(zip(data["vocab"], data["ids"]))
        obj._id_to_char = {v: k for k, v in obj._char_to_id.items()}
        return obj
