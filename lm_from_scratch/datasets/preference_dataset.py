"""Preference dataset for DPO."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset


@dataclass
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    meta: dict = field(default_factory=dict)


class PreferenceDataset(Dataset):
    def __init__(self, examples: list[PreferenceExample], tokenizer, max_len: int = 128) -> None:
        self._items = []
        for ex in examples:
            p = tokenizer.encode(ex.prompt)
            c = tokenizer.encode(ex.chosen)
            r = tokenizer.encode(ex.rejected)
            self._items.append((p[:max_len], c[:max_len], r[:max_len]))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        return self._items[idx]
