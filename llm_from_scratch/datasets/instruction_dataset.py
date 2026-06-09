"""Instruction-tuning dataset with loss masking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.utils.data import Dataset


@dataclass
class InstructionExample:
    user: str
    assistant: str
    system: str | None = None
    meta: dict = field(default_factory=dict)


def default_template(ex: InstructionExample) -> tuple[str, str]:
    """Returns (prompt_str, response_str). Response is what we train on."""
    prompt = f"### Instruction:\n{ex.user}\n\n### Response:\n"
    return prompt, ex.assistant + "\n"


class InstructionDataset(Dataset):
    """Tokenize once, store input_ids and a loss mask (1 over response)."""

    def __init__(
        self,
        examples: list[InstructionExample],
        tokenizer,
        block_size: int = 256,
        template: Callable[[InstructionExample], tuple[str, str]] = default_template,
    ) -> None:
        self.block_size = block_size
        self._items: list[tuple[list[int], list[int]]] = []
        for ex in examples:
            prompt, response = template(ex)
            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response)
            full = prompt_ids + response_ids
            mask = [0] * len(prompt_ids) + [1] * len(response_ids)
            full = full[:block_size]
            mask = mask[:block_size]
            # Pad if shorter
            if len(full) < block_size:
                pad = tokenizer.pad_id
                full = full + [pad] * (block_size - len(full))
                mask = mask + [0] * (block_size - len(mask))
            self._items.append((full, mask))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids, mask = self._items[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        m = torch.tensor(mask[1:], dtype=torch.float32)
        return x, y, m
