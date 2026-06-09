"""KV cache: per-layer (k, v) buffers grown one decode step at a time."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class KVCache:
    """A list of per-layer ``(k, v)`` tensors. Initialized empty; grown by appending."""

    keys: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    num_layers: int = 0

    @classmethod
    def empty(cls, num_layers: int) -> "KVCache":
        return cls(keys=[None] * num_layers, values=[None] * num_layers, num_layers=num_layers)  # type: ignore[list-item]

    def append(self, layer: int, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        if self.keys[layer] is None:
            self.keys[layer] = new_k
            self.values[layer] = new_v
        else:
            self.keys[layer] = torch.cat([self.keys[layer], new_k], dim=-2)
            self.values[layer] = torch.cat([self.values[layer], new_v], dim=-2)

    def length(self) -> int:
        if self.keys and self.keys[0] is not None:
            return int(self.keys[0].size(-2))
        return 0

    def memory_bytes(self) -> int:
        total = 0
        for k in self.keys:
            if k is not None:
                total += k.element_size() * k.numel()
        for v in self.values:
            if v is not None:
                total += v.element_size() * v.numel()
        return total
