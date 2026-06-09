"""LoRA: low-rank adaptation of linear layers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen Linear plus a trainable low-rank update.

    Output: ``y = x W^T + (alpha/r) * x A^T B^T``
    where ``A`` is ``(r, in)`` and ``B`` is ``(out, r)``. The base weight
    is held frozen.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)  # Start as identity-ish (LoRA = 0 at init).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        update = (x @ self.A.t()) @ self.B.t() * (self.alpha / self.rank)
        return out + update

    def trainable_parameter_count(self) -> int:
        return self.A.numel() + self.B.numel()


def apply_lora(model: nn.Module, target_module_names: list[str], rank: int = 8, alpha: float = 16.0) -> nn.Module:
    """Replace every nn.Linear whose name contains one of ``target_module_names`` with LoRALinear."""
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(t in child_name for t in target_module_names):
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
    return model


def lora_state_dict(model: nn.Module) -> dict:
    """Return only the LoRA-trainable parameters."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
