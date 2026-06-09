"""Efficient training helpers: mixed precision and gradient checkpointing.

These are wrappers that flip a small number of switches; they do not
reimplement the underlying machinery. The educational point is that each
switch comes with a documented tradeoff.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def enable_gradient_checkpointing(model: nn.Module, modules: list[type] | None = None) -> None:
    """Wrap every module of the given types with ``torch.utils.checkpoint``.

    Defaults to checkpointing every ``TransformerBlock`` in the model.
    """
    from llm_from_scratch.models.transformer_block import TransformerBlock
    targets = modules or [TransformerBlock]

    for name, module in model.named_modules():
        if isinstance(module, tuple(targets)):
            original_forward = module.forward

            def make_ckpt(orig):
                def _ckpt_forward(*args, **kwargs):
                    return torch.utils.checkpoint.checkpoint(orig, *args, use_reentrant=False, **kwargs)
                return _ckpt_forward

            module.forward = make_ckpt(original_forward)


def autocast_context(dtype: str = "bf16"):
    """Return an autocast context manager for the requested dtype.

    Use as ``with autocast_context("bf16"): logits, loss = model(x, y)``.
    """
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    target = dtype_map[dtype]
    return torch.amp.autocast(device_type="cuda", dtype=target)


def use_fused_sdpa() -> bool:
    """Return True if PyTorch's fused scaled_dot_product_attention is available."""
    return hasattr(torch.nn.functional, "scaled_dot_product_attention")
