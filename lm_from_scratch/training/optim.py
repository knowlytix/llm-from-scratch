"""Hand-rolled AdamW and weight-decay parameter grouping."""

from __future__ import annotations

import torch


class HandRolledAdamW(torch.optim.Optimizer):
    """AdamW from scratch, for teaching.

    The book defaults to ``torch.optim.AdamW`` everywhere; this class is
    used in one notebook cell to verify the update matches the reference
    implementation step-for-step.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                state["m"].mul_(beta1).add_(g, alpha=1 - beta1)
                state["v"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                m_hat = state["m"] / bc1
                v_hat = state["v"] / bc2
                # Decoupled weight decay.
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        return loss


def param_groups_for_weight_decay(
    model: torch.nn.Module, weight_decay: float
) -> list[dict]:
    """Return two parameter groups: one with weight decay (2-D weights), one without.

    Biases and 1-D parameters (LayerNorm weight, RMSNorm scale, embeddings
    if you treat them that way) are excluded from decay.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
