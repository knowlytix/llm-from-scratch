"""Direct Preference Optimization (Rafailov et al. 2023)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def sequence_logprob(model, prompt_ids: list[int], response_ids: list[int], device) -> float:
    """Sum of log-probabilities of response tokens given prompt+previous response tokens."""
    full = prompt_ids + response_ids
    if len(full) < 2:
        return 0.0
    x = torch.tensor(full[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(full[1:], dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]
    log_probs = F.log_softmax(logits, dim=-1)
    target_lp = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1)
    # Sum only over response positions.
    resp_len = len(response_ids)
    return float(target_lp[0, -resp_len:].sum().item())


def dpo_loss(
    policy_logp_chosen: torch.Tensor,
    policy_logp_rejected: torch.Tensor,
    reference_logp_chosen: torch.Tensor,
    reference_logp_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    r"""DPO loss.

    .. math::

        \mathcal{L}_{\mathrm{DPO}} = -\log \sigma\!\big(\beta\,(\Delta^+ - \Delta^-)\big)

    where :math:`\Delta^\pm = \log \pi_\theta(y^\pm) - \log \pi_{\mathrm{ref}}(y^\pm)`.
    """
    delta_chosen = policy_logp_chosen - reference_logp_chosen
    delta_rejected = policy_logp_rejected - reference_logp_rejected
    margin = beta * (delta_chosen - delta_rejected)
    return -F.logsigmoid(margin).mean()
