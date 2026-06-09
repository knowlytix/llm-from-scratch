"""Back-of-envelope scaling estimates: parameters, FLOPs and memory."""

from __future__ import annotations

from typing import Any

from llm_from_scratch.models.configs import GPTConfig


def count_parameters(config: GPTConfig, exclude_embedding: bool = False) -> int:
    """Estimate the parameter count of a TinyGPT-style model from its config."""
    V, d, L, r = (
        config.vocab_size,
        config.embedding_dim,
        config.num_layers,
        config.mlp_ratio,
    )
    emb = V * d
    if not config.tie_embeddings:
        emb += V * d  # separate LM head
    if config.positional == "learned":
        emb += config.block_size * d
    per_block = 4 * d * d + 2 * r * d * d  # attention + MLP, ignoring LN scales
    body = L * per_block + d  # plus final LN scale
    total = (0 if exclude_embedding else emb) + body
    return int(total)


def training_flops(num_params: int, num_tokens: int) -> float:
    """Approximate training FLOPs as 6 * N * D."""
    return 6.0 * num_params * num_tokens


def inference_flops(num_params: int, num_tokens: int) -> float:
    """Approximate inference FLOPs as 2 * N * D (forward only)."""
    return 2.0 * num_params * num_tokens


def memory_footprint(
    config: GPTConfig,
    batch_size: int = 32,
    dtype: str = "bf16",
    optimizer: str = "adamw",
) -> dict[str, int]:
    """Bytes-per-component estimate. Activations bound is approximate."""
    bytes_per_elem = {"fp32": 4, "bf16": 2, "fp16": 2}[dtype]
    N = count_parameters(config)
    params_bytes = N * bytes_per_elem
    grads_bytes = N * bytes_per_elem
    # AdamW maintains fp32 m, v (and sometimes fp32 master params).
    if optimizer == "adamw":
        optimizer_bytes = 2 * N * 4
    else:
        optimizer_bytes = N * 4
    # Activation memory: rough estimate dominated by attention scores
    # of shape (B, H, T, T) plus MLP activations (B, T, r*d).
    B, T, d, L, H = (
        batch_size,
        config.block_size,
        config.embedding_dim,
        config.num_layers,
        config.num_heads,
    )
    attn_acts = L * B * H * T * T * bytes_per_elem
    mlp_acts = L * B * T * (config.mlp_ratio * d) * bytes_per_elem
    activations_bytes = attn_acts + mlp_acts
    total = params_bytes + grads_bytes + optimizer_bytes + activations_bytes
    return {
        "params_bytes": int(params_bytes),
        "grads_bytes": int(grads_bytes),
        "optimizer_bytes": int(optimizer_bytes),
        "activations_bytes": int(activations_bytes),
        "total_bytes": int(total),
    }


def chinchilla_optimal(
    compute_budget: float | None = None,
    target_params: int | None = None,
    tokens_per_param: float = 20.0,
) -> dict[str, float]:
    """Apply the Chinchilla rule of thumb that D ≈ 20 * N.

    Pass either ``compute_budget`` (FLOPs) or ``target_params``; the
    function fills in the rest.
    """
    if compute_budget is not None:
        # C = 6 N D, D = 20 N -> C = 120 N^2 -> N = sqrt(C / 120)
        N = (compute_budget / 120.0) ** 0.5
        D = tokens_per_param * N
        return {"params": N, "tokens": D, "compute": 6 * N * D}
    if target_params is not None:
        N = float(target_params)
        D = tokens_per_param * N
        return {"params": N, "tokens": D, "compute": 6 * N * D}
    raise ValueError("provide either compute_budget or target_params")


def time_estimate(flops: float, throughput_flops_per_sec: float) -> float:
    """Wall-clock seconds at the given throughput."""
    return flops / throughput_flops_per_sec
