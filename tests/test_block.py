"""Tests for Chapter 12 Transformer block and normalization."""

import torch

from lm_from_scratch.models.norms import RMSNorm
from lm_from_scratch.models.transformer_block import MLP, TransformerBlock


def test_rmsnorm_preserves_shape() -> None:
    n = RMSNorm(16)
    x = torch.randn(2, 5, 16)
    out = n(x)
    assert out.shape == x.shape


def test_rmsnorm_unit_rms_after_norm() -> None:
    n = RMSNorm(16)
    x = torch.randn(2, 5, 16) * 5
    out = n(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    # With weight=1 the RMS should be close to 1.
    assert (rms - 1.0).abs().mean().item() < 1e-2


def test_mlp_shapes_gelu() -> None:
    m = MLP(dim=16, mlp_ratio=4, activation="gelu")
    x = torch.randn(2, 5, 16)
    assert m(x).shape == (2, 5, 16)


def test_mlp_shapes_swiglu() -> None:
    m = MLP(dim=16, mlp_ratio=4, activation="swiglu")
    x = torch.randn(2, 5, 16)
    assert m(x).shape == (2, 5, 16)


def test_transformer_block_forward_shape() -> None:
    b = TransformerBlock(embedding_dim=16, num_heads=4, mlp_ratio=2, block_size=8)
    x = torch.randn(2, 6, 16)
    out = b(x)
    assert out.shape == x.shape


def test_transformer_block_residual_identity_near_init() -> None:
    # At initialization, sublayers start near zero (LayerNorm with zero
    # biases on weights), so the block is approximately identity.
    torch.manual_seed(0)
    b = TransformerBlock(embedding_dim=16, num_heads=4, mlp_ratio=2, block_size=8)
    # Zero out attention output and MLP output projections to enforce identity.
    with torch.no_grad():
        b.attn.out_proj.weight.zero_()
        if hasattr(b.mlp, "fc2"):
            b.mlp.fc2.weight.zero_()
            b.mlp.fc2.bias.zero_()
    x = torch.randn(1, 4, 16)
    out = b(x)
    assert torch.allclose(out, x, atol=1e-5)


def test_transformer_block_gradient_flow_deep_stack() -> None:
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList(
        [TransformerBlock(embedding_dim=16, num_heads=4, mlp_ratio=2, block_size=8) for _ in range(8)]
    )
    x = torch.randn(2, 6, 16, requires_grad=True)
    h = x
    for b in blocks:
        h = b(h)
    h.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_post_norm_block_forward() -> None:
    b = TransformerBlock(
        embedding_dim=16, num_heads=4, mlp_ratio=2, block_size=8, norm_style="post"
    )
    x = torch.randn(2, 6, 16)
    assert b(x).shape == x.shape
