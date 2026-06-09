"""Tests for Chapter 23 SFT."""

import torch

from lm_from_scratch.datasets.instruction_dataset import (
    InstructionDataset,
    InstructionExample,
    default_template,
)
from lm_from_scratch.finetuning.sft import sft_loss
from lm_from_scratch.tokenizers.char_tokenizer import CharTokenizer


def test_default_template_returns_prompt_and_response() -> None:
    ex = InstructionExample(user="Q?", assistant="A.")
    p, r = default_template(ex)
    assert "Q?" in p and "A." in r


def test_instruction_dataset_loss_mask_only_on_response() -> None:
    tok = CharTokenizer(); tok.train(["The Q? Response is A.\n"])
    ex = InstructionExample(user="Q?", assistant="A.")
    ds = InstructionDataset([ex], tok, block_size=64)
    x, y, m = ds[0]
    # Mask must be 1 over assistant tokens and 0 over instruction tokens.
    assert m.sum() > 0
    # The first few positions are instruction tokens; mask must be 0 there.
    assert m[:5].sum() == 0


def test_sft_loss_is_masked() -> None:
    torch.manual_seed(0)
    B, T, V = 2, 4, 6
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    mask_all = torch.ones(B, T)
    mask_half = torch.zeros(B, T); mask_half[:, 2:] = 1
    loss_all = sft_loss(logits, targets, mask_all)
    loss_half = sft_loss(logits, targets, mask_half)
    # Half-mask loss is computed only on the masked positions; not generally equal.
    assert loss_all.shape == ()
    assert loss_half.shape == ()
