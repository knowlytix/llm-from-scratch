"""Smoke tests for TinyGPTClassifier."""

from __future__ import annotations

import torch

from llm_from_scratch.finetuning.classification import (
    ClassificationDataset,
    ClassifierConfig,
    TinyGPTClassifier,
    classification_train,
    evaluate_classifier,
)
from llm_from_scratch.models.configs import GPTConfig
from llm_from_scratch.models.gpt import TinyGPT


def _tiny_backbone(vocab_size: int = 280, block_size: int = 16) -> TinyGPT:
    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=32,
        num_layers=2,
        num_heads=2,
        mlp_ratio=2,
        dropout=0.0,
    )
    return TinyGPT(cfg)


def test_forward_shapes_and_loss_runs() -> None:
    backbone = _tiny_backbone()
    model = TinyGPTClassifier(backbone, ClassifierConfig(num_classes=3))
    input_ids = torch.randint(low=4, high=280, size=(4, 12))
    labels = torch.tensor([0, 1, 2, 1])
    logits, loss = model(input_ids, labels)
    assert logits.shape == (4, 3)
    assert loss is not None
    assert torch.isfinite(loss)


def test_pool_uses_last_non_pad_position() -> None:
    backbone = _tiny_backbone()
    model = TinyGPTClassifier(backbone, ClassifierConfig(num_classes=3, pad_id=0))
    # Two sequences with different effective lengths; pad id = 0.
    ids = torch.tensor([
        [10, 11, 12, 0, 0, 0],
        [20, 21, 22, 23, 24, 0],
    ])
    hidden = model._hidden_states(ids)
    pooled = model._pool(hidden, ids)
    # The pooled feature for row 0 should equal the hidden state at index 2.
    assert torch.allclose(pooled[0], hidden[0, 2])
    assert torch.allclose(pooled[1], hidden[1, 4])


def test_classify_returns_label_and_confidence() -> None:
    backbone = _tiny_backbone()
    model = TinyGPTClassifier(backbone, ClassifierConfig(num_classes=3))
    ids = torch.randint(low=4, high=280, size=(8,))  # 1-D input is accepted
    pred, conf = model.classify(ids)
    assert pred.shape == (1,)
    assert conf.shape == (1,)
    assert 0 <= int(pred.item()) < 3
    assert 0.0 <= float(conf.item()) <= 1.0


def test_training_decreases_loss_on_trivial_task() -> None:
    """One class per first-token: the head alone should learn this in a few epochs."""
    torch.manual_seed(0)
    backbone = _tiny_backbone(vocab_size=64, block_size=8)
    model = TinyGPTClassifier(backbone, ClassifierConfig(num_classes=3))

    examples: list[tuple[list[int], int]] = []
    for cls_id in range(3):
        for _ in range(20):
            # Trivial signal: first non-pad token IDs cluster by class.
            seq = [4 + cls_id] * 5
            examples.append((seq, cls_id))

    dataset = ClassificationDataset(examples, block_size=8, pad_id=0)

    pre_loss, pre_acc = evaluate_classifier(model, dataset, batch_size=8)
    history = classification_train(
        model, dataset, batch_size=8, lr=5e-3,
        max_epochs=10, unfreeze_last_blocks=1,
    )
    post_loss, post_acc = evaluate_classifier(model, dataset, batch_size=8)

    assert history["train_loss"][-1] < history["train_loss"][0]
    assert post_loss < pre_loss
    assert post_acc >= pre_acc


def test_save_load_roundtrip(tmp_path) -> None:
    backbone = _tiny_backbone()
    model = TinyGPTClassifier(backbone, ClassifierConfig(num_classes=3))
    ids = torch.randint(low=4, high=280, size=(2, 10))
    logits_before, _ = model(ids)

    save_dir = tmp_path / "clf"
    model.save(save_dir)

    loaded = TinyGPTClassifier.load(save_dir)
    logits_after, _ = loaded(ids)
    assert torch.allclose(logits_before, logits_after, atol=1e-5)
