"""Sequence classification head + trainer for TinyGPT.

Adds a sequence-classification interface on top of an existing TinyGPT
backbone. The backbone is reused as a feature extractor: its token
embedding, positional embedding, transformer blocks and final norm are
applied as in normal training, but the language-modeling head is bypassed.
The hidden state at the last non-pad position is pooled and projected
through a single Linear layer to ``num_classes`` logits.

This module is the fine-tuning example used by both ``llm-tutorial``
(Chapter 13) and ``agent-tutorial-private`` (Chapter 15 capstone:
classifying banking complaints vs. inquiries vs. other).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from llm_from_scratch.models.configs import GPTConfig
from llm_from_scratch.models.gpt import TinyGPT


@dataclass
class ClassifierConfig:
    """Configuration for a :class:`TinyGPTClassifier`."""

    num_classes: int
    pad_id: int = 0
    head_dropout: float = 0.0


class TinyGPTClassifier(nn.Module):
    """TinyGPT backbone + linear classification head.

    Parameters
    ----------
    backbone:
        A :class:`TinyGPT` instance. Its parameters are reused; the
        backbone's ``lm_head`` is left in place but never called.
    cls_config:
        :class:`ClassifierConfig` describing the number of classes and the
        pad id used to identify the last non-pad position for pooling.
    """

    def __init__(self, backbone: TinyGPT, cls_config: ClassifierConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.cls_config = cls_config
        self.head_dropout = nn.Dropout(cls_config.head_dropout)
        self.cls_head = nn.Linear(backbone.config.embedding_dim, cls_config.num_classes)
        nn.init.normal_(self.cls_head.weight, mean=0.0, std=backbone.config.init_std)
        nn.init.zeros_(self.cls_head.bias)

    # --- Forward ----------------------------------------------------------

    def _hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        T = input_ids.size(1)
        if T > b.config.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {b.config.block_size}"
            )
        x = b.token_embedding(input_ids)
        x = b.position_embedding(x)
        x = b.dropout(x)
        for block in b.blocks:
            x = block(x)
        x = b.norm_final(x)
        return x

    def _pool(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != self.cls_config.pad_id).long()
        lengths = mask.sum(dim=1).clamp_min(1)
        last_idx = (lengths - 1).clamp_min(0)
        B, _, D = hidden.shape
        gather_idx = last_idx.view(B, 1, 1).expand(B, 1, D)
        return hidden.gather(dim=1, index=gather_idx).squeeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self._hidden_states(input_ids)
        pooled = self._pool(hidden, input_ids)
        pooled = self.head_dropout(pooled)
        logits = self.cls_head(pooled)
        loss = None if labels is None else nn.functional.cross_entropy(logits, labels)
        return logits, loss

    @torch.no_grad()
    def classify(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(predicted_label, confidence)``.

        ``input_ids`` may be 1-D (one sequence) or 2-D ``(B, T)``.
        Confidence is the softmax probability of the predicted label.
        """
        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        logits, _ = self.forward(input_ids)
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        return pred, conf

    # --- Staged unfreezing -----------------------------------------------

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_blocks(self, k: int) -> None:
        """Unfreeze the final ``k`` transformer blocks and the final norm."""
        k = max(0, min(k, len(self.backbone.blocks)))
        if k == 0:
            return
        for blk in self.backbone.blocks[-k:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.backbone.norm_final.parameters():
            p.requires_grad = True

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    # --- Persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        meta = {
            "backbone_config": asdict(self.backbone.config),
            "cls_config": asdict(self.cls_config),
        }
        (path / "config.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> "TinyGPTClassifier":
        path = Path(path)
        meta = json.loads((path / "config.json").read_text())
        backbone = TinyGPT(GPTConfig(**meta["backbone_config"]))
        obj = cls(backbone, ClassifierConfig(**meta["cls_config"]))
        state = torch.load(path / "model.pt", map_location=device or "cpu")
        obj.load_state_dict(state)
        if device is not None:
            obj.to(device)
        obj.eval()
        return obj


# --- Dataset + Trainer ----------------------------------------------------


class ClassificationDataset(Dataset):
    """Padded in-memory dataset of ``(token_ids, label_id)`` pairs."""

    def __init__(
        self,
        examples: list[tuple[list[int], int]],
        block_size: int,
        pad_id: int = 0,
    ) -> None:
        self.block_size = block_size
        self.pad_id = pad_id
        self.inputs: list[torch.Tensor] = []
        self.labels: list[int] = []
        for ids, lbl in examples:
            ids = list(ids[:block_size])
            ids = ids + [pad_id] * (block_size - len(ids))
            self.inputs.append(torch.tensor(ids, dtype=torch.long))
            self.labels.append(int(lbl))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], torch.tensor(self.labels[idx], dtype=torch.long)


def classification_train(
    model: TinyGPTClassifier,
    train_dataset: ClassificationDataset,
    valid_dataset: ClassificationDataset | None = None,
    *,
    batch_size: int = 16,
    lr: float = 3e-4,
    max_epochs: int = 20,
    unfreeze_last_blocks: int = 1,
    eval_every: int = 1,
    device: str | torch.device | None = None,
) -> dict:
    """Train the head (and optionally the final ``k`` transformer blocks).

    Returns a history dict with per-epoch loss and accuracy for train and
    (if provided) validation.
    """
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    model.freeze_backbone()
    model.unfreeze_last_blocks(unfreeze_last_blocks)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=0.0)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    history: dict[str, list] = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "valid_loss": [], "valid_acc": [],
    }
    for epoch in range(1, max_epochs + 1):
        model.train()
        tot_loss = 0.0
        tot_n = 0
        tot_correct = 0
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item()) * x.size(0)
            tot_n += x.size(0)
            tot_correct += int((logits.argmax(dim=-1) == y).sum().item())
        train_loss = tot_loss / max(1, tot_n)
        train_acc = tot_correct / max(1, tot_n)
        v_loss = float("nan"); v_acc = float("nan")
        if valid_dataset is not None and (epoch % eval_every == 0):
            v_loss, v_acc = evaluate_classifier(
                model, valid_dataset, batch_size=batch_size, device=device
            )
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(v_loss)
        history["valid_acc"].append(v_acc)
    return history


@torch.no_grad()
def evaluate_classifier(
    model: TinyGPTClassifier,
    dataset: ClassificationDataset,
    *,
    batch_size: int = 16,
    device: str | torch.device | None = None,
) -> tuple[float, float]:
    """Return ``(mean_cross_entropy, accuracy)`` over ``dataset``."""
    if device is None:
        device = next(model.parameters()).device
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    tot_loss = 0.0
    tot_n = 0
    tot_correct = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits, loss = model(x, y)
        tot_loss += float(loss.item()) * x.size(0)
        tot_n += x.size(0)
        tot_correct += int((logits.argmax(dim=-1) == y).sum().item())
    return tot_loss / max(1, tot_n), tot_correct / max(1, tot_n)
