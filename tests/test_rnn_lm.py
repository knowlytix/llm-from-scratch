"""Tests for the Chapter 7 RNN language models."""

import torch

from llm_from_scratch.datasets.sequence_dataset import CausalLMDataset
from llm_from_scratch.models.rnn_from_scratch import HandRolledGRUCell
from llm_from_scratch.models.rnn_lm import RNNLanguageModel
from llm_from_scratch.training.simple_loop import simple_train
from llm_from_scratch.utils.env import set_seed


def test_rnn_lm_forward_shape() -> None:
    m = RNNLanguageModel(vocab_size=10, embedding_dim=8, hidden_dim=16, num_layers=1)
    x = torch.randint(0, 10, (2, 5))
    logits, _ = m(x)
    assert logits.shape == (2, 5, 10)


def test_gru_lm_overfits_small_sequence() -> None:
    set_seed(0)
    torch.manual_seed(0)
    vocab = 5
    ids = ([0, 1, 2, 3, 4] * 50)
    ds = CausalLMDataset(ids, block_size=8, stride=4)
    m = RNNLanguageModel(vocab_size=vocab, embedding_dim=8, hidden_dim=16, num_layers=1, rnn_type="gru")
    history = simple_train(m, ds, valid_dataset=None, max_steps=200, batch_size=8, lr=3e-3, eval_every=50)
    assert history.train_loss[-1] < 0.5  # well below ln(5) ≈ 1.61


def test_hand_rolled_gru_matches_pytorch_grucell() -> None:
    torch.manual_seed(0)
    H, In = 8, 6
    ref = torch.nn.GRUCell(In, H)
    custom = HandRolledGRUCell(In, H)
    # Copy weights so the two cells are equivalent.
    with torch.no_grad():
        custom.weight_ih.copy_(ref.weight_ih)
        custom.weight_hh.copy_(ref.weight_hh)
        custom.bias_ih.copy_(ref.bias_ih)
        custom.bias_hh.copy_(ref.bias_hh)
    x = torch.randn(3, In)
    h = torch.randn(3, H)
    out_ref = ref(x, h)
    out_custom = custom(x, h)
    assert torch.allclose(out_ref, out_custom, atol=1e-5)


def test_lstm_returns_tuple_hidden() -> None:
    m = RNNLanguageModel(vocab_size=10, embedding_dim=8, hidden_dim=16, rnn_type="lstm")
    x = torch.randint(0, 10, (2, 4))
    _, hidden = m(x)
    assert isinstance(hidden, tuple)
    assert len(hidden) == 2
