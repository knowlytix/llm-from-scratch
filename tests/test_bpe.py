"""Tests for the BPE tokenizer (Chapter 4)."""

from pathlib import Path

import pytest

from llm_from_scratch.tokenizers.bpe_internals import get_pair_counts, merge_pair
from llm_from_scratch.tokenizers.bpe_tokenizer import BPETokenizer


# --- Internals ---------------------------------------------------------


def test_get_pair_counts_simple() -> None:
    counts = get_pair_counts([[1, 2, 3], [1, 2]])
    assert counts == {(1, 2): 2, (2, 3): 1}


def test_merge_pair_replaces_all_occurrences() -> None:
    out = merge_pair([[1, 2, 1, 2, 3]], (1, 2), 99)
    assert out == [[99, 99, 3]]


def test_merge_pair_no_overlap() -> None:
    # Two overlapping candidates: only the first wins on a left-to-right scan.
    out = merge_pair([[1, 1, 1]], (1, 1), 99)
    assert out == [[99, 1]]


# --- Training ----------------------------------------------------------


def test_bpe_train_grows_vocab_to_target() -> None:
    # Need enough distinct pairs to support the requested number of merges.
    from llm_from_scratch.text.toy_corpus import load_tiny_shakespeare

    tok = BPETokenizer(vocab_size=300)  # 260 base + 40 merges
    tok.train([load_tiny_shakespeare()[:50_000]])
    assert tok.vocab_size == 300


def test_bpe_train_zero_merges_when_target_is_min() -> None:
    tok = BPETokenizer(vocab_size=260)
    tok.train(["abc"])  # no merges should be added
    assert tok.vocab_size == 260
    assert tok.merges == []


# --- Round trip on training text --------------------------------------


def test_bpe_round_trip_on_training_text() -> None:
    text = "the quick brown fox jumps over the lazy dog " * 20
    tok = BPETokenizer(vocab_size=350)
    tok.train([text])
    assert tok.decode(tok.encode(text)) == text


# --- Byte fallback: arbitrary UTF-8 round trips even without training -


def test_bpe_round_trip_arbitrary_unicode_without_training() -> None:
    tok = BPETokenizer(vocab_size=260)
    s = "Hello, 🌍! Café résumé naïve"
    assert tok.decode(tok.encode(s)) == s


def test_bpe_round_trip_arbitrary_unicode_with_training() -> None:
    tok = BPETokenizer(vocab_size=320)
    tok.train(["english training text " * 50])
    s = "Hello, 🌍! 你好"
    assert tok.decode(tok.encode(s)) == s


# --- Save / load -------------------------------------------------------


def test_bpe_save_load_round_trip(tmp_path: Path) -> None:
    tok = BPETokenizer(vocab_size=320)
    tok.train(["the quick brown fox jumps over the lazy dog " * 10])
    p = tmp_path / "bpe.json"
    tok.save(p)
    loaded = BPETokenizer.load(p)
    sample = "the quick fox"
    assert loaded.encode(sample) == tok.encode(sample)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.merges == tok.merges


# --- Encoding consistency: a longer training corpus shortens encodings -


def test_bpe_with_more_merges_shortens_encoding() -> None:
    text = "the quick brown fox jumps over the lazy dog " * 200
    short_vocab = BPETokenizer(vocab_size=261)
    short_vocab.train([text])
    long_vocab = BPETokenizer(vocab_size=500)
    long_vocab.train([text])
    sample = "the quick brown fox"
    assert len(long_vocab.encode(sample)) < len(short_vocab.encode(sample))


# --- Validation --------------------------------------------------------


def test_bpe_rejects_too_small_vocab() -> None:
    with pytest.raises(ValueError):
        BPETokenizer(vocab_size=100)
