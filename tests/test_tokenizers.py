"""Tests for the lm_from_scratch.tokenizers subpackage (Chapter 3)."""

from pathlib import Path

import pytest

from lm_from_scratch.tokenizers.base import BaseTokenizer
from lm_from_scratch.tokenizers.byte_tokenizer import ByteTokenizer
from lm_from_scratch.tokenizers.char_tokenizer import CharTokenizer
from lm_from_scratch.tokenizers.diagnostics import (
    compression_ratio,
    tokenizer_report,
    tokens_per_word,
    unknown_token_rate,
)
from lm_from_scratch.tokenizers.word_tokenizer import WhitespaceTokenizer


# --- Round-trip identity on training corpus ---------------------------


def test_char_tokenizer_round_trip() -> None:
    tok = CharTokenizer()
    text = "Hello, world!"
    tok.train([text])
    assert tok.decode(tok.encode(text)) == text


def test_whitespace_tokenizer_round_trip_on_simple_text() -> None:
    tok = WhitespaceTokenizer()
    text = "the quick brown fox"
    tok.train([text])
    # Whitespace tokenizer collapses internal whitespace; reassembled text has single spaces.
    assert tok.decode(tok.encode(text)) == text


def test_byte_tokenizer_round_trip_arbitrary_unicode() -> None:
    tok = ByteTokenizer()
    text = "Hello, 🌍! Hé, café é"
    assert tok.decode(tok.encode(text)) == text


# --- Vocabulary size correctness ---------------------------------------


def test_char_tokenizer_vocab_size_correct() -> None:
    tok = CharTokenizer()
    tok.train(["abc"])  # 3 unique chars + 4 specials = 7
    assert tok.vocab_size == 7


def test_byte_tokenizer_vocab_size_constant() -> None:
    tok = ByteTokenizer()
    # 4 specials + 256 byte values
    assert tok.vocab_size == 260


# --- Save/load round-trip ----------------------------------------------


def test_char_tokenizer_save_load(tmp_path: Path) -> None:
    tok = CharTokenizer()
    tok.train(["hello world"])
    p = tmp_path / "char.json"
    tok.save(p)
    loaded = CharTokenizer.load(p)
    assert loaded.encode("hello world") == tok.encode("hello world")


def test_whitespace_tokenizer_save_load(tmp_path: Path) -> None:
    tok = WhitespaceTokenizer()
    tok.train(["the quick brown fox"])
    p = tmp_path / "ws.json"
    tok.save(p)
    loaded = WhitespaceTokenizer.load(p)
    assert loaded.encode("the quick") == tok.encode("the quick")


def test_byte_tokenizer_save_load(tmp_path: Path) -> None:
    tok = ByteTokenizer()
    p = tmp_path / "byte.json"
    tok.save(p)
    loaded = ByteTokenizer.load(p)
    assert loaded.encode("hello") == tok.encode("hello")


# --- Unknown handling --------------------------------------------------


def test_whitespace_tokenizer_unk_on_oov() -> None:
    tok = WhitespaceTokenizer()
    tok.train(["seen word"])
    ids = tok.encode("seen unseen word")
    assert tok.unk_id in ids


# --- Diagnostics --------------------------------------------------------


def test_compression_ratio_byte_is_one() -> None:
    tok = ByteTokenizer()
    texts = ["abc", "def"]
    # ASCII: 1 char = 1 byte = 1 token
    assert compression_ratio(tok, texts) == pytest.approx(1.0)


def test_tokens_per_word_char_is_avg_word_length_plus_one() -> None:
    tok = CharTokenizer()
    text = "the quick brown fox"  # 4 words, 19 chars including spaces
    tok.train([text])
    # 19 tokens / 4 words = 4.75
    assert tokens_per_word(tok, [text]) == pytest.approx(19 / 4)


def test_unk_rate_word_tokenizer() -> None:
    tok = WhitespaceTokenizer()
    tok.train(["seen"])
    rate = unknown_token_rate(tok, ["seen unseen unseen"])
    # 1 known, 2 unknown out of 3
    assert rate == pytest.approx(2 / 3)


def test_tokenizer_report_keys() -> None:
    tok = CharTokenizer()
    tok.train(["hello"])
    rep = tokenizer_report(tok, ["hello world"])
    assert {"type", "vocab_size", "compression_ratio", "tokens_per_word", "unknown_token_rate"} == set(
        rep
    )


# --- Subclass relationship ---------------------------------------------


def test_all_tokenizers_are_basetokenizers() -> None:
    assert isinstance(CharTokenizer(), BaseTokenizer)
    assert isinstance(WhitespaceTokenizer(), BaseTokenizer)
    assert isinstance(ByteTokenizer(), BaseTokenizer)
