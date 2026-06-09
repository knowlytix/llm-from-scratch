"""Tests for llm_from_scratch.text.toy_corpus."""

from llm_from_scratch.text.toy_corpus import load_tiny_shakespeare, next_token_counts


def test_load_tiny_shakespeare_returns_nonempty_text() -> None:
    text = load_tiny_shakespeare()
    assert isinstance(text, str)
    assert len(text) > 100_000
    assert "First Citizen" in text  # well-known phrase in Tiny Shakespeare


def test_next_token_counts_on_small_string() -> None:
    counts = next_token_counts("aab")
    assert counts == {("a", "a"): 1, ("a", "b"): 1}


def test_next_token_counts_total_pairs() -> None:
    text = "abcabc"
    counts = next_token_counts(text)
    assert sum(counts.values()) == len(text) - 1


def test_next_token_counts_empty_and_singleton() -> None:
    assert next_token_counts("") == {}
    assert next_token_counts("x") == {}
