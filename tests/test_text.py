"""Tests for the llm_from_scratch.text subpackage (Chapter 2)."""

from llm_from_scratch.text.cleaning import (
    clean_text,
    normalize_whitespace,
    strip_control_chars,
)
from llm_from_scratch.text.corpus import TextCorpus
from llm_from_scratch.text.dedup import exact_dedup, near_dedup, true_jaccard
from llm_from_scratch.text.diagnostics import corpus_summary
from llm_from_scratch.text.packing import pack_documents, sliding_windows
from llm_from_scratch.text.splitting import detect_contamination, split_corpus


# --- cleaning -----------------------------------------------------------


def test_normalize_whitespace_collapses_spaces() -> None:
    assert normalize_whitespace("a   b\t\tc") == "a b c"


def test_strip_control_chars_keeps_newlines() -> None:
    assert strip_control_chars("a\x00b\nc") == "ab\nc"


def test_clean_text_idempotent() -> None:
    raw = "Hello \t\tWorld\x00!\n\n\n\nNext"
    once = clean_text(raw)
    twice = clean_text(once)
    assert once == twice


# --- corpus -------------------------------------------------------------


def test_textcorpus_summary_counts() -> None:
    c = TextCorpus(["abc", "de", "fghij"])
    s = c.summary()
    assert s["num_documents"] == 3
    assert s["total_chars"] == 10
    assert s["max_doc_chars"] == 5


def test_textcorpus_map_and_filter() -> None:
    c = TextCorpus(["a", "bb", "ccc"])
    longer = c.filter(lambda d: len(d) >= 2).map(str.upper)
    assert longer.documents == ["BB", "CCC"]


# --- dedup --------------------------------------------------------------


def test_exact_dedup_removes_byte_equal_duplicates() -> None:
    docs = ["x", "y", "x", "z", "y"]
    assert exact_dedup(docs) == ["x", "y", "z"]


def test_near_dedup_drops_near_duplicates() -> None:
    # Two documents that differ only by a short suffix; their n-gram sets
    # overlap heavily so MinHash-estimated Jaccard reliably clears 0.5.
    base = ("the quick brown fox jumps over the lazy dog " * 8).strip()
    perturbed = base + " a brief addition that does not change much"
    far = ("completely unrelated text about something else entirely " * 8).strip()
    docs = [base, perturbed, far]
    out = near_dedup(docs, ngram=13, threshold=0.5)
    # base and perturbed are near-duplicates; far is not
    assert len(out) == 2
    assert far in out


def test_true_jaccard_self_and_disjoint() -> None:
    assert true_jaccard("the quick brown fox", "the quick brown fox") == 1.0
    assert true_jaccard("abcdef", "uvwxyz", ngram=3) == 0.0


# --- splitting ----------------------------------------------------------


def test_split_sizes_sum_to_total() -> None:
    docs = [f"doc {i}" for i in range(100)]
    train, valid, test = split_corpus(TextCorpus(docs), 0.8, 0.1, 0.1, seed=0)
    assert len(train) + len(valid) + len(test) == 100


def test_split_reproducible() -> None:
    docs = [f"doc {i}" for i in range(50)]
    a = split_corpus(TextCorpus(docs), 0.8, 0.1, 0.1, seed=7)
    b = split_corpus(TextCorpus(docs), 0.8, 0.1, 0.1, seed=7)
    assert [d for d in a[0]] == [d for d in b[0]]


def test_detect_contamination_finds_planted_leak() -> None:
    leak = "this is a very specific sentence that should be detected reliably"
    train = TextCorpus(["benign text and the " + leak + " and more text"])
    eval_c = TextCorpus(["other text " + leak + " continues here"])
    hits = detect_contamination(train, eval_c, ngram=40)
    assert len(hits) > 0
    assert any(leak[:40] in span for _, _, span in hits)


def test_detect_contamination_negative() -> None:
    train = TextCorpus(["the quick brown fox"])
    eval_c = TextCorpus(["completely different unrelated content here"])
    hits = detect_contamination(train, eval_c, ngram=30)
    assert hits == []


# --- packing ------------------------------------------------------------


def test_sliding_windows_nonoverlapping() -> None:
    tokens = list(range(10))
    out = sliding_windows(tokens, block_size=4, stride=4)
    assert out == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_sliding_windows_overlapping() -> None:
    tokens = list(range(6))
    out = sliding_windows(tokens, block_size=3, stride=1)
    assert out == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]


def test_pack_documents_block_count() -> None:
    streams = [[1, 2, 3], [4, 5]]
    bos, eos = 100, 101
    out = pack_documents(streams, block_size=4, bos_id=bos, eos_id=eos)
    # Buffer: [100,1,2,3,101,100,4,5,101] = 9 tokens -> floor(9/4)=2 blocks of size 4
    assert len(out) == 2
    assert all(len(b) == 4 for b in out)


# --- diagnostics --------------------------------------------------------


def test_corpus_summary_duplicate_rate() -> None:
    c = TextCorpus(["a", "a", "b", "c", "c"])
    s = corpus_summary(c)
    # 3 distinct out of 5 -> duplicate rate 0.4
    assert abs(s["duplicate_rate"] - 0.4) < 1e-9
