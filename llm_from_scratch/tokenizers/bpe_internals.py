"""Helpers for the BPE training loop.

Kept in a separate module so the merge logic and pair-counting can be
inspected and unit-tested without the surrounding tokenizer state.
"""

from __future__ import annotations

from collections import Counter


def get_pair_counts(pre_tokens: list[list[int]]) -> Counter[tuple[int, int]]:
    """Count adjacent token pairs across all pre-tokens.

    A pre-token is a list of current token ids representing a chunk
    produced by the BPE pre-tokenizer (typically a word). Pair counts are
    summed over all pre-tokens.
    """
    counts: Counter[tuple[int, int]] = Counter()
    for seq in pre_tokens:
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] += 1
    return counts


def merge_pair(
    pre_tokens: list[list[int]],
    pair: tuple[int, int],
    new_id: int,
) -> list[list[int]]:
    """Apply a single merge to every pre-token.

    Replace every adjacent occurrence of ``pair`` with ``new_id``. We scan
    each pre-token left to right, never overlapping a previous merge.
    """
    a, b = pair
    out: list[list[int]] = []
    for seq in pre_tokens:
        if len(seq) < 2:
            out.append(seq)
            continue
        new_seq: list[int] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                new_seq.append(new_id)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        out.append(new_seq)
    return out
