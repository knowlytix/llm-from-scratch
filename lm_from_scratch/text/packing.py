"""Pack a stream of tokens into fixed-length sequences for training.

Two strategies:

* ``sliding_windows`` slides over the token stream with a configurable stride.
  Set ``stride == block_size`` for non-overlapping windows.
* ``pack_documents`` concatenates documents with bos/eos markers and slices
  into fixed-length blocks. This is what GPT-style training uses in practice.
"""

from __future__ import annotations

from collections.abc import Iterable


def sliding_windows(
    tokens: list[int],
    block_size: int,
    stride: int | None = None,
) -> list[list[int]]:
    """Return overlapping or non-overlapping windows of ``block_size`` tokens.

    If ``stride`` is None, defaults to ``block_size`` (non-overlapping). A
    smaller stride produces more, overlapping windows; useful for training
    but biases evaluation if reused there.
    """
    if stride is None:
        stride = block_size
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    n = len(tokens)
    out: list[list[int]] = []
    for i in range(0, n - block_size + 1, stride):
        out.append(tokens[i : i + block_size])
    return out


def pack_documents(
    token_streams: Iterable[list[int]],
    block_size: int,
    bos_id: int,
    eos_id: int,
) -> list[list[int]]:
    """Concatenate documents separated by ``eos_id`` (and prefixed by ``bos_id``)
    and slice into ``block_size`` blocks.

    The final partial block is dropped to keep all training examples the
    same shape. The first token of every block is whatever the next token in
    the stream happens to be; documents may span block boundaries.
    """
    buffer: list[int] = []
    for stream in token_streams:
        buffer.append(bos_id)
        buffer.extend(stream)
        buffer.append(eos_id)

    n_blocks = len(buffer) // block_size
    return [buffer[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
