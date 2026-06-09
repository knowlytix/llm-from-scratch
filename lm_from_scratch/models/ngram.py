"""N-gram language model with add-k smoothing.

The classical count-based language model. Fits in a few hundred lines, fits
in memory for modestly-sized corpora and produces a perplexity baseline
every later model in the book must beat.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable


class NGramLanguageModel:
    """An n-gram language model with add-k smoothing.

    Parameters
    ----------
    n:
        Order of the model. ``n=1`` is unigram, ``n=2`` is bigram, etc.
    smoothing:
        ``"none"`` produces the maximum-likelihood estimate (zero probability
        for unseen contexts). ``"add_k"`` adds a pseudo-count of ``k`` to
        every numerator and ``k * vocab_size`` to every denominator.
    k:
        Smoothing pseudo-count for ``add_k``. Defaults to 0.1.
    """

    def __init__(
        self,
        n: int,
        smoothing: str = "add_k",
        k: float = 0.1,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if smoothing not in {"none", "add_k"}:
            raise ValueError(f"unsupported smoothing: {smoothing}")
        self.n = n
        self.smoothing = smoothing
        self.k = k
        self._context_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self._context_totals: Counter[tuple[int, ...]] = Counter()
        self._vocab_size = 0

    # --- Fitting -------------------------------------------------------

    def fit(self, token_ids: Iterable[int]) -> None:
        """Count contexts and the next tokens that followed them."""
        ids = list(token_ids)
        if not ids:
            raise ValueError("empty token stream")
        self._vocab_size = max(self._vocab_size, max(ids) + 1)

        # Use a fixed BOS-padded prefix so positions 0..n-2 have valid contexts.
        bos = -1  # sentinel; we treat ``-1`` as "before the start"
        padded = [bos] * (self.n - 1) + ids

        # For n=1 we collect a single unigram context (the empty tuple).
        for i in range(len(padded) - self.n + 1):
            context = tuple(padded[i : i + self.n - 1])
            target = padded[i + self.n - 1]
            if target == bos:
                continue
            self._context_counts[context][target] += 1
            self._context_totals[context] += 1

    # --- Probability ---------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def prob(self, context: tuple[int, ...], next_id: int) -> float:
        ctx = self._effective_context(context)
        counts = self._context_counts.get(ctx, Counter())
        total = self._context_totals.get(ctx, 0)
        if self.smoothing == "add_k":
            return (counts.get(next_id, 0) + self.k) / (total + self.k * self._vocab_size)
        if total == 0:
            return 0.0
        return counts.get(next_id, 0) / total

    def log_prob_sequence(self, token_ids: list[int]) -> float:
        """Sum of natural-log probabilities of each token under the model."""
        bos = -1
        padded = [bos] * (self.n - 1) + list(token_ids)
        total = 0.0
        for i in range(len(padded) - self.n + 1):
            ctx = tuple(padded[i : i + self.n - 1])
            target = padded[i + self.n - 1]
            if target == bos:
                continue
            p = self.prob(ctx, target)
            if p <= 0:
                return float("-inf")
            total += math.log(p)
        return total

    def perplexity(self, token_ids: list[int]) -> float:
        """``exp(-log_prob / N)`` for sequence of length ``N``.

        Raises a ``ValueError`` if the model assigns zero probability to any
        token in the sequence (unsmoothed) since perplexity is undefined.
        """
        ids = list(token_ids)
        if not ids:
            raise ValueError("empty token stream")
        lp = self.log_prob_sequence(ids)
        if lp == float("-inf"):
            raise ValueError("model assigns zero probability; smoothing is required")
        return math.exp(-lp / len(ids))

    # --- Generation ----------------------------------------------------

    def sample(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        rng: random.Random | None = None,
    ) -> list[int]:
        """Sample autoregressively from the model.

        The conditional distribution at each step is given by the n-gram
        counts (with smoothing). The sampler walks character-by-character
        without temperature; Chapter~19 introduces sampling decorations.
        """
        if rng is None:
            rng = random.Random(0)
        ids = list(prompt_ids)
        bos = -1
        for _ in range(max_new_tokens):
            ctx_full = (
                [bos] * (self.n - 1 - len(ids)) + ids[-(self.n - 1) :] if self.n > 1 else []
            )
            ctx = tuple(ctx_full)
            ctx = self._effective_context(ctx)
            probs = [self.prob(ctx, i) for i in range(self._vocab_size)]
            total = sum(probs)
            if total == 0:
                break
            r = rng.random() * total
            acc = 0.0
            chosen = 0
            for i, p in enumerate(probs):
                acc += p
                if r <= acc:
                    chosen = i
                    break
            ids.append(chosen)
        return ids

    # --- Internals -----------------------------------------------------

    def _effective_context(self, context: tuple[int, ...]) -> tuple[int, ...]:
        """Truncate the context to (n-1) tokens; longer contexts are reduced."""
        if self.n == 1:
            return ()
        return tuple(context[-(self.n - 1) :])
