"""Grounding: does the response stay close to the supplied context?"""

from __future__ import annotations


class GroundingCheck:
    def __init__(self, ngram: int = 5, threshold: float = 0.3) -> None:
        self.ngram = ngram
        self.threshold = threshold

    def _ngrams(self, s: str) -> set[str]:
        words = s.lower().split()
        if len(words) < self.ngram:
            return {" ".join(words)}
        return {" ".join(words[i: i + self.ngram]) for i in range(len(words) - self.ngram + 1)}

    def score(self, response: str, context: str) -> float:
        r = self._ngrams(response)
        c = self._ngrams(context)
        if not r:
            return 0.0
        return len(r & c) / len(r)

    def passes(self, response: str, context: str) -> bool:
        return self.score(response, context) >= self.threshold
