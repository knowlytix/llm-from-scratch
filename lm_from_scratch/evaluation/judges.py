"""Judges: reference-match, heuristic, and LLM-as-judge (stub)."""

from __future__ import annotations

import re
from typing import Any


class ReferenceMatchJudge:
    """Fuzzy reference match."""
    def __init__(self, ignore_case: bool = True) -> None:
        self.ignore_case = ignore_case

    def __call__(self, response: str, reference: str) -> dict[str, Any]:
        r = response.lower() if self.ignore_case else response
        ref = reference.lower() if self.ignore_case else reference
        match = ref.strip() in r.strip()
        return {"score": 1 if match else 0, "rationale": "exact substring match" if match else "no match"}


class HeuristicJudge:
    """Rule-based judge for task-specific signals."""
    def __init__(self, expected_starts_with: str | None = None,
                 must_contain: list[str] | None = None) -> None:
        self.expected_starts_with = expected_starts_with
        self.must_contain = must_contain or []

    def __call__(self, response: str, reference: str = "") -> dict[str, Any]:
        score = 1
        reasons = []
        if self.expected_starts_with and not response.strip().startswith(self.expected_starts_with):
            score = 0
            reasons.append(f"did not start with {self.expected_starts_with!r}")
        for s in self.must_contain:
            if s not in response:
                score = 0
                reasons.append(f"missing required substring {s!r}")
        return {"score": score, "rationale": "; ".join(reasons) if reasons else "pass"}


class LLMJudge:
    """LLM-as-judge wrapper.

    In production this calls a Claude API endpoint. In the book the
    real call is gated behind ``ANTHROPIC_API_KEY``; without it, the
    judge falls back to a deterministic stub so notebooks can run
    offline. The stub returns ``-1`` to indicate no judgment.
    """

    def __init__(self, model: str = "claude-opus-4-6") -> None:
        self.model = model

    def __call__(self, response: str, reference: str = "", criterion: str = "helpful and correct") -> dict[str, Any]:
        # Real implementation would call the Anthropic API here; we keep
        # the chapter offline-runnable.
        return {
            "score": -1,
            "rationale": "LLM judge requires ANTHROPIC_API_KEY; returning stub.",
            "model": self.model,
        }
