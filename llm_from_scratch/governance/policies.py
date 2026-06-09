"""Input and output policies for the governance harness."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PolicyResult:
    status: str  # "pass" | "flag" | "block"
    reasons: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class InputPolicy:
    def __init__(self, *, blocked_patterns: list[str] | None = None,
                 max_prompt_length: int = 4096,
                 allowed_tasks: list[str] | None = None) -> None:
        self.blocked_patterns = [re.compile(p, re.IGNORECASE) for p in (blocked_patterns or [])]
        self.max_prompt_length = max_prompt_length
        self.allowed_tasks = set(allowed_tasks) if allowed_tasks else None

    def validate(self, prompt: str, task_type: str | None = None) -> PolicyResult:
        reasons = []
        if len(prompt) > self.max_prompt_length:
            reasons.append(f"prompt exceeds max length ({len(prompt)} > {self.max_prompt_length})")
        for pat in self.blocked_patterns:
            if pat.search(prompt):
                reasons.append(f"matched blocked pattern: {pat.pattern}")
        if self.allowed_tasks is not None and task_type is not None and task_type not in self.allowed_tasks:
            reasons.append(f"task {task_type!r} not allowed")
        status = "block" if reasons else "pass"
        return PolicyResult(status=status, reasons=reasons)


class OutputPolicy:
    def __init__(self, *, blocked_patterns: list[str] | None = None,
                 max_length: int = 4096,
                 require_citations: bool = False) -> None:
        self.blocked_patterns = [re.compile(p, re.IGNORECASE) for p in (blocked_patterns or [])]
        self.max_length = max_length
        self.require_citations = require_citations

    def validate(self, response: str) -> PolicyResult:
        reasons = []
        if len(response) > self.max_length:
            reasons.append(f"response exceeds max length")
        for pat in self.blocked_patterns:
            if pat.search(response):
                reasons.append(f"matched blocked pattern: {pat.pattern}")
        if self.require_citations and not re.search(r"\[\d+\]|\([A-Za-z]+ \d{4}\)", response):
            reasons.append("citations required but none found")
        status = "block" if reasons else "pass"
        return PolicyResult(status=status, reasons=reasons)
