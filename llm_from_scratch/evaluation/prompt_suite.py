"""Prompt suite for behavioral evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Prompt:
    prompt_id: str
    prompt: str
    task_type: str = "open"
    difficulty: str = "medium"
    risk_type: str = ""
    expected_behavior: str = ""
    reference_answer: str = ""
    tags: list[str] = field(default_factory=list)


class PromptSuite:
    def __init__(self, prompts: list[Prompt] | None = None) -> None:
        self.prompts = list(prompts) if prompts else []

    @classmethod
    def from_csv(cls, path: str | Path) -> "PromptSuite":
        prompts: list[Prompt] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tags = row.get("tags", "")
                tag_list = [t.strip() for t in tags.split(";") if t.strip()] if tags else []
                prompts.append(Prompt(
                    prompt_id=row["prompt_id"], prompt=row["prompt"],
                    task_type=row.get("task_type", "open"),
                    difficulty=row.get("difficulty", "medium"),
                    risk_type=row.get("risk_type", ""),
                    expected_behavior=row.get("expected_behavior", ""),
                    reference_answer=row.get("reference_answer", ""),
                    tags=tag_list,
                ))
        return cls(prompts)

    def to_jsonl(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for p in self.prompts:
                f.write(json.dumps(asdict(p)) + "\n")

    def run(self, generate_fn, **gen_kwargs) -> list[dict[str, Any]]:
        """Apply ``generate_fn(prompt_str)`` to each prompt; return results."""
        out = []
        for p in self.prompts:
            try:
                response = generate_fn(p.prompt, **gen_kwargs)
            except Exception as e:  # pragma: no cover
                response = f"<error: {e}>"
            out.append({**asdict(p), "response": response})
        return out

    def factor_balance(self) -> dict[str, dict[str, int]]:
        from collections import Counter
        c_task = Counter(p.task_type for p in self.prompts)
        c_diff = Counter(p.difficulty for p in self.prompts)
        c_risk = Counter(p.risk_type for p in self.prompts)
        return {"task_type": dict(c_task), "difficulty": dict(c_diff), "risk_type": dict(c_risk)}
