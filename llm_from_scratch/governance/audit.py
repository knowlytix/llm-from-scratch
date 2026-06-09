"""Append-only audit logger for governance harness."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


def audit_record_schema() -> dict[str, Any]:
    return {
        "timestamp": "string (ISO 8601)",
        "request_id": "string",
        "prompt_hash": "string (sha256, first 16 hex)",
        "model_checkpoint": "string",
        "tokenizer_hash": "string",
        "generation_config": "dict",
        "input_policy_result": "dict",
        "output_policy_result": "dict",
        "grounding_score": "float | None",
        "final_status": "string",
        "latency_ms": "float",
    }


class AuditLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _hash(self, s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    def log(self, *,
            request_id: str,
            prompt: str,
            model_checkpoint: str,
            tokenizer_hash: str,
            generation_config: dict,
            input_result,
            output_result,
            grounding_score: float | None,
            final_status: str,
            latency_ms: float) -> dict:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": request_id,
            "prompt_hash": self._hash(prompt),
            "model_checkpoint": model_checkpoint,
            "tokenizer_hash": tokenizer_hash,
            "generation_config": generation_config,
            "input_policy_result": asdict(input_result) if hasattr(input_result, "__dataclass_fields__") else input_result.__dict__,
            "output_policy_result": asdict(output_result) if hasattr(output_result, "__dataclass_fields__") else output_result.__dict__,
            "grounding_score": grounding_score,
            "final_status": final_status,
            "latency_ms": latency_ms,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record
