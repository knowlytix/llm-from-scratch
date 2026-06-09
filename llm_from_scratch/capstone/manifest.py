"""Artifact manifest: fingerprint every file the pipeline produced."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ArtifactManifest:
    artifacts: dict[str, str] = field(default_factory=dict)
    config: dict | None = None
    env: dict | None = None

    def add(self, name: str, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            self.artifacts[name] = "missing"
            return
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        self.artifacts[name] = h

    def fingerprint_environment(self) -> None:
        import torch
        self.env = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "platform": platform.platform(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "artifacts": self.artifacts,
            "config": self.config,
            "env": self.env,
        }, indent=2))
