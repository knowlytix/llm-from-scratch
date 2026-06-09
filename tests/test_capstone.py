"""Tests for Chapter 29 capstone helpers."""

from pathlib import Path

from llm_from_scratch.capstone.manifest import ArtifactManifest
from llm_from_scratch.capstone.pipeline import run_pipeline


def test_pipeline_skeleton_returns_stages() -> None:
    out = run_pipeline({"a": 1})
    assert "stages" in out
    assert len(out["stages"]) > 5


def test_manifest_fingerprints_environment_and_artifacts(tmp_path: Path) -> None:
    m = ArtifactManifest()
    m.fingerprint_environment()
    assert m.env is not None
    f = tmp_path / "x.txt"
    f.write_text("hello")
    m.add("file_x", f)
    assert len(m.artifacts["file_x"]) == 16
    m.save(tmp_path / "manifest.json")
    assert (tmp_path / "manifest.json").exists()
