"""Tests for Chapter 28 governance."""

import json
from pathlib import Path

from lm_from_scratch.governance.audit import AuditLogger, audit_record_schema
from lm_from_scratch.governance.grounding import GroundingCheck
from lm_from_scratch.governance.harness import GovernanceHarness
from lm_from_scratch.governance.policies import InputPolicy, OutputPolicy, PolicyResult
from lm_from_scratch.governance.validators import contains_pii


class _StubEngine:
    def generate(self, prompt: str, **kw):
        return f"response to: {prompt}"


def test_input_policy_blocks_pattern() -> None:
    pol = InputPolicy(blocked_patterns=["password"])
    r = pol.validate("Tell me the password")
    assert r.status == "block"
    assert any("password" in reason for reason in r.reasons)


def test_input_policy_passes_clean() -> None:
    pol = InputPolicy(blocked_patterns=["password"])
    assert pol.validate("hello").status == "pass"


def test_output_policy_blocks_long() -> None:
    pol = OutputPolicy(max_length=5)
    r = pol.validate("this is too long")
    assert r.status == "block"


def test_grounding_score_self_is_one() -> None:
    g = GroundingCheck(ngram=3)
    s = g.score("the quick brown fox jumps", "the quick brown fox jumps")
    assert s == 1.0


def test_grounding_score_disjoint() -> None:
    g = GroundingCheck(ngram=3)
    s = g.score("the quick brown fox", "completely unrelated content here forever")
    assert s == 0.0


def test_validators_finds_ssn() -> None:
    has, found = contains_pii("My SSN is 123-45-6789.")
    assert has and "ssn" in found


def test_audit_logger_writes_jsonl(tmp_path: Path) -> None:
    log = AuditLogger(tmp_path / "audit.jsonl")
    ip = PolicyResult(status="pass")
    op = PolicyResult(status="pass")
    rec = log.log(
        request_id="r1", prompt="hi", model_checkpoint="ckpt",
        tokenizer_hash="t", generation_config={"temperature": 0.5},
        input_result=ip, output_result=op,
        grounding_score=None, final_status="pass", latency_ms=12.3,
    )
    assert "timestamp" in rec
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["final_status"] == "pass"


def test_harness_blocks_bad_input(tmp_path: Path) -> None:
    log = AuditLogger(tmp_path / "audit.jsonl")
    harness = GovernanceHarness(
        engine=_StubEngine(),
        input_policy=InputPolicy(blocked_patterns=["forbidden"]),
        audit_logger=log,
    )
    out = harness.generate("forbidden request")
    assert out["status"] == "blocked_input"


def test_harness_passes_clean_input(tmp_path: Path) -> None:
    log = AuditLogger(tmp_path / "audit.jsonl")
    harness = GovernanceHarness(engine=_StubEngine(), audit_logger=log)
    out = harness.generate("hello")
    assert out["status"] == "pass"


def test_audit_schema_has_expected_keys() -> None:
    schema = audit_record_schema()
    assert "timestamp" in schema
    assert "request_id" in schema
    assert "final_status" in schema
