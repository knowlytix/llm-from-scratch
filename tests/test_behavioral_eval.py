"""Tests for Chapter 22 behavioral evaluation."""

import csv

import numpy as np

from lm_from_scratch.evaluation.judges import HeuristicJudge, LLMJudge, ReferenceMatchJudge
from lm_from_scratch.evaluation.prompt_suite import Prompt, PromptSuite
from lm_from_scratch.evaluation.significance import bootstrap_ci, paired_bootstrap


def test_reference_match_judge() -> None:
    j = ReferenceMatchJudge()
    assert j("the answer is yes", "yes")["score"] == 1
    assert j("the answer is no", "yes")["score"] == 0


def test_heuristic_judge_starts_with() -> None:
    j = HeuristicJudge(expected_starts_with="The")
    assert j("The cat sat")["score"] == 1
    assert j("A cat sat")["score"] == 0


def test_llm_judge_stub_no_api_key() -> None:
    j = LLMJudge()
    out = j("response")
    assert out["score"] == -1


def test_prompt_suite_from_csv_and_balance(tmp_path) -> None:
    p = tmp_path / "prompts.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_id", "prompt", "task_type", "difficulty", "risk_type", "tags"])
        w.writerow(["a", "hi", "open", "easy", "low", "x;y"])
        w.writerow(["b", "?", "qa", "hard", "low", ""])
    suite = PromptSuite.from_csv(p)
    assert len(suite.prompts) == 2
    bal = suite.factor_balance()
    assert bal["task_type"] == {"open": 1, "qa": 1}


def test_bootstrap_ci() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=5.0, scale=1.0, size=200)
    mean, low, high = bootstrap_ci(samples)
    assert 4.5 < mean < 5.5
    assert low < mean < high


def test_paired_bootstrap_zero_diff() -> None:
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0]
    out = paired_bootstrap(a, b)
    assert abs(out["mean_diff"]) < 1e-9
