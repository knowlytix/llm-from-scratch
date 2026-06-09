"""Failure-mode probes: hallucination, prompt sensitivity, bias and refusal."""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from llm_from_scratch.generation.chat import generate_text


@torch.no_grad()
def hallucination_check(model, tokenizer, qa_pairs: list[tuple[str, str]], **gen_kwargs) -> dict[str, Any]:
    """Run questions whose answers are unlikely in the model's training. Return generations."""
    results = []
    for q, expected in qa_pairs:
        out = generate_text(model, tokenizer, q, max_new_tokens=30, temperature=0.0, **gen_kwargs)
        # Heuristic: model says expected fact = correct, else hallucinated
        correct = expected.lower() in out.lower()
        results.append({"q": q, "expected": expected, "out": out, "correct": correct})
    return {"results": results,
            "correct_fraction": sum(1 for r in results if r["correct"]) / max(1, len(results))}


@torch.no_grad()
def prompt_sensitivity(model, tokenizer, prompts: list[str], perturbations: list[callable], **gen_kwargs) -> dict[str, Any]:
    """For each prompt, generate under several perturbations and count how many distinct outputs result."""
    rows = []
    for p in prompts:
        outs = set()
        for fn in perturbations:
            perturbed = fn(p)
            out = generate_text(model, tokenizer, perturbed, max_new_tokens=10, temperature=0.0, **gen_kwargs)
            outs.add(out[len(perturbed):].strip())
        rows.append({"prompt": p, "distinct_outputs": len(outs)})
    return {"results": rows,
            "mean_distinct": sum(r["distinct_outputs"] for r in rows) / max(1, len(rows))}


@torch.no_grad()
def template_bias_test(model, tokenizer, template: str, attribute_pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Fill ``template`` with attribute pairs (e.g. (man, woman)) and measure next-token log-prob differences."""
    device = next(model.parameters()).device
    deltas = []
    for a, b in attribute_pairs:
        p_a = template.format(a)
        p_b = template.format(b)
        ids_a = tokenizer.encode(p_a); ids_b = tokenizer.encode(p_b)
        x_a = torch.tensor(ids_a, dtype=torch.long, device=device).unsqueeze(0)
        x_b = torch.tensor(ids_b, dtype=torch.long, device=device).unsqueeze(0)
        logits_a, _ = model(x_a); logits_b, _ = model(x_b)
        if isinstance(logits_a, tuple): logits_a = logits_a[0]
        if isinstance(logits_b, tuple): logits_b = logits_b[0]
        # Difference in entropy as a quick scalar proxy.
        ent_a = -(logits_a[0, -1].softmax(-1) * logits_a[0, -1].log_softmax(-1)).sum().item()
        ent_b = -(logits_b[0, -1].softmax(-1) * logits_b[0, -1].log_softmax(-1)).sum().item()
        deltas.append({"a": a, "b": b, "entropy_a": ent_a, "entropy_b": ent_b, "delta": ent_a - ent_b})
    return {"deltas": deltas, "mean_abs_delta": sum(abs(d["delta"]) for d in deltas) / max(1, len(deltas))}


@torch.no_grad()
def refusal_probe(model, tokenizer, sensitive_prompts: list[str], refusal_keywords: list[str]) -> dict[str, Any]:
    """Fraction of sensitive prompts where the model's output contains a refusal keyword."""
    refused = 0
    for p in sensitive_prompts:
        out = generate_text(model, tokenizer, p, max_new_tokens=30, temperature=0.0)
        if any(k.lower() in out.lower() for k in refusal_keywords):
            refused += 1
    return {"refusal_rate": refused / max(1, len(sensitive_prompts)),
            "num_prompts": len(sensitive_prompts)}
