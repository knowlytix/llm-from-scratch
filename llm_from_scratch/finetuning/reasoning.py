"""Chain-of-thought formatting helpers."""

from __future__ import annotations


def format_cot(prompt: str, reasoning: str, answer: str, template: str = "think_answer") -> str:
    """Format a chain-of-thought training example."""
    if template == "think_answer":
        return f"{prompt}\nThink: {reasoning}\nAnswer: {answer}"
    if template == "scratchpad":
        return f"{prompt}\n<scratchpad>{reasoning}</scratchpad>\n{answer}"
    raise ValueError(f"unknown template: {template}")


def extract_answer(generation: str, marker: str = "Answer:") -> str:
    """Pull the substring after the final ``marker`` from a generation."""
    idx = generation.rfind(marker)
    if idx < 0:
        return generation.strip()
    return generation[idx + len(marker):].strip()
