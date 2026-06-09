"""Pattern validators used by InputPolicy and OutputPolicy."""

from __future__ import annotations

import re

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")


def contains_pii(text: str) -> tuple[bool, list[str]]:
    """Return (has_pii, list_of_pattern_names) for a basic regex sweep."""
    found = []
    if _SSN_RE.search(text):
        found.append("ssn")
    if _EMAIL_RE.search(text):
        found.append("email")
    if _PHONE_RE.search(text):
        found.append("phone")
    return (len(found) > 0, found)
