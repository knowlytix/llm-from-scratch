"""Text cleaning primitives.

Each function is idempotent (applying it twice gives the same result as
applying it once). Idempotence is the practical sanity property; it lets us
re-run cleaning over a corpus without worrying about double-application.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"[ \t ]+")
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces and tabs into single spaces, normalize newlines."""
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTIPLE_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def strip_control_chars(text: str) -> str:
    """Remove ASCII control characters except for tab and newline.

    Unicode "control" general category characters are also stripped. The
    function intentionally preserves newline so paragraph structure survives.
    """
    return "".join(
        ch
        for ch in text
        if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C"
    )


def clean_text(text: str) -> str:
    """Composition of the cleaning steps used throughout the book."""
    return normalize_whitespace(strip_control_chars(text))
