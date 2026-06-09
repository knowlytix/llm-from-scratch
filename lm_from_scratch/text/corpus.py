"""A simple corpus container with a small functional API.

A corpus is, for our purposes, a list of documents. We avoid more complex
representations (datasets, streaming) at this stage so the reader can inspect
the whole thing in memory.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any


class TextCorpus:
    """A list of documents with a small functional API.

    Parameters
    ----------
    documents:
        A list of strings, one per document.
    metadata:
        Optional dict of corpus-level metadata (source, license, dates, etc).
    """

    def __init__(self, documents: list[str], metadata: dict[str, Any] | None = None):
        self._documents = list(documents)
        self.metadata: dict[str, Any] = dict(metadata) if metadata else {}

    # --- Construction --------------------------------------------------

    @classmethod
    def from_folder(cls, path: str | Path, glob: str = "*.txt") -> "TextCorpus":
        """Load a corpus by reading every file matching ``glob`` under ``path``."""
        root = Path(path)
        docs = [p.read_text(encoding="utf-8") for p in sorted(root.glob(glob))]
        return cls(docs, metadata={"source": str(root), "glob": glob})

    @classmethod
    def from_files(cls, paths: Iterable[str | Path]) -> "TextCorpus":
        docs = [Path(p).read_text(encoding="utf-8") for p in paths]
        return cls(docs)

    # --- Inspection ----------------------------------------------------

    @property
    def documents(self) -> list[str]:
        return self._documents

    def __len__(self) -> int:
        return len(self._documents)

    def __iter__(self) -> Iterator[str]:
        return iter(self._documents)

    def __getitem__(self, idx: int) -> str:
        return self._documents[idx]

    def summary(self) -> dict[str, Any]:
        """A compact dict summary: counts and average document length."""
        lengths = [len(d) for d in self._documents]
        return {
            "num_documents": len(self._documents),
            "total_chars": sum(lengths),
            "avg_doc_chars": sum(lengths) / max(1, len(lengths)),
            "min_doc_chars": min(lengths) if lengths else 0,
            "max_doc_chars": max(lengths) if lengths else 0,
            **self.metadata,
        }

    # --- Functional transforms -----------------------------------------

    def map(self, fn: Callable[[str], str]) -> "TextCorpus":
        return TextCorpus([fn(d) for d in self._documents], metadata=self.metadata)

    def filter(self, predicate: Callable[[str], bool]) -> "TextCorpus":
        return TextCorpus([d for d in self._documents if predicate(d)], metadata=self.metadata)
