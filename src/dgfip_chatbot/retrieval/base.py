"""The shared retrieval contract — every retriever (dense, BM25, hybrid) implements it."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RetrievalResult:
    """One ranked fiche returned for a query."""

    fiche_id: int
    score: float  # retriever-specific: cosine [-1,1] (dense), BM25 score >=0 (bm25),
    # or fused [0,1] (hybrid). Only comparable within one retriever.
    titre: str
    url: str
    snippet: str  # the best-matching chunk's text (for display / grounding)


class Retriever(Protocol):
    """A query -> ranked fiches function. Phase 2 ships `DenseRetriever`; Phase 3 adds
    BM25 and hybrid behind this same interface so the eval loop can swap them."""

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]: ...
