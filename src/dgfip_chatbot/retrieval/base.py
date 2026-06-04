"""The shared retrieval contract — every retriever (dense, BM25, hybrid) implements it."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RetrievalResult:
    """One ranked fiche returned for a query."""

    fiche_id: int
    score: float  # cosine in [-1, 1] for the dense retriever
    titre: str
    url: str
    snippet: str  # the best-matching chunk's text (for display / grounding)


class Retriever(Protocol):
    """A query -> ranked fiches function. Phase 2 ships `DenseRetriever`; Phase 3 adds
    BM25 and hybrid behind this same interface so the eval loop can swap them."""

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]: ...
